"""线条渲染。

线宽与虚线间距都以**输出像素**为单位（与 PS 一致）。绘制用 Pillow，
重叠线条的 alpha 会正常叠加（不透明度 > 0 时交叠处更深）。

虚线按**弧长**采样，因此拐角处不会出现间距忽宽忽窄的问题；同一条线内
虚线相位连续，视觉上均匀。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import LineStyle, OutputMode, RGB
from .geometry import Line

__all__ = ["draw_lines_rgba", "composite", "render"]

#: 虚线节距的**内部固定值**（像素）。界面不再暴露"实段长/间隔"。
_DASH_LEN = 24.0
_DASH_GAP = 14.0

#: 默认超采样倍数（抗锯齿）。实测：线宽 10px 的浅角度边，
#: 硬边渲染时有 8%~36% 的列出现 ≥0.9px 的整像素跳变（就是肉眼看到的"台阶"），
#: 4x 超采样后这个比例降到 **0%**，最大跳变从 1.000px 降到 0.338px。
SS_DEFAULT = 4
#: 超采样画布的像素上限（约 72M 像素 ≈ 288MB RGBA）。
#: 线条包围盒很大时自动把倍率降到 2（而不是切条带——切条带会让线宽取整
#: 在条带边界处不一致）。
_SS_MAX_PIXELS = 72_000_000


def _scale_style(style: LineStyle, factor: float) -> LineStyle:
    """按倍数缩放线条样式（宽度取整、其余不变）。"""
    s = style.normalized()
    return LineStyle(
        enabled=s.enabled,
        style=s.style,
        color=s.color,
        width=max(1, int(round(s.width * factor))),
        alpha=s.alpha,
    )


def _resample(points: np.ndarray, max_points: int = 4096) -> list[tuple[float, float]]:
    """把折线转成 Pillow 需要的点列表；过长的折线等距抽稀以控制绘制开销。"""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n <= max_points:
        return [(float(x), float(y)) for x, y in pts]
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    idx = np.unique(idx)
    return [(float(pts[i, 0]), float(pts[i, 1])) for i in idx]


def _iter_dashes(
    pts: list[tuple[float, float]],
    style: str,
    scale: float = 1.0,
) -> Iterable[list[tuple[float, float]]]:
    """把折线切成虚线段。

    节距是**内部固定值**（按线宽缩放），界面不再暴露"实段长/间隔"——
    实线时这两个参数本来就不生效。

    Args:
        pts: 折线点。
        style: ``"dashed"`` 或 ``"dashdot"``。
        scale: 线宽缩放系数（预览时 <1），用于让虚线节距跟着一起缩。

    Yields:
        每个虚线段（至少 2 个点）。
    """
    dash = max(1.0, _DASH_LEN * float(scale))
    gap = max(1.0, _DASH_GAP * float(scale))
    if style == "dashdot":
        # 长划 - 间隔 - 点 - 间隔
        pattern = [(dash, True), (gap, False), (max(1.0, dash * 0.18), True), (gap, False)]
    else:
        pattern = [(dash, True), (gap, False)]
    period = sum(p[0] for p in pattern)
    if period <= 0:
        yield pts
        return

    # 每条线都从实线段开头，保证同一条线的虚线相位稳定
    pi = 0
    remaining_in_piece = pattern[pi][0]
    drawing = pattern[pi][1]

    current: list[tuple[float, float]] = []

    def _emit() -> list[tuple[float, float]] | None:
        if len(current) >= 2:
            return list(current)
        return None

    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len <= 1e-9:
            continue
        # 若上一段结束在绘制状态，本段起点要接上，避免断开
        if drawing and not current:
            current = [(x0, y0)]
        consumed = 0.0
        while consumed < seg_len:
            step = min(remaining_in_piece, seg_len - consumed)
            t0 = consumed / seg_len
            t1 = (consumed + step) / seg_len
            p0 = (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0)
            p1 = (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)
            if drawing:
                if not current:
                    current = [p0]
                current.append(p1)
            else:
                seg = _emit()
                if seg is not None:
                    yield seg
                current = []
            consumed += step
            remaining_in_piece -= step
            if remaining_in_piece <= 1e-9:
                pi = (pi + 1) % len(pattern)
                remaining_in_piece = pattern[pi][0]
                drawing = pattern[pi][1]
    seg = _emit()
    if seg is not None:
        yield seg


def lines_bbox(
    lines: list[Line], styles: dict[str, LineStyle], size: tuple[int, int], pad_extra: int = 2
) -> tuple[int, int, int, int] | None:
    """所有可见线条的并集包围盒 ``(x0, y0, x1, y1)``（右开区间）。

    只在这个区域内绘制与合成，可把 4928x3264 全图操作降为局部操作。
    """
    w, h = int(size[0]), int(size[1])
    xs_min = ys_min = np.inf
    xs_max = ys_max = -np.inf
    pad = pad_extra
    for kind in ("outer", "inner"):
        st = styles.get(kind)
        if st is None or not st.enabled or st.alpha <= 0:
            continue
        pad = max(pad, int(st.width) // 2 + 2)
    for ln in lines:
        st = styles.get(ln.kind)
        if st is None or not st.enabled or st.alpha <= 0 or ln.n == 0:
            continue
        x0, y0, x1, y1 = ln.bbox()
        xs_min = min(xs_min, x0)
        ys_min = min(ys_min, y0)
        xs_max = max(xs_max, x1)
        ys_max = max(ys_max, y1)
    if not np.isfinite(xs_min):
        return None
    ix0 = max(0, int(math.floor(xs_min)) - pad)
    iy0 = max(0, int(math.floor(ys_min)) - pad)
    ix1 = min(w, int(math.ceil(xs_max)) + pad + 1)
    iy1 = min(h, int(math.ceil(ys_max)) + pad + 1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1, iy1


def draw_lines_rgba(
    size: tuple[int, int],
    lines: list[Line],
    styles: dict[str, LineStyle],
    *,
    phase_mode: str = "continuous",
    clip: tuple[int, int, int, int] | None = None,
    width_scale: float = 1.0,
    supersample: int = 1,
) -> np.ndarray:
    """把线条画到透明 RGBA 图层上。

    Args:
        size: ``(width, height)`` —— **画布尺寸**。若给了 ``clip``，
            画布尺寸即为裁剪区尺寸，线条坐标会自动平移。
        lines: 折线列表（坐标为整幅图像坐标）。
        styles: ``{"outer": LineStyle, "inner": LineStyle}``。
        phase_mode: 保留参数（虚线节距已固定，不再做跨线相位对齐）。
        clip: 可选裁剪区 ``(x0, y0, x1, y1)``，右开区间。
        width_scale: 虚线节距的缩放系数（预览缩小时让节距同比缩小）。
        supersample: 抗锯齿的超采样倍数（1 = 关闭）。>1 时先按该倍数放大绘制，
            再面积平均缩回 —— **几何不变，只把硬边变成 1px 的柔和过渡**，
            这正是"浅角度长边出现整像素台阶"的解药。线宽/虚线节距同步缩放，
            缩回后宽度不变。

    Returns:
        ``(H, W, 4)`` uint8，RGB 为线条颜色，A 为覆盖度。
    """
    ss = max(1, int(supersample))
    if ss == 1:
        return _draw_lines_rgba_once(size, lines, styles, clip=clip, width_scale=width_scale)

    w, h = int(size[0]), int(size[1])
    # 内存保护：超采样画布 = w*h*ss^2 个像素。包围盒很大时自动降倍数
    # （宁可少一点抗锯齿，也不能申请十几 GB 内存）。**不切条带**——
    # 切条带会让"线宽取整"在条带边界处不一致，整体渲染才能保证逐像素一致。
    while ss > 2 and w * h * ss * ss > _SS_MAX_PIXELS:
        ss //= 2
    if ss == 1:
        return _draw_lines_rgba_once(size, lines, styles, clip=clip, width_scale=width_scale)

    ox = oy = 0
    if clip is not None:
        ox, oy = int(clip[0]), int(clip[1])
    scaled_styles = {k: _scale_style(v, ss) for k, v in styles.items()}
    # 线条坐标一并放大 ss 倍：画布、线条、clip 原点处在同一尺度里
    scaled_lines = [
        Line(ln.points * float(ss), ln.kind, ln.crystal_id, ln.layer_id, dict(ln.attrs))
        for ln in lines
    ]
    sub = _draw_lines_rgba_once(
        (w * ss, h * ss),
        scaled_lines,
        scaled_styles,
        clip=(ox * ss, oy * ss, (ox + w) * ss, (oy + h) * ss),
        width_scale=width_scale * ss,
    )
    return cv2.resize(sub, (w, h), interpolation=cv2.INTER_AREA)


def _draw_lines_rgba_once(
    size: tuple[int, int],
    lines: list[Line],
    styles: dict[str, LineStyle],
    *,
    clip: tuple[int, int, int, int] | None = None,
    width_scale: float = 1.0,
) -> np.ndarray:
    """单次绘制（不做超采样）。画布坐标与线条坐标同尺度。"""
    w, h = int(size[0]), int(size[1])
    ox = oy = 0
    if clip is not None:
        ox, oy = int(clip[0]), int(clip[1])
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    def shift(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if ox == 0 and oy == 0:
            return pts
        return [(x - ox, y - oy) for x, y in pts]

    for kind in ("outer", "inner"):
        style = styles.get(kind)
        if style is None or not style.enabled or style.alpha <= 0:
            continue
        style = style.normalized()
        color: RGB = style.color
        rgba = (int(color[0]), int(color[1]), int(color[2]), int(style.alpha))
        for ln in lines:
            if ln.kind != kind or ln.n < 2:
                continue
            pts = _resample(ln.points)
            if clip is not None:
                x0, y0, x1, y1 = ln.bbox()
                m = int(style.width) // 2 + 2
                if x1 < ox - m or x0 > ox + w + m or y1 < oy - m or y0 > oy + h + m:
                    continue
            if style.style == "solid":
                pts_draw = _closed_pts(pts) if ln.closed else pts
                draw.line(shift(pts_draw), fill=rgba, width=int(style.width), joint="curve")
            else:
                pts_closed = _closed_pts(pts) if ln.closed else pts
                for seg in _iter_dashes(pts_closed, style.style, width_scale):
                    draw.line(shift(seg), fill=rgba, width=int(style.width), joint="curve")
    return np.asarray(canvas, dtype=np.uint8)


def _closed_pts(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """闭合折线补上"末点 -> 首点"那一段。

    ``ImageDraw.line`` 只连**相邻**点，不会自动闭合。以前闭合轮廓缺这一段，
    表现就是"线描到一半消失、根本没闭合"（实测缺口 1440px / 1543px）。
    """
    if len(pts) >= 3 and (pts[0][0] != pts[-1][0] or pts[0][1] != pts[-1][1]):
        return pts + [pts[0]]
    return pts


def _polyline_pts_length(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        total += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    return total


def composite(
    base_bgr: np.ndarray | None,
    rgba: np.ndarray,
    mode: OutputMode,
    reference_color: RGB = (255, 255, 255),
) -> tuple[np.ndarray, np.ndarray | None]:
    """把线条图层与原图合成。

    Args:
        base_bgr: 原图 ``(H, W, 3)`` uint8；``transparent`` 模式可为 None。
        rgba: 线条图层 ``(H, W, 4)``。
        mode: ``"overlay"``（原图 + 线条）或 ``"transparent"``（透明底 + 线条）。
        reference_color: 透明模式下供**预览**使用的底色（导出时不会被写入）。

    Returns:
        ``(bgr, alpha)``。``alpha`` 仅在 ``transparent`` 模式返回，导出 PNG 用。
    """
    h, w = rgba.shape[:2]
    if base_bgr is not None and (base_bgr.shape[0] != h or base_bgr.shape[1] != w):
        raise ValueError(
            f"底图尺寸 {base_bgr.shape[1]}x{base_bgr.shape[0]} 与线条图层 {w}x{h} 不一致"
        )

    if mode == "transparent":
        # 保留线条自身的 alpha 作为输出 alpha；RGB 填线条颜色
        alpha = rgba[:, :, 3].copy()
        rgb = rgba[:, :, :3]
        # 未覆盖处填参考色，便于看图（导出时 alpha=0，颜色无关紧要）
        ref = np.array(reference_color, dtype=np.uint8)
        rgb = np.where(alpha[..., None] > 0, rgb, ref)
        bgr = rgb[:, :, ::-1].copy()
        return np.ascontiguousarray(bgr), np.ascontiguousarray(alpha)

    base = Image.fromarray(base_bgr[:, :, ::-1], mode="RGB").convert("RGBA")
    layer = Image.fromarray(rgba, mode="RGBA")
    merged = Image.alpha_composite(base, layer)
    arr = np.asarray(merged, dtype=np.uint8)[:, :, :3]
    return np.ascontiguousarray(arr[:, :, ::-1]), None


def render(
    size: tuple[int, int],
    lines: list[Line],
    styles: dict[str, LineStyle],
    base_bgr: np.ndarray | None,
    mode: OutputMode,
    reference_color: RGB = (255, 255, 255),
    supersample: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """一步完成"画线 + 合成"。

    内部只在**线条包围盒**内绘制与合成，避免整幅图级别的内存拷贝，
    因此可以做到交互式实时刷新。

    Args:
        supersample: 抗锯齿超采样倍数（1 = 关闭）。导出建议用
            :data:`SS_DEFAULT`；预览请留 1（超采样是 16 倍像素代价）。
    """
    w, h = int(size[0]), int(size[1])
    box = lines_bbox(lines, styles, (w, h))
    if box is None:
        if mode == "transparent":
            ref = np.array(reference_color, dtype=np.uint8)
            bgr = np.empty((h, w, 3), dtype=np.uint8)
            bgr[:] = ref[::-1]
            return bgr, np.zeros((h, w), dtype=np.uint8)
        if base_bgr is None:
            raise ValueError("overlay 模式必须提供底图")
        return np.ascontiguousarray(base_bgr.copy()), None

    x0, y0, x1, y1 = box
    sub_w, sub_h = x1 - x0, y1 - y0
    rgba_sub = draw_lines_rgba(
        (sub_w, sub_h), lines, styles, clip=box, supersample=supersample
    )

    if mode == "transparent":
        bgr = np.empty((h, w, 3), dtype=np.uint8)
        ref = np.array(reference_color, dtype=np.uint8)
        bgr[:] = ref[::-1]
        alpha = np.zeros((h, w), dtype=np.uint8)
        bgr[y0:y1, x0:x1] = rgba_sub[:, :, :3][:, :, ::-1]
        alpha[y0:y1, x0:x1] = rgba_sub[:, :, 3]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(alpha)

    if base_bgr is None:
        raise ValueError("overlay 模式必须提供底图")
    out = np.ascontiguousarray(base_bgr.copy())
    base_sub = Image.fromarray(out[y0:y1, x0:x1][:, :, ::-1], mode="RGB").convert("RGBA")
    merged = Image.alpha_composite(base_sub, Image.fromarray(rgba_sub, mode="RGBA"))
    out[y0:y1, x0:x1] = np.asarray(merged, dtype=np.uint8)[:, :, :3][:, :, ::-1]
    return out, None
