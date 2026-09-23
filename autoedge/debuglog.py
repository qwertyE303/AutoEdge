"""诊断日志：每次点「预览」都把全部坐标与判定数据写进一份文本文件。

用途：使用者的界面出现坐标错位之类的问题时，不需要截图，
直接把生成的 log 文件发回来即可逐字核对。

写出的内容包括：

1. 图像层：原图尺寸、画布显示图尺寸、缩放、自洽性检查
2. 画布层：viewport 尺寸、sceneRect、m11（Qt 实际变换）
3. 每个 ROI 顶点：视口坐标 → 场景坐标 → 原图坐标（三级并列）
4. 每个提示点：同一套三级坐标 + 该点在原图上的灰度与到基底色的 ΔE
5. 分析参数与分割结果诊断
6. 一致性结论（哪些环节自洽、哪些不自洽）
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

__all__ = ["write_debug_log", "default_log_path"]

LOG_NAME = "debug log.txt"


def default_log_path() -> str:
    """日志路径：优先落在可执行文件/项目根目录下。"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, LOG_NAME)


def _fmt_pt(px: Any, py: Any) -> str:
    return f"({float(px):8.1f},{float(py):8.1f})"


def write_debug_log(
    path: str,
    *,
    image_size: tuple[int, int],
    image_path: str | None,
    canvas: Any,
    roi: Any,
    preset: Any,
    result: Any = None,
    extra_notes: list[str] | None = None,
    layers: list[tuple] | None = None,
) -> str:
    """写出诊断日志，返回实际写入的路径。

    Args:
        path: 目标文件路径。
        image_size: 原图 ``(宽, 高)``。
        image_path: 原图路径。
        canvas: :class:`~autoedge_gui.canvas.ImageCanvas`，用于取画布状态与换算。
        roi: :class:`~autoedge.roi.RoiSet`（当前图层的）。
        preset: :class:`~autoedge.config.Preset`（当前图层的）。
        result: 分析结果（可为 None）。
        extra_notes: 额外要写入的说明行。
        layers: 各图层摘要，每项为
            ``(名称, RoiSet, SegConfig, 线条数, 是否显示, 是否当前层)``。
    """
    lines: list[str] = []
    add = lines.append
    ow, oh = int(image_size[0]), int(image_size[1])

    add("=" * 88)
    add(f"AutoEdge 诊断日志    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    add("=" * 88)

    # ---------------- 1. 图像层 ----------------
    add("")
    add("【1】图像层")
    add(f"    原图路径        : {image_path}")
    add(f"    原图尺寸        : {ow} x {oh}")
    dw, dh = canvas._image_size
    add(f"    画布显示图尺寸  : {dw} x {dh}")
    add(f"    坐标缩放        : {canvas.image_to_display_scale():.5f}"
        f"   (显示图/原图 = {dw / max(1, ow):.5f})")
    ok_scale = abs(canvas.image_to_display_scale() - dw / max(1, ow)) < 1e-6
    add(f"    缩放自洽        : {'✓' if ok_scale else '★不一致★'}")

    # ---------------- 2. 画布层 ----------------
    add("")
    add("【2】画布层")
    vw, vh = canvas.viewport().width(), canvas.viewport().height()
    sr = canvas.sceneRect()
    m11 = canvas.transform().m11()
    fit = min(vw / max(1.0, sr.width()), vh / max(1.0, sr.height()))
    add(f"    viewport        : {vw} x {vh}")
    add(f"    sceneRect       : {sr.width():.0f} x {sr.height():.0f}")
    add(f"    m11 (Qt变换)    : {m11:.5f}")
    add(f"    fitInView 应有  : {fit:.5f}")
    add(f"    变换匹配        : {'✓' if abs(m11 - fit) < 0.02 else '★不一致（变换未随窗口更新）★'}")
    add(f"    场景尺寸 == 显示图尺寸 ? "
        f"{'✓' if abs(sr.width() - dw) < 1 and abs(sr.height() - dh) < 1 else '★否★'}")

    # ---------------- 3. ROI ----------------
    add("")
    add(f"【3】ROI（共 {len(roi.polygons)} 个框）")
    out_of_scene = 0
    if roi.polygons:
        for k, poly in enumerate(roi.polygons, start=1):
            add(f"    --- 第 {k} 个框，{len(poly)} 个顶点 ---")
            for j, (ix, iy) in enumerate(poly, start=1):
                sx, sy = canvas.to_scene(float(ix), float(iy))
                inside = 0 <= sx <= sr.width() and 0 <= sy <= sr.height()
                if not inside:
                    out_of_scene += 1
                add(
                    f"      顶点{j:<2d} 原图{_fmt_pt(ix, iy)}"
                    f" -> 场景{_fmt_pt(sx, sy)}"
                    f"  {'场景内✓' if inside else '★超出场景范围★'}"
                )
        allp = np.vstack(roi.polygons)
        bx0, by0 = float(allp[:, 0].min()), float(allp[:, 1].min())
        bx1, by1 = float(allp[:, 0].max()), float(allp[:, 1].max())
        s0x, s0y = canvas.to_scene(bx0, by0)
        s1x, s1y = canvas.to_scene(bx1, by1)
        add(f"    → 框范围  原图 x[{bx0:.0f},{bx1:.0f}] y[{by0:.0f},{by1:.0f}]")
        add(f"              屏幕 x[{s0x:.0f},{s1x:.0f}] y[{s0y:.0f},{s1y:.0f}]")
        add(f"              原图尺寸 {bx1 - bx0:.0f} x {by1 - by0:.0f} 像素")
        add(f"    越界顶点数      : {out_of_scene}"
            f"  {'✓' if out_of_scene == 0 else '★存在越界，说明绘制方向或缩放有误★'}")
    else:
        add("    （无，将使用全自动模式）")

    # ---------------- 4. 提示点 ----------------
    add("")
    pos = roi.positive_points()
    neg = roi.negative_points()
    add(f"【4】提示点（绿 {len(pos)} 个，红 {len(neg)} 个）")
    detail_p = (result.seg.info.get("pos_detail", []) if result and result.seg else [])
    detail_n = (result.seg.info.get("neg_detail", []) if result and result.seg else [])
    gray = _gray_of_image(image_path)

    def dump_hints(kind: str, pts: list[tuple[float, float]], details: list[dict]) -> None:
        if not pts:
            return
        add(f"    --- {kind} ---")
        for i, (ix, iy) in enumerate(pts, start=1):
            sx, sy = canvas.to_scene(ix, iy)
            g = _sample_gray(gray, ix, iy)
            de = details[i - 1]["de"] if i - 1 < len(details) else None
            add(
                f"      点{i:<2d} 原图{_fmt_pt(ix, iy)}"
                f" -> 场景{_fmt_pt(sx, sy)}"
                f"  该点灰度={g if g is not None else '—'}"
                f"  ΔE={f'{de:.2f}' if de is not None else '—'}"
            )

    dump_hints("绿点（这里是晶体）", pos, detail_p)
    dump_hints("红点（这里不是晶体）", neg, detail_n)

    # ---------------- 4b. 原始点击记录（最关键的核对依据） ----------------
    add("")
    add("【4b】原始点击记录（点击那一刻的三个坐标，用于定位换算在哪一步出错）")
    clicks = list(getattr(canvas, "_pending_clicks", []) or [])
    last = getattr(canvas, "_last_click_debug", {}) or {}
    if last:
        add("    --- 最后一次单点点击（提示点） ---")
        add(f"      视口坐标(鼠标在画布内的位置) = {last.get('viewport')}")
        add(f"      mapToScene 原始返回          = {last.get('scene_raw')}")
        add(f"      换算后的原图坐标             = {last.get('image')}")
        add(f"      当时 image_size={last.get('image_size')}  "
            f"original_size={last.get('original_size')}  "
            f"scale={last.get('scale')}  m11={last.get('m11')}")
    if clicks:
        add("    --- 框选时每个顶点 ---")
        for i, ck in enumerate(clicks, start=1):
            add(f"      顶点{i}: 视口={ck.get('viewport')}  "
                f"mapToScene={ck.get('scene_raw')}  -> 原图={ck.get('image')}")
    if not last and not clicks:
        add("    （本次预览前没有点击记录）")

    # ---------------- 5. 参数与结果 ----------------
    seg = preset.seg
    add("")
    add("【5】分析参数")
    add(f"    描边阈值 ΔE={seg.threshold}（越小越灵敏）  种子倍数={seg.seed_factor}"
        f"  最小面积={seg.min_area}px")
    add(f"    简化容差={seg.simplify_tolerance}  薄片合并={seg.thin_span}px")
    add(f"    光照补偿: 亮度={seg.brightness} 对比度={seg.contrast} 伽马={seg.gamma}")
    add(f"    输出模式={preset.output_mode}  线条={preset.outer.style}/{preset.outer.width}px "
        f"颜色={preset.outer.color}  不透明度={round(preset.outer.alpha / 255 * 100)}%")

    add("")
    add("【6】分割结果")
    if result is None or result.seg is None:
        add("    （无结果）")
    else:
        s = result.seg
        info = s.info
        add(f"    晶体 {s.count} 块，外轮廓 {len(result.outer_lines)} 条")
        add(f"    阈值 ΔE={s.threshold_low:.3f}（界面上的描边阈值）"
            f"　种子阈值 ΔE={s.threshold:.3f}（= 描边阈值 × {seg.seed_factor}）"
            f"　来源 {s.method}")
        for key in (
            "mode", "roi_box", "roi_pixels", "valid_pixels",
            "base_color", "min_component_px", "components_after_threshold",
            "rejected_negative", "ridge_moved", "ridge_added_px",
        ):
            if key in info:
                add(f"    {key} = {info[key]}")
        if s.stats.get("areas"):
            add(f"    晶体面积(前8) = {sorted(s.stats['areas'], reverse=True)[:8]}")

    # ---------------- 6b. 各图层 ----------------
    if layers:
        add("")
        add(f"【6b】图层（共 {len(layers)} 个，★=当前图层）")
        for i, (name, lay_roi, lay_seg, lay_lines, visible, active) in enumerate(layers):
            mark = "★" if active else " "
            add(f"    {mark} [{i}] {name}　{'显示' if visible else '隐藏'}　"
                f"阈值 ΔE={lay_seg.threshold:.1f}　线条 {lay_lines} 条　"
                f"框 {len(lay_roi.polygons)} 个　绿点 {len(lay_roi.positive_points())} 个"
                f"　红点 {len(lay_roi.negative_points())} 个")

    # ---------------- 7. 结论 ----------------
    add("")
    add("【7】一致性结论")
    problems: list[str] = []
    if not ok_scale:
        problems.append("图像缩放与尺寸不自洽")
    if abs(m11 - fit) >= 0.02:
        problems.append("画布变换 m11 与 viewport/sceneRect 不匹配")
    if out_of_scene:
        problems.append(f"有 {out_of_scene} 个 ROI 顶点绘制到场景外（to_scene 方向可能反了）")
    if result is not None and result.seg is not None and result.seg.info.get("signal_weak"):
        problems.append("绿点与红点处的色差几乎相同（信号弱或标注有误）")
    if problems:
        for p in problems:
            add(f"    ★ {p}")
    else:
        add("    ✓ 未发现自洽性问题")
    if extra_notes:
        add("")
        add("【附注】")
        for n in extra_notes:
            add(f"    {n}")

    add("")
    add("=" * 88)

    text = "\n".join(lines)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        # 目标目录不可写时退回到临时目录
        alt = os.path.join(
            os.environ.get("TEMP", "."), LOG_NAME
        )
        with open(alt, "w", encoding="utf-8") as f:
            f.write(text)
        return alt
    return path


def _gray_of_image(image_path: str | None) -> np.ndarray | None:
    """读一次原图灰度，用于显示提示点处的实际灰度。"""
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        import cv2

        from .imgio import imread

        return cv2.cvtColor(imread(image_path).bgr, cv2.COLOR_BGR2GRAY)
    except Exception:  # noqa: BLE001
        return None


def _sample_gray(gray: np.ndarray | None, x: float, y: float) -> int | None:
    if gray is None:
        return None
    h, w = gray.shape[:2]
    ix = int(np.clip(round(x), 0, w - 1))
    iy = int(np.clip(round(y), 0, h - 1))
    return int(gray[iy, ix])
