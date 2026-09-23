"""分析管线与缓存。

界面上的参数分两类，缓存策略也分两级：

* **分割类参数**（阈值、色差、最小面积…）改变 -> 重新走
  ``preprocess -> segment -> outer -> interior``（约 1 秒）。
* **线条样式**（颜色、线宽、实虚、间距）改变 -> 只重新渲染（100 ms 以内）。

:class:`Analyzer` 负责这套缓存，界面只需调用 :meth:`Analyzer.render`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import LineStyle, OutputMode, Preset, RGB, SegConfig
from .crystal import Crystal, analyze_outer, assign_crystal_ids
from .geometry import Line, polyline_length
from .imgio import ImageData
from .interior import extract_interior_lines
from .preprocess import PreprocessResult, prepare
from .render import composite, render
from .roi import RoiSet
from .segment import SegmentationResult, extract_crystals

__all__ = [
    "AnalysisResult",
    "Analyzer",
    "Layer",
    "LayerStack",
    "default_styles",
    "render_groups",
]


def default_styles(preset: Preset) -> dict[str, LineStyle]:
    return {"outer": preset.outer, "inner": preset.inner}


@dataclass
class AnalysisResult:
    """一次完整分析（分割 + 边界提取）的结果。"""

    size: tuple[int, int]
    lines: list[Line] = field(default_factory=list)
    crystals: list[Crystal] = field(default_factory=list)
    prep: PreprocessResult | None = None
    seg: SegmentationResult | None = None
    info: dict = field(default_factory=dict)

    @property
    def outer_lines(self) -> list[Line]:
        return [ln for ln in self.lines if ln.kind == "outer"]

    @property
    def inner_lines(self) -> list[Line]:
        return [ln for ln in self.lines if ln.kind == "inner"]

    def summary(self) -> str:
        outer = self.outer_lines
        inner = self.inner_lines
        total = sum(ln.length for ln in self.lines)
        return (
            f"晶体 {len(self.crystals)} 块 | 外轮廓 {len(outer)} 条 | "
            f"内部分界 {len(inner)} 条 | 线条总长 {total:.0f} px"
        )


class Analyzer:
    """带缓存的分析器。"""

    def __init__(self, image: ImageData, seg: SegConfig | None = None) -> None:
        self.image = image
        self.seg: SegConfig = (seg or SegConfig()).normalized()
        #: ROI 与提示点（每次导入图片重新画，不进入预设）
        self.roi: RoiSet | None = None
        self._prep: PreprocessResult | None = None
        self._seg: SegmentationResult | None = None
        self._result: AnalysisResult | None = None
        self._prep_key: tuple | None = None
        self._seg_key: tuple | None = None
        self.manual_edits_applied = False

    def set_roi(self, roi: RoiSet | None) -> None:
        """设置 ROI。

        与上次**等价**时直接返回、不清缓存 —— 多图层下每次切换图层都会重设 ROI，
        若无条件清空，会把别的图层已经算好的结果一起清掉。
        """
        cur = self.roi
        if roi is None and cur is None:
            return
        if roi is not None and cur is not None and roi.same_as(cur):
            return
        self.roi = roi
        self._seg = None
        self._result = None
        self._seg_key = None

    # ------------------------------------------------------------------ 缓存键
    def _prep_signature(self) -> tuple:
        s = self.seg
        return (
            s.normalize_illumination,
            round(float(s.background_sigma), 3),
            int(s.brightness),
            round(float(s.contrast), 4),
            round(float(s.gamma), 4),
        )

    def _seg_signature(self) -> tuple:
        s = self.seg
        roi_key: tuple = ()
        if self.roi is not None and not self.roi.empty:
            pts = np.vstack(self.roi.polygons)
            hints = tuple((round(p.x, 1), round(p.y, 1), bool(p.positive)) for p in self.roi.hints)
            roi_key = (
                int(pts.shape[0]),
                round(float(pts.sum()), 3),
                round(float((pts * pts).sum()), 3),
                hints,
            )
        return (
            round(float(s.threshold), 4),
            round(float(s.seed_factor), 4),
            int(s.min_area),
            int(s.max_area),
            int(s.open_radius),
            bool(s.corner_filter),
            round(float(s.simplify_tolerance), 4),
            round(float(s.thin_span), 4),
            bool(s.ridge_snap),
            round(float(s.ridge_peak_ratio), 4),
            round(float(s.ridge_cut_ratio), 4),
            int(s.ridge_max_out),
            int(s.ridge_max_in),
            roi_key,
        )

    def _interior_signature(self) -> tuple:
        return ()

    # ------------------------------------------------------------------ 分析
    def update_seg(self, seg: SegConfig) -> None:
        new = seg.normalized()
        if new.to_dict() == self.seg.to_dict():
            return
        self.seg = new
        self._result = None

    def preprocess(self, cached: PreprocessResult | None = None) -> PreprocessResult:
        key = self._prep_signature()
        # 共享的预处理结果优先：多图层时它由图层栈算一次并复用，
        # 绝不能因为"本层缓存是空的"就重算，更不能顺手把分割缓存清掉。
        if cached is not None and self._prep_key is not None and key == self._prep_key:
            self._prep = cached
            return cached
        if self._prep is not None and key == self._prep_key:
            return self._prep
        s = self.seg
        self._prep = prepare(
            self.image.bgr,
            normalize_illumination=s.normalize_illumination,
            background_sigma=s.background_sigma,
            brightness=s.brightness,
            contrast=s.contrast,
            gamma=s.gamma,
        )
        self._prep_key = key
        self._seg = None
        return self._prep

    def segment(self, cached_prep: PreprocessResult | None = None) -> SegmentationResult:
        key = self._seg_signature()
        prep = self.preprocess(cached_prep)
        if self._seg is not None and key == self._seg_key:
            return self._seg
        s = self.seg
        self._seg = extract_crystals(
            self.image.bgr,
            threshold=s.threshold,
            seed_factor=s.seed_factor,
            min_area=s.min_area,
            max_area=s.max_area,
            open_radius=s.open_radius,
            roi=self.roi,
            apply_ridge_snap=s.ridge_snap,
            ridge_peak_ratio=s.ridge_peak_ratio,
            ridge_cut_ratio=s.ridge_cut_ratio,
            ridge_max_out=s.ridge_max_out,
            ridge_max_in=s.ridge_max_in,
        )
        self._seg_key = key
        return self._seg

    def analyze(
        self, *, force: bool = False, cached_prep: PreprocessResult | None = None
    ) -> AnalysisResult:
        """执行（或复用缓存）完整分析，返回线条集合。

        只提取**外轮廓**。内部分界线已按使用者要求从实战流程中移除
        （``autoedge/interior.py`` 仍保留，便于日后恢复）。

        Args:
            force: 忽略结果缓存，强制重算。
            cached_prep: 外部共享的预处理结果（多图层时由图层栈注入，
                避免每个图层各自重算一遍光照归一化与 Lab 转换）。
        """
        key = (self._prep_signature(), self._seg_signature())
        if (
            self._result is not None
            and not force
            and key == self._result.info.get("_key")
        ):
            return self._result

        t0 = time.perf_counter()
        prep = self.preprocess(cached_prep)
        seg = self.segment(prep)
        t_seg = time.perf_counter()

        crystals = analyze_outer(seg.labels, self.seg, prep.lab)
        t_outer = time.perf_counter()

        info: dict = {
            "threshold": seg.threshold,
            "threshold_method": seg.method,
            "noise": prep.noise,
            "background_level": prep.background_level,
            "auto_sigma": prep.auto_sigma,
            "seg_stats": seg.stats,
        }

        lines: list[Line] = []
        for c in crystals:
            lines.extend(c.lines)
        _annotate(lines)

        info["timing"] = {
            "segment": round(t_seg - t0, 3),
            "outer": round(t_outer - t_seg, 3),
            "interior": 0.0,
            "total": round(t_outer - t0, 3),
        }
        info["_key"] = key

        self._result = AnalysisResult(
            size=self.image.size,
            lines=lines,
            crystals=crystals,
            prep=prep,
            seg=seg,
            info=info,
        )
        return self._result

    # ------------------------------------------------------------------ 渲染
    def render(
        self,
        result: AnalysisResult | None = None,
        *,
        lines: list[Line] | None = None,
        outer: LineStyle | None = None,
        inner: LineStyle | None = None,
        mode: OutputMode = "overlay",
        reference_color: RGB = (255, 255, 255),
        scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """渲染出图。

        Args:
            result: 分析结果；``None`` 时自动调用 :meth:`analyze`。
            lines: 用给定的线条集合代替分析结果（人工编辑后使用）。
            outer: 外轮廓样式；``None`` 时使用默认。
            inner: 内部分界线样式。
            mode: 输出模式。
            reference_color: 透明模式的预览底色。
            scale: 输出缩放（1.0 = 原尺寸）。**只有预览允许小于 1**。

        Returns:
            ``(bgr, alpha)``。
        """
        if result is None:
            result = self.analyze()
        use_lines = result.lines if lines is None else lines
        styles = {
            "outer": (outer or LineStyle()),
            "inner": (inner or LineStyle(style="dashed")),
        }
        if scale != 1.0:
            use_lines = _scale_lines(use_lines, scale)
            styles = {k: _scale_style(v, scale) for k, v in styles.items()}
            size = (max(1, int(round(result.size[0] * scale))), max(1, int(round(result.size[1] * scale))))
            if mode == "overlay":
                base = _resize_bgr(self.image.bgr, size)
            else:
                base = None
            return render(size, use_lines, styles, base, mode, reference_color)

        base = self.image.bgr if mode == "overlay" else None
        return render(result.size, use_lines, styles, base, mode, reference_color)


def _resize_bgr(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(bgr, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)


def _scale_lines(lines: list[Line], scale: float) -> list[Line]:
    out: list[Line] = []
    for ln in lines:
        out.append(
            Line(
                ln.points * float(scale),
                kind=ln.kind,
                crystal_id=ln.crystal_id,
                layer_id=ln.layer_id,
                attrs=dict(ln.attrs),
            )
        )
    return out


def _scale_style(style: LineStyle, scale: float) -> LineStyle:
    s = style.normalized()
    return LineStyle(
        enabled=s.enabled,
        style=s.style,
        color=s.color,
        width=max(1, int(round(s.width * scale))),
        alpha=s.alpha,
    )


def _annotate(lines: list[Line]) -> None:
    """给每条线补充长度、端点、最长段倾角等统计信息（界面状态栏使用）。"""
    for ln in lines:
        if ln.n < 2:
            ln.attrs.update({"length": 0.0, "longest_len": 0.0, "longest_angle": 0.0})
            continue
        closed = bool(ln.attrs.get("closed", False))
        length = polyline_length(ln.points, closed=closed)
        pts = ln.points
        p = np.vstack([pts, pts[:1]]) if closed else pts
        d = np.diff(p, axis=0)
        seg_len = np.hypot(d[:, 0], d[:, 1])
        i = int(np.argmax(seg_len)) if seg_len.size else 0
        from .geometry import orientation_deg

        ln.attrs.update(
            {
                "length": float(length),
                "longest_len": float(seg_len[i]) if seg_len.size else 0.0,
                "longest_angle": float(orientation_deg(d[i])) if seg_len.size else 0.0,
                "start": (float(pts[0, 0]), float(pts[0, 1])),
                "end": (float(pts[-1, 0]), float(pts[-1, 1])),
            }
        )


# ---------------------------------------------------------------------------- 图层
@dataclass
class Layer:
    """一个描边图层。

    每个图层拥有**自己的一套**：选区（ROI）、提示点、阈值、以及线条样式。
    最终显示/输出是"所有可见图层的叠加"，因此可以先用高阈值描出明显轮廓，
    再新建一层用低阈值去描单层区。

    Attributes:
        name: 图层名（界面显示）。
        visible: 是否参与叠加显示与输出。
        analyzer: 该层自己的分析器（持有 ROI、提示点、阈值与各级缓存）。
        style: 该层的线条样式（颜色/线宽/线型）。
    """

    name: str
    visible: bool = True
    analyzer: Analyzer | None = None
    style: LineStyle = field(default_factory=LineStyle)
    #: 上一次预览之后参数是否被改过（界面用来提示"需要重新预览"）
    stale: bool = True

    def ensure_analyzer(self, image: ImageData, seg: SegConfig) -> Analyzer:
        """确保图层有分析器（共享原图；预处理结果由图层栈注入）。"""
        if self.analyzer is None:
            self.analyzer = Analyzer(image, seg)
        return self.analyzer

    @property
    def lines(self) -> list[Line]:
        if self.analyzer is None or self.analyzer._result is None:
            return []
        return self.analyzer._result.lines

    @property
    def result(self) -> AnalysisResult | None:
        return None if self.analyzer is None else self.analyzer._result


class LayerStack:
    """图层集合。

    负责：图层的增删、**共享预处理结果**（光照归一化/Lab 转换与图层无关，
    绝不能每层各算一遍）、以及在"全部图层"口径下的分组渲染。
    """

    def __init__(self, image: ImageData, seg: SegConfig | None = None) -> None:
        self.image = image
        self.base_seg: SegConfig = (seg or SegConfig()).normalized()
        self.layers: list[Layer] = []
        self.current: int = 0
        self.add_layer()

    # ---------------------------------------------------------------- 管理
    def add_layer(self, name: str = "") -> Layer:
        """新建图层。

        **参数与线条样式是全局通用的**，所以新图层直接继承"上一个图层"的设置，
        不从默认值重新开始——否则新建图层会把使用者刚调好的阈值/线宽冲掉。
        """
        idx = len(self.layers) + 1
        prev = self.layers[self.current] if self.layers else None
        if prev is not None and prev.analyzer is not None:
            seg = prev.analyzer.seg
            style = LineStyle(
                enabled=prev.style.enabled,
                style=prev.style.style,
                color=prev.style.color,
                width=prev.style.width,
                alpha=prev.style.alpha,
            )
        else:
            seg = self.base_seg
            style = LineStyle(
                style="solid",
                color=self._next_color(idx),
                width=10,
            )
        lay = Layer(name=name or f"图层{idx}", style=style)
        lay.analyzer = Analyzer(self.image, seg)
        self.layers.append(lay)
        self.current = len(self.layers) - 1
        return lay

    def duplicate_layer(self, index: int | None = None) -> Layer | None:
        i = self.current if index is None else index
        if not (0 <= i < len(self.layers)):
            return None
        src = self.layers[i]
        new = self.add_layer(f"{src.name} 副本")
        # 样式与参数都各存一份，避免两层共享同一个对象互相影响
        new.style = LineStyle(
            enabled=src.style.enabled,
            style=src.style.style,
            color=src.style.color,
            width=src.style.width,
            alpha=src.style.alpha,
        )
        new.visible = src.visible
        if src.analyzer is not None:
            new.analyzer.seg = SegConfig(**src.analyzer.seg.to_dict()).normalized()
            new.analyzer.set_roi(src.analyzer.roi)
        return new

    def remove_layer(self, index: int | None = None) -> bool:
        i = self.current if index is None else index
        if len(self.layers) <= 1 or not (0 <= i < len(self.layers)):
            return False
        del self.layers[i]
        self.current = max(0, min(self.current, len(self.layers) - 1))
        return True

    def move_layer(self, delta: int) -> bool:
        i = self.current
        j = i + delta
        if not (0 <= i < len(self.layers) and 0 <= j < len(self.layers)):
            return False
        self.layers[i], self.layers[j] = self.layers[j], self.layers[i]
        self.current = j
        return True

    @property
    def active(self) -> Layer:
        self.current = max(0, min(self.current, len(self.layers) - 1))
        return self.layers[self.current]

    # ---------------------------------------------------------------- 分析
    def shared_prep(self) -> PreprocessResult:
        """共享的预处理结果（按当前基准参数算一次，所有图层复用）。"""
        for lay in self.layers:
            if lay.analyzer is None:
                continue
            return lay.analyzer.preprocess()
        raise RuntimeError("没有可用图层")

    def preview(self, index: int | None = None, *, force: bool = True) -> AnalysisResult:
        """分析指定图层（默认当前层），并复用共享的预处理结果。"""
        i = self.current if index is None else index
        lay = self.layers[i]
        assert lay.analyzer is not None
        prep = self.shared_prep()
        res = lay.analyzer.analyze(force=force, cached_prep=prep)
        for ln in res.lines:
            ln.layer_id = i
        lay.stale = False
        return res

    def preview_all(self) -> list[AnalysisResult]:
        out: list[AnalysisResult] = []
        prep = self.shared_prep()
        for i, lay in enumerate(self.layers):
            if not lay.visible or lay.analyzer is None:
                continue
            res = lay.analyzer.analyze(force=True, cached_prep=prep)
            for ln in res.lines:
                ln.layer_id = i
            lay.stale = False
            out.append(res)
        return out

    # ---------------------------------------------------------------- 渲染
    def style_groups(self) -> list[tuple[list[Line], LineStyle]]:
        """所有可见图层的 ``(线条, 样式)`` 分组，供分组渲染使用。"""
        groups: list[tuple[list[Line], LineStyle]] = []
        for lay in self.layers:
            if not lay.visible:
                continue
            lines = lay.lines
            if lines:
                groups.append((lines, lay.style))
        return groups

    def all_lines(self) -> list[Line]:
        out: list[Line] = []
        for i, lay in enumerate(self.layers):
            for ln in lay.lines:
                ln.layer_id = i
                out.append(ln)
        return out

    @staticmethod
    def _next_color(idx: int) -> RGB:
        palette = [
            (255, 60, 60),
            (60, 200, 255),
            (255, 200, 0),
            (120, 255, 120),
            (255, 120, 255),
            (255, 255, 255),
        ]
        return palette[(idx - 1) % len(palette)]  # type: ignore[return-value]


def render_groups(
    size: tuple[int, int],
    groups: list[tuple[list[Line], LineStyle]],
    *,
    base_bgr: np.ndarray | None,
    mode: OutputMode = "overlay",
    reference_color: RGB = (255, 255, 255),
    scale: float = 1.0,
    supersample: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """把多组 ``(线条, 样式)`` 依次叠加重绘成一张图（多图层输出用）。

    做法：每一组各自画到一张透明 RGBA 上（组内是正常的抗锯齿绘制），
    再用 alpha 把它们依次叠起来，最后只与底图合成**一次**。

    Args:
        size: 输出尺寸（原图像素）。
        groups: 分组线条与各自样式。
        base_bgr: ``overlay`` 模式的底图（原图）；``transparent`` 模式传 ``None``。
        mode: 输出模式。
        reference_color: 透明模式的预览底色。
        scale: 缩放（预览用 <1，导出用 1.0）。线宽/虚线/坐标同步缩放。
        supersample: 抗锯齿超采样倍数（1 = 关闭）。**导出建议 4**：
            实测浅角度长边的整像素台阶比例从 8%~36% 降到 0%。
            预览请留 1（16 倍像素代价）。

    Returns:
        ``(bgr, alpha)``。
    """
    from PIL import Image

    from .render import draw_lines_rgba

    out_size = size
    if scale != 1.0:
        out_size = (
            max(1, int(round(size[0] * scale))),
            max(1, int(round(size[1] * scale))),
        )

    acc: np.ndarray | None = None
    for lines, style in groups:
        if not lines:
            continue
        use_lines = _scale_lines(lines, scale) if scale != 1.0 else lines
        use_style = _scale_style(style, scale) if scale != 1.0 else style.normalized()
        if not use_style.enabled or use_style.alpha <= 0:
            continue
        # draw_lines_rgba 按 line.kind 取样式，这里统一用 "outer" 键，
        # 于是"每条线用什么样式"完全由图层的分组决定。
        use_lines = [
            Line(ln.points, "outer", ln.crystal_id, ln.layer_id, dict(ln.attrs))
            for ln in use_lines
        ]
        one = draw_lines_rgba(out_size, use_lines, {"outer": use_style}, supersample=supersample)
        if acc is None:
            acc = one
        else:
            merged = Image.alpha_composite(
                Image.fromarray(acc, mode="RGBA"), Image.fromarray(one, mode="RGBA")
            )
            acc = np.asarray(merged, dtype=np.uint8)

    if acc is None:
        # 一条线都没有：直接返回底图（或透明模式的参考底色）
        if base_bgr is not None and mode == "overlay":
            b = _resize_bgr(base_bgr, out_size) if scale != 1.0 else base_bgr
            return b, None
        b = np.full(
            (out_size[1], out_size[0], 3),
            np.array(reference_color, np.uint8)[::-1],
            np.uint8,
        )
        return b, None

    base = None
    if mode == "overlay":
        base = _resize_bgr(base_bgr, out_size) if (scale != 1.0 and base_bgr is not None) else base_bgr
    return composite(base, acc, mode, reference_color)
