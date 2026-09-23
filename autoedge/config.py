"""算法层配置对象。

分成两组，对应界面上两组行为完全不同的控件：

* ``SegConfig``  —— 分割类参数。改动后需要重新计算分割/边界（约 1 秒）。
* ``LineStyle``  —— 线条样式。改动后只需重新渲染（100 ms 以内，实时预览）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RGB = tuple[int, int, int]
LineKindName = Literal["solid", "dashed", "dashdot"]
OutputMode = Literal["overlay", "transparent"]

DEFAULT_OUTER_COLOR: RGB = (255, 60, 60)
DEFAULT_INNER_COLOR: RGB = (60, 140, 255)


@dataclass
class LineStyle:
    """单条线的样式。线宽以【输出像素】为单位。

    虚线由**固定的内部节距**渲染，界面不再暴露"实段长/间隔"——
    实线时这两个参数完全不生效，留着只会误导。
    """

    enabled: bool = True
    style: LineKindName = "solid"
    color: RGB = DEFAULT_OUTER_COLOR
    width: int = 10
    #: 不透明度 0~255（界面按百分比显示）
    alpha: int = 255

    def normalized(self) -> "LineStyle":
        return LineStyle(
            enabled=bool(self.enabled),
            style=self.style if self.style in ("solid", "dashed", "dashdot") else "solid",
            color=tuple(max(0, min(255, int(c))) for c in self.color),  # type: ignore[arg-type]
            width=max(1, int(self.width)),
            alpha=max(0, min(255, int(self.alpha))),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["color"] = list(self.color)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LineStyle":
        color = tuple(d.get("color", DEFAULT_OUTER_COLOR))
        if len(color) != 3:
            color = DEFAULT_OUTER_COLOR
        return LineStyle(
            enabled=bool(d.get("enabled", True)),
            style=d.get("style", "solid"),
            color=color,  # type: ignore[arg-type]
            width=int(d.get("width", 4)),
            alpha=int(d.get("alpha", 255)),
        ).normalized()


@dataclass
class SegConfig:
    """分割与边界提取参数。"""

    # ---- 光照归一化 ----
    #: 背景估计核大小（像素）。必须远大于晶体尺寸，通常取图像短边的 1/6~1/3。
    background_sigma: float = 0.0  # 0 表示自动
    normalize_illumination: bool = True
    #: 手动亮度/对比度补偿，用于旧照片（0=不变）
    brightness: int = 0
    contrast: float = 1.0
    gamma: float = 1.0

    # ---- 晶体提取（外轮廓）----
    #: 描边阈值（Lab 色差 ΔE）——**界面上的唯一主控参数**。
    #: 越小越灵敏：颜色很浅的单层区也能描上；越大越严格：只有明显对比的区域留下。
    threshold: float = 12.0
    #: 面积下限（像素），小于此值的区域不描边
    min_area: int = 200
    #: 面积上限（像素），0 表示不限制
    max_area: int = 0
    #: 形态学开运算核（像素），去毛刺
    open_radius: int = 1
    #: 是否剔除"贴着画面角落的亮弧"（视场光阑/衬底边缘造成的假区域）
    corner_filter: bool = True
    #: 高阈值（种子）相对描边阈值的倍数：滞后生长用，
    #: 只让与"明显强区"连通的弱区被纳入，避免基底噪声自成一块
    seed_factor: float = 2.5
    #: 是否把外轮廓边界吸附到"色差脊线"（方案A）
    ridge_snap: bool = True
    #: 判定"存在明显更强的脊线"的倍数（越大越保守，越不容易被基底起伏带走）
    ridge_peak_ratio: float = 1.6
    #: 向外续接终止比例：色差掉到脊线处该比例即停
    ridge_cut_ratio: float = 0.6
    #: 脊线向外搜索/外接的最大距离（像素）
    ridge_max_out: int = 14
    #: 脊线向内搜索的最大距离（像素），用于回收被阈值推得太靠外的边界
    ridge_max_in: int = 40

    # ---- 线条几何 ----
    #: 折线简化容差（像素）。只消除像素级锯齿，不做任何平滑、不改变长直线。
    simplify_tolerance: float = 5.0
    #: 薄片合并阈值（像素）：合并"出去又马上回来"的 1~2px 毛刺，
    #: 否则 findContours + DP 会把毛刺两侧合并成一条横穿晶体的长弦。<=0 关闭。
    thin_span: float = 2.5

    def normalized(self) -> "SegConfig":
        def clamp(v: float, lo: float, hi: float) -> float:
            return float(max(lo, min(hi, v)))

        return SegConfig(
            background_sigma=max(0.0, float(self.background_sigma)),
            normalize_illumination=bool(self.normalize_illumination),
            brightness=int(max(-100, min(100, self.brightness))),
            contrast=clamp(self.contrast, 0.2, 4.0),
            gamma=clamp(self.gamma, 0.2, 4.0),
            threshold=clamp(self.threshold, 0.5, 200.0),
            min_area=int(max(0, self.min_area)),
            max_area=int(max(0, self.max_area)),
            open_radius=int(max(0, min(12, self.open_radius))),
            corner_filter=bool(self.corner_filter),
            seed_factor=clamp(self.seed_factor, 1.0, 12.0),
            ridge_snap=bool(self.ridge_snap),
            ridge_peak_ratio=clamp(self.ridge_peak_ratio, 1.01, 10.0),
            ridge_cut_ratio=clamp(self.ridge_cut_ratio, 0.05, 0.95),
            ridge_max_out=int(max(0, min(64, self.ridge_max_out))),
            ridge_max_in=int(max(0, min(128, self.ridge_max_in))),
            simplify_tolerance=clamp(self.simplify_tolerance, 0.0, 20.0),
            thin_span=clamp(self.thin_span, 0.0, 12.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SegConfig":
        valid = {f for f in SegConfig().__dataclass_fields__}  # type: ignore[attr-defined]
        return SegConfig(**{k: v for k, v in d.items() if k in valid}).normalized()


@dataclass
class Preset:
    """一个可保存/载入的预设（分割参数 + 线条样式 + 输出设置）。"""

    name: str = "默认"
    seg: SegConfig = field(default_factory=SegConfig)
    outer: LineStyle = field(
        default_factory=lambda: LineStyle(style="solid", color=DEFAULT_OUTER_COLOR, width=10)
    )
    inner: LineStyle = field(
        default_factory=lambda: LineStyle(style="dashed", color=DEFAULT_INNER_COLOR, width=3)
    )
    #: 输出模式。默认 **透明底 + 线条**（使用者要求：导出默认就是最终交付格式，
    #: 免得忘了改设置导错图）。实际运行时以 `autoedge.settings` 里记住的
    #: 上一次选择为准，载入预设**不会**改变它。
    output_mode: OutputMode = "transparent"
    #: 透明模式下预览用的底色（导出 PNG 时不写入，仅影响看图）
    reference_color: RGB = (255, 255, 255)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seg": self.seg.to_dict(),
            "outer": self.outer.to_dict(),
            "inner": self.inner.to_dict(),
            "output_mode": self.output_mode,
            "reference_color": list(self.reference_color),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Preset":
        color = tuple(d.get("reference_color", (255, 255, 255)))
        if len(color) != 3:
            color = (255, 255, 255)
        return Preset(
            name=str(d.get("name", "预设")),
            seg=SegConfig.from_dict(d.get("seg", {})),
            outer=LineStyle.from_dict(d.get("outer", {})),
            inner=LineStyle.from_dict(d.get("inner", {})),
            output_mode=d.get("output_mode", "overlay"),
            reference_color=color,  # type: ignore[arg-type]
        )

    @staticmethod
    def from_json(text: str) -> "Preset":
        return Preset.from_dict(json.loads(text))
