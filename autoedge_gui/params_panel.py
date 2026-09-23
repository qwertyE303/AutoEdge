"""参数面板。

只负责"界面控件 <-> :class:`autoedge.config.Preset`"的双向映射，
不参与任何图像处理，也不持有分析逻辑。

界面刻意精简：**只保留一个阈值**（描边阈值），其余都是线条样式与输出设置。
阈值越低越灵敏（颜色很浅的单层区也能描上），越高越严格（只有明显对比留下）。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from autoedge.config import LineStyle, Preset, SegConfig

__all__ = ["ParamPanel"]

#: 描边阈值滑块的刻度：内部放大 10 倍，因此最小步进 = 0.1 ΔE
_THR_SCALE = 10
_THR_MIN = 10
_THR_MAX = 1500


class _StepSlider(QSlider):
    """滚轮按**最小步进**走的滑块（Qt 默认的滚轮是翻页，一格跳太多）。"""

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        step = max(1, self.singleStep())
        self.setValue(self.value() + (step if delta > 0 else -step))
        event.accept()


class ParamPanel(QScrollArea):
    """左侧参数面板。

    信号：
    * :attr:`analysis_changed` —— 分析类参数变化（需要重新分析）
    * :attr:`render_changed`   —— 渲染类参数变化（只需重绘）
    """

    analysis_changed = Signal()
    render_changed = Signal()
    #: 请求载入某个预设（参数 + 线条样式，应用到**所有图层**）
    preset_load_requested = Signal(str)
    #: 请求保存当前设置为预设
    preset_save_requested = Signal()
    #: 请求删除某个预设
    preset_delete_requested = Signal(str)

    def __init__(self, preset: Preset, parent=None) -> None:
        super().__init__(parent)
        self.preset = preset
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._colors: dict[str, QColor] = {}

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(self._build_seg_box())
        layout.addWidget(self._build_style_box())
        layout.addWidget(self._build_output_box())
        layout.addWidget(self._build_preset_box())
        layout.addStretch(1)
        self.setWidget(body)
        self.refresh_presets()
        self.sync_from_preset()

    # ------------------------------------------------------------------ 构建
    def _build_seg_box(self) -> QGroupBox:
        box = QGroupBox("参数")
        v = QVBoxLayout(box)

        # ---------- 主参数：只放阈值 ----------
        top = QFormLayout()
        top.setLabelAlignment(Qt.AlignRight)
        self.sld_threshold = _StepSlider(Qt.Horizontal)
        self.sld_threshold.setRange(_THR_MIN, _THR_MAX)
        # 内部是 ΔE×10，所以 singleStep=1 就是 0.1 ΔE（滚轮/方向键同样按这个走）
        self.sld_threshold.setSingleStep(1)
        self.sld_threshold.setPageStep(20)
        self.sld_threshold.setTracking(True)
        self.sld_threshold.setToolTip(
            "描边阈值（色差 ΔE）。\n"
            "越小越灵敏：颜色很浅的单层区也能描上；\n"
            "越大越严格：只留下对比明显的轮廓。\n"
            "调完后点「▶ 预览」重新分析。"
        )
        self.sld_threshold.valueChanged.connect(self._on_threshold_changed)
        self.sp_threshold = QDoubleSpinBox()
        self.sp_threshold.setRange(_THR_MIN / _THR_SCALE, _THR_MAX / _THR_SCALE)
        self.sp_threshold.setDecimals(2)
        self.sp_threshold.setSingleStep(0.1)
        self.sp_threshold.setToolTip("可以直接输入精确值（到 0.01 ΔE）")
        self.sp_threshold.valueChanged.connect(self._on_threshold_spin)
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.sld_threshold, 1)
        row.addWidget(self.sp_threshold, 0)
        top.addRow("描边阈值", holder)
        v.addLayout(top)

        # ---------- 高级选项（可折叠） ----------
        self.btn_advanced = QToolButton()
        self.btn_advanced.setText("高级选项")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setChecked(False)
        self.btn_advanced.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_advanced.setArrowType(Qt.RightArrow)
        self.btn_advanced.clicked.connect(self._toggle_advanced)
        v.addWidget(self.btn_advanced)

        self.adv_body = QWidget()
        adv = QFormLayout(self.adv_body)
        adv.setContentsMargins(12, 0, 0, 0)
        adv.setLabelAlignment(Qt.AlignRight)

        self.sp_min_area = QDoubleSpinBox()
        self.sp_min_area.setRange(0, 10_000_000)
        self.sp_min_area.setDecimals(0)
        self.sp_min_area.setSingleStep(50)
        self.sp_min_area.setToolTip("小于该面积的区域不描边")
        self.sp_min_area.valueChanged.connect(self.analysis_changed)
        adv.addRow("最小面积(px)", self.sp_min_area)

        self.sp_simplify = QDoubleSpinBox()
        self.sp_simplify.setRange(0.0, 20.0)
        self.sp_simplify.setDecimals(2)
        self.sp_simplify.setSingleStep(0.5)
        self.sp_simplify.setToolTip(
            "折线简化容差（像素）。越大顶点越少、线条越干净；\n"
            "不做任何平滑，只删掉偏差小于该值的中间顶点。"
        )
        self.sp_simplify.valueChanged.connect(self.analysis_changed)
        adv.addRow("简化容差(px)", self.sp_simplify)

        self.adv_body.setVisible(False)
        v.addWidget(self.adv_body)
        return box

    def _toggle_advanced(self, checked: bool) -> None:
        self.adv_body.setVisible(checked)
        self.btn_advanced.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _build_style_box(self) -> QGroupBox:
        box = QGroupBox("线条样式（本图层）")
        f = QFormLayout(box)
        self._colors["outer"] = QColor(255, 60, 60)

        self.cmb_style = QComboBox()
        self.cmb_style.addItem("实线", "solid")
        self.cmb_style.addItem("虚线", "dashed")
        self.cmb_style.addItem("点划线", "dashdot")
        self.cmb_style.currentIndexChanged.connect(self.render_changed)
        f.addRow("线型", self.cmb_style)

        self.btn_color = QPushButton()
        self.btn_color.clicked.connect(lambda: self._pick_color("outer"))
        f.addRow("颜色", self.btn_color)

        self.sp_width = QDoubleSpinBox()
        self.sp_width.setRange(1, 200)
        self.sp_width.setDecimals(0)
        self.sp_width.setSingleStep(1)
        self.sp_width.valueChanged.connect(self.render_changed)
        f.addRow("线宽(px)", self.sp_width)

        self.sp_alpha = QDoubleSpinBox()
        self.sp_alpha.setRange(0, 100)
        self.sp_alpha.setDecimals(0)
        self.sp_alpha.setSingleStep(5)
        self.sp_alpha.setSuffix(" %")
        self.sp_alpha.setToolTip("线条不透明度（百分比）")
        self.sp_alpha.valueChanged.connect(self.render_changed)
        f.addRow("不透明度", self.sp_alpha)
        return box

    def _build_output_box(self) -> QGroupBox:
        box = QGroupBox("输出")
        f = QFormLayout(box)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("原图 + 线条", "overlay")
        self.cmb_mode.addItem("透明底 + 线条(PNG)", "transparent")
        self.cmb_mode.setToolTip(
            "导出格式。**默认「透明底 + 线条」**，且会被记住（下次启动仍是它）。\n"
            "输出模式是全局设置，不随图层/预设变化；透明底只能存 PNG。"
        )
        # 初值来自上次的选择（记住的设置），只影响导出，不影响画布预览
        from autoedge import settings as app_settings

        self._set_mode_combo(app_settings.output_mode())
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        f.addRow("模式", self.cmb_mode)
        return box

    def _set_mode_combo(self, mode: str) -> None:
        """按模式值选中下拉项（屏蔽信号，避免"初始化即写盘"）。"""
        self.cmb_mode.blockSignals(True)
        self.cmb_mode.setCurrentIndex(1 if mode == "transparent" else 0)
        self.cmb_mode.blockSignals(False)

    def _on_mode_changed(self, _idx: int) -> None:
        """模式改动：立即记住（跨会话生效）并请求重绘。"""
        from autoedge import settings as app_settings

        app_settings.set_output_mode(self.output_mode())
        self.render_changed.emit()

    def _build_preset_box(self) -> QGroupBox:
        box = QGroupBox("预设（参数 + 线条样式，所有图层通用）")
        v = QVBoxLayout(box)
        self.cmb_preset = QComboBox()
        self.cmb_preset.setToolTip("选择一个预设即可套用到所有图层")
        self.cmb_preset.activated.connect(self._on_preset_activated)
        v.addWidget(self.cmb_preset)

        row = QHBoxLayout()
        self.btn_preset_save = QPushButton("保存当前为预设…")
        self.btn_preset_save.clicked.connect(lambda: self.preset_save_requested.emit())
        self.btn_preset_delete = QPushButton("删除")
        self.btn_preset_delete.clicked.connect(self._on_preset_delete)
        row.addWidget(self.btn_preset_save, 1)
        row.addWidget(self.btn_preset_delete, 0)
        v.addLayout(row)
        return box

    def _on_preset_activated(self, _idx: int) -> None:
        name = self.cmb_preset.currentText()
        if name:
            self.preset_load_requested.emit(name)

    def _on_preset_delete(self) -> None:
        name = self.cmb_preset.currentText()
        if name:
            self.preset_delete_requested.emit(name)

    def refresh_presets(self, select: str | None = None) -> None:
        """重新扫描预设目录，刷新下拉框。"""
        from autoedge.presets import list_presets

        cur = select if select is not None else self.cmb_preset.currentText()
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        names = list_presets()
        self.cmb_preset.addItems(names)
        if cur and cur in names:
            self.cmb_preset.setCurrentText(cur)
        elif names:
            self.cmb_preset.setCurrentIndex(0)
        self.cmb_preset.blockSignals(False)
        self.btn_preset_delete.setEnabled(bool(names))

    # ------------------------------------------------------------------ 交互
    def _on_threshold_changed(self, v: int) -> None:
        """滑块动了：同步精确输入框，并请求重新分析。"""
        self.sp_threshold.blockSignals(True)
        self.sp_threshold.setValue(v / _THR_SCALE)
        self.sp_threshold.blockSignals(False)
        self.analysis_changed.emit()

    def _on_threshold_spin(self, v: float) -> None:
        """精确输入框改了：同步滑块（滑块只到 0.1），并请求重新分析。"""
        self.sld_threshold.blockSignals(True)
        self.sld_threshold.setValue(int(round(v * _THR_SCALE)))
        self.sld_threshold.blockSignals(False)
        self.analysis_changed.emit()

    def threshold_value(self) -> float:
        """当前描边阈值（ΔE），保留精确输入框给到的精度。"""
        return float(self.sp_threshold.value())

    def _pick_color(self, kind: str = "outer") -> None:
        c = QColorDialog.getColor(self._colors[kind], self, "选择线条颜色")
        if c.isValid():
            self._colors[kind] = c
            self._update_color_button()
            self.render_changed.emit()

    def _update_color_button(self) -> None:
        c = self._colors["outer"]
        self.btn_color.setText(c.name())
        self.btn_color.setStyleSheet(
            f"background-color: {c.name()}; color: "
            f"{'black' if c.lightness() > 128 else 'white'};"
        )

    # ------------------------------------------------------------------ 与图层同步
    def load_layer(self, style: LineStyle, seg: SegConfig) -> None:
        """把**当前图层**的样式与参数载入控件（屏蔽信号）。"""
        widgets = (
            self.sld_threshold,
            self.sp_threshold,
            self.sp_min_area,
            self.sp_simplify,
            self.cmb_style,
            self.sp_width,
            self.sp_alpha,
        )
        for w in widgets:
            w.blockSignals(True)

        self.sld_threshold.setValue(int(round(seg.threshold * _THR_SCALE)))
        self.sp_threshold.setValue(float(seg.threshold))
        self.sp_min_area.setValue(float(seg.min_area))
        self.sp_simplify.setValue(float(seg.simplify_tolerance))

        st = style.normalized()
        self.cmb_style.setCurrentIndex({"solid": 0, "dashed": 1, "dashdot": 2}.get(st.style, 0))
        self.sp_width.setValue(float(st.width))
        self.sp_alpha.setValue(round(st.alpha / 255.0 * 100.0))
        self._colors["outer"] = QColor(*st.color)
        self._update_color_button()

        for w in widgets:
            w.blockSignals(False)

    def apply_to_layer(self, style: LineStyle, seg: SegConfig) -> tuple[LineStyle, SegConfig]:
        """把控件值写回该图层的样式与参数。"""
        c = self._colors["outer"]
        seg.threshold = self.threshold_value()
        seg.min_area = int(round(self.sp_min_area.value()))
        seg.simplify_tolerance = float(self.sp_simplify.value())
        style.enabled = True
        style.style = self.cmb_style.currentData()
        style.color = (c.red(), c.green(), c.blue())
        style.width = int(round(self.sp_width.value()))
        style.alpha = int(round(self.sp_alpha.value() / 100.0 * 255.0))
        return style, seg

    # ------------------------------------------------------------------ 输出设置
    def output_mode(self) -> str:
        return self.cmb_mode.currentData()

    def reference_color(self) -> tuple[int, int, int]:
        return self.preset.reference_color

    # ------------------------------------------------------------------ 加载预设
    def sync_from_preset(self) -> None:
        """把预设写回控件（会屏蔽信号，避免触发重复分析）。

        **不碰输出模式**：模式是全局记忆项（由 :mod:`autoedge.settings` 保存），
        预设里既不含它、载入预设也不该把它改掉——否则会退回"忘了改就导错图"。
        """
        p = self.preset
        self.load_layer(p.outer, p.seg)
        for w, v in ((self.sp_width, p.outer.width),):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    def collect(self, name: str = "") -> Preset:
        """读控件，生成 :class:`Preset`（仅用于导出/回写当前图层）。"""
        style = LineStyle()
        seg = SegConfig()
        style, seg = self.apply_to_layer(style, seg)
        return Preset(
            name=name or self.preset.name,
            seg=seg,
            outer=style,
            inner=self.preset.inner,
            output_mode=self.output_mode(),
            reference_color=self.reference_color(),
        )
