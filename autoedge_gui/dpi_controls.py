"""导出 DPI 控件（水平 / 垂直 + 联动）。

主窗口「输出」组与编辑窗口的导出工具条**共用这一个控件**，所以两处行为
不可能不一致。控件自己负责与 :mod:`autoedge.settings` 的读写；导出时由调用方
用 :meth:`DpiControls.value` **现读**，因此不存在"哪个窗口的值才算数"的问题。

设计约定（改的时候别破坏）：

- 默认 **300 × 300**，联动**默认勾选**；勾选时以"水平"为准同步垂直；
- 改动**立即落盘**（跨会话保留），与「输出模式」同构；
- DPI 只是写出文件的元数据（PNG 的 ``pHYs`` / JPEG 的 JFIF density），
  **不参与任何像素计算**，所以这里不发 ``render_changed`` / ``analysis_changed``
  ——发了会让"改个 DPI"触发一次重算，纯属浪费。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from autoedge import settings as app_settings

__all__ = ["DpiControls"]

_TIP = (
    "导出文件的分辨率（dpi）。原图是 300，导出保持一致下游才不会按错误比例处理。\n"
    "只写进文件元数据（PNG 用 pHYs，和 Photoshop 自己导出的 PNG 写法一致），\n"
    "**不改任何像素**：输出仍是原图尺寸（如 4928×3264）。\n"
    "改完立即记住，关掉程序再开还是它。"
)


class DpiControls(QWidget):
    """「水平 / 垂直 + 联动」三个控件。

    ``orientation`` 决定它们怎么摆（信号逻辑完全一样）：

    - ``"row"``：自带一行 ``水平 [300.00 dpi] 垂直 [300.00 dpi] [√]联动``，
      给**编辑窗口的工具栏**用（窗口 1300px 宽，一行放得下）；
    - ``"split"``：不自建布局，把 ``sp_x`` / ``sp_y`` / ``chk_link`` 交给调用方
      摆（主窗口左侧栏只有 ~350px，三个控件横着放会把输入框挤到看不清数字，
      所以那里让 ``QFormLayout`` 竖着排成两行 + 一个勾选行）。
    """

    def __init__(self, parent: QWidget | None = None, orientation: str = "row") -> None:
        super().__init__(parent)
        self._loading = False

        self.sp_x = self._make_spin()
        self.sp_y = self._make_spin()
        self.chk_link = QCheckBox("联动")
        self.chk_link.setToolTip(
            "勾选时改一个另一个跟着变（以「水平」为准）；取消后可分别设置。"
        )
        self.chk_link.setChecked(app_settings.dpi_linked())

        if orientation == "row":
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(QLabel("水平"))
            row.addWidget(self.sp_x)
            row.addWidget(QLabel("垂直"))
            row.addWidget(self.sp_y)
            row.addWidget(self.chk_link)

        self.sp_x.valueChanged.connect(lambda _v: self._on_spin("x"))
        self.sp_y.valueChanged.connect(lambda _v: self._on_spin("y"))
        self.chk_link.toggled.connect(self._on_link)
        self.reload()

    # ------------------------------------------------------------------ 构建
    def _make_spin(self) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(app_settings.DPI_MIN, app_settings.DPI_MAX)
        sp.setDecimals(2)
        sp.setSingleStep(1.0)
        sp.setSuffix(" dpi")
        sp.setFixedWidth(98)
        sp.setToolTip(_TIP)
        return sp

    # ------------------------------------------------------------------ 取值
    def value(self) -> tuple[float, float]:
        """现读全局设置——**导出的唯一取值来源**。"""
        return app_settings.export_dpi()

    def reload(self) -> None:
        """从全局设置读回控件（窗口被激活时调用）。

        输出模式与 DPI 是两个窗口共用的全局项：在编辑窗口改成 600 以后，主窗口
        面板上还写着 300，就会出现"界面显示与实际写入不一致"。激活时重新读一次，
        保证看到的永远是真的。
        """
        x, y = app_settings.export_dpi()
        self._loading = True
        try:
            self.chk_link.setChecked(app_settings.dpi_linked())
            self.sp_x.setValue(x)
            self.sp_y.setValue(y)
        finally:
            self._loading = False

    # ------------------------------------------------------------------ 交互
    def _on_spin(self, which: str) -> None:
        if self._loading:
            return
        x = float(self.sp_x.value())
        y = float(self.sp_y.value())
        if self.chk_link.isChecked():
            if which == "x":
                y = x
            else:
                x = y
            self._loading = True
            try:
                self.sp_x.setValue(x)
                self.sp_y.setValue(y)
            finally:
                self._loading = False
        app_settings.set_export_dpi(x, y)

    def _on_link(self, checked: bool) -> None:
        if self._loading:
            return
        app_settings.set_dpi_linked(checked)
        if checked:
            x = float(self.sp_x.value())
            self._loading = True
            try:
                self.sp_y.setValue(x)
            finally:
                self._loading = False
            app_settings.set_export_dpi(x, x)
