"""手动编辑窗口。

设计要点：编辑对象是**矢量线段集合**，不是位图。删/补都精确操作，
导出时再统一按各图层样式光栅化，输出尺寸与原图严格一致。

多图层：列表里包含**所有可见图层**的线条，每条线记住自己属于哪个图层
（``Line.layer_id``），渲染时按图层各自的颜色/线宽/线型绘制。
顶部有"显示哪些图层"的筛选，方便逐层清理。

交互（三选一的工具）：

* **浏览** —— Shift+左键 或 中键拖动平移，滚轮缩放（不改线条）。
* **加线** —— 左键点第一个点，再左键点第二个点（或右键）连成一条**直线**；
  线宽可调（"新线线宽"），新线归属当前选中的图层。
* **擦除** —— 出现一个圆圈（大小可调），按住左键拖动即可擦掉圈内的线段，
  和 PS 的橡皮一样；线段被擦断会拆成多条。

其余：撤销/重做/重置/输出最终图片；``Esc`` 取消当前画线。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autoedge import settings as app_settings
from autoedge.config import LineStyle
from autoedge.geometry import Line, polyline_length
from autoedge.imgio import ImageData, save_image
from autoedge.pipeline import LayerStack, render_groups
from autoedge.render import SS_DEFAULT
from autoedge_gui.canvas import ImageCanvas, bgr_to_qimage
from autoedge_gui.notifications import error, notify, notify_success

__all__ = ["EditWindow"]

PREVIEW_MAX_SIDE = 1600


def _lh(a: Line) -> float:
    return polyline_length(a.points, closed=a.closed)


def _bgra_to_qimage(bgra: np.ndarray) -> QImage:
    """``(h, w, 4)`` uint8 BGRA（OpenCV 顺序）-> ``QImage``（Format_ARGB32 小端即 BGRA）。"""
    arr = np.ascontiguousarray(bgra)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 4 * w, QImage.Format_ARGB32).copy()


def _draw_disc(tile: np.ndarray, cx: float, cy: float, r: float,
               fill: tuple[int, int, int, int] | None = None,
               ring: tuple[int, int, int, int] | None = None,
               ring_w: float = 2.0, dash: float = 0.0) -> None:
    """在 BGRA 图上画圆，用 numpy 直接写（避免 Qt 绘制生命周期问题）。

    * ``fill`` 非空 -> 实心填充
    * ``ring`` 非空 -> 描边；``dash`` > 0 时画成**虚线圆圈**（沿圆周按弧长断开）
    """
    h, w = tile.shape[:2]
    pad = r + (ring_w if ring is not None else 0.0) + 2.0
    x0 = max(0, int(np.floor(cx - pad)))
    x1 = min(w, int(np.ceil(cx + pad)) + 1)
    y0 = max(0, int(np.floor(cy - pad)))
    y1 = min(h, int(np.ceil(cy + pad)) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xs - cx, ys - cy)
    if fill is not None:
        tile[y0:y1, x0:x1][d <= r] = fill
    if ring is not None:
        band = np.abs(d - r) <= max(0.5, ring_w / 2.0)
        if dash > 0:
            # 沿圆周的弧长 -> 按"实段/间隔"切开
            ang = np.arctan2(ys - cy, xs - cx)
            arc = (ang % (2 * np.pi)) * r
            band = band & ((arc % (dash * 2.0)) < dash)
        tile[y0:y1, x0:x1][band] = ring


def _draw_line(tile: np.ndarray, x0: float, y0: float, x1: float, y1: float,
               color: tuple[int, int, int, int], width: float = 2.0,
               dash: float = 0.0) -> None:
    """在 BGRA 图上画一条线（``dash`` > 0 时按间隔断开），用 numpy 直接写。"""
    h, w = tile.shape[:2]
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < 1e-6:
        return
    n = max(2, int(length) + 1)
    t = np.linspace(0.0, 1.0, n)
    if dash > 0:
        s = t * length
        t = t[(s % (dash * 2)) < dash]
    px = x0 + (x1 - x0) * t
    py = y0 + (y1 - y0) * t
    r = max(0.5, width / 2.0)
    rad = int(np.ceil(r)) + 1
    ix = np.rint(px).astype(int)
    iy = np.rint(py).astype(int)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if dx * dx + dy * dy > r * r + 0.25:
                continue
            xx = np.clip(ix + dx, 0, w - 1)
            yy = np.clip(iy + dy, 0, h - 1)
            tile[yy, xx] = color


class EditWindow(QMainWindow):
    def __init__(
        self,
        image: ImageData,
        stack: LayerStack,
        output_mode: str = "overlay",
        reference_color: tuple[int, int, int] = (255, 255, 255),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.image = image
        self.stack = stack
        self.output_mode = output_mode
        self.reference_color = reference_color
        self.setWindowTitle("AutoEdge — 手动编辑（所有图层）")
        self.resize(1300, 950)

        #: 编辑对象 = 所有可见图层的线条（带 layer_id）
        self.lines: list[Line] = [
            Line(
                ln.points.copy(),
                kind=ln.kind,
                crystal_id=ln.crystal_id,
                layer_id=ln.layer_id,
                attrs=dict(ln.attrs),
            )
            for ln in stack.all_lines()
            if stack.layers[ln.layer_id].visible
        ]
        self._undo: list[list[Line]] = []
        self._redo: list[list[Line]] = []

        self.tool = "pan"          # pan | line | erase
        self._pending_pt: tuple[float, float] | None = None
        self._cursor_img: tuple[float, float] | None = None
        self._erasing = False
        self._erased_any = False
        self._preview_scale = 1.0
        self._base_bgr = None
        self._base_cache = None
        self._view_mode = "full"
        self._hi_zoom = 1.05
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(90)
        self._zoom_timer.timeout.connect(self._refresh_for_zoom)

        self._build_ui()
        self._build_menu()
        self._refresh()

    # ================================================================== 样式
    def layer_style(self, layer_id: int) -> LineStyle:
        if 0 <= layer_id < len(self.stack.layers):
            return self.stack.layers[layer_id].style
        return LineStyle()

    def layer_name(self, layer_id: int) -> str:
        if 0 <= layer_id < len(self.stack.layers):
            return self.stack.layers[layer_id].name
        return f"图层{layer_id + 1}"

    def visible_layer_ids(self) -> list[int]:
        data = self.cmb_layer.currentData()
        if data is None:
            return list(range(len(self.stack.layers)))
        return [int(data)]

    def _line_visible_in_editor(self, ln: Line) -> bool:
        return ln.layer_id in self.visible_layer_ids()

    def _style_groups(self, lines: list[Line], scale: float):
        """按 layer_id 分组，返回 render_groups 需要的分组。

        ``scale`` 必须传**预览缩放**：线宽是"原图像素"单位，
        不跟着缩就会在预览里显得比主窗口粗一大截。
        """
        groups = []
        by_layer: dict[int, list[Line]] = {}
        for ln in lines:
            by_layer.setdefault(ln.layer_id, []).append(ln)
        for lid, ls in by_layer.items():
            st = self.layer_style(lid)
            if scale != 1.0:
                st = LineStyle(
                    enabled=st.enabled,
                    style=st.style,
                    color=st.color,
                    width=max(1, int(round(st.width * scale))),
                    alpha=st.alpha,
                )
            # 统一的 outer 键：每条线用什么样式由图层的分组决定
            groups.append(
                ([Line(l.points, "outer", l.crystal_id, l.layer_id, dict(l.attrs)) for l in ls], st)
            )
        return groups

    # ================================================================== 界面
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        bar = QHBoxLayout()
        self.btn_pan = QPushButton("浏览")
        self.btn_add = QPushButton("加线")
        self.btn_erase = QPushButton("擦除")
        for b in (self.btn_pan, self.btn_add, self.btn_erase):
            b.setCheckable(True)
            b.setMinimumWidth(64)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        for b in (self.btn_pan, self.btn_add, self.btn_erase):
            self._tool_group.addButton(b)
        self.btn_pan.setChecked(True)
        self.btn_pan.clicked.connect(lambda: self._set_tool("pan"))
        self.btn_add.clicked.connect(lambda: self._set_tool("line"))
        self.btn_erase.clicked.connect(lambda: self._set_tool("erase"))
        bar.addWidget(QLabel("工具："))
        bar.addWidget(self.btn_pan)
        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_erase)

        bar.addSpacing(12)
        self.sp_new_width = QDoubleSpinBox()
        self.sp_new_width.setRange(1, 200)
        self.sp_new_width.setDecimals(0)
        self.sp_new_width.setSingleStep(1)
        self.sp_new_width.setSuffix(" px")
        self.sp_new_width.setToolTip("用「加线」新建的线条宽度（原图像素）")
        bar.addWidget(QLabel("新线线宽："))
        bar.addWidget(self.sp_new_width)

        bar.addSpacing(12)
        self.sp_brush = QDoubleSpinBox()
        self.sp_brush.setRange(4, 400)
        self.sp_brush.setDecimals(0)
        self.sp_brush.setSingleStep(4)
        self.sp_brush.setSuffix(" px")
        self.sp_brush.setToolTip("「擦除」圆圈的大小（原图像素）")
        self.sp_brush.valueChanged.connect(lambda _v: self._refresh())
        bar.addWidget(QLabel("擦除圆圈："))
        bar.addWidget(self.sp_brush)

        bar.addSpacing(12)
        bar.addWidget(QLabel("图层筛选"))
        self.cmb_layer = QComboBox()
        self.cmb_layer.addItem("显示：全部图层", None)
        for i, lay in enumerate(self.stack.layers):
            if lay.visible:
                self.cmb_layer.addItem(f"只显示：{lay.name}", i)
        self.cmb_layer.setToolTip("只显示某一图层的线条，方便逐层清理")
        # 先填好再加信号，避免初始 addItem 就触发一次刷新
        self.cmb_layer.currentIndexChanged.connect(lambda _i: self._refresh())
        bar.addWidget(self.cmb_layer)
        bar.addStretch(1)
        layout.addLayout(bar)

        bar2 = QHBoxLayout()
        self.btn_undo = QPushButton("撤销 (Ctrl+Z)")
        self.btn_redo = QPushButton("重做 (Ctrl+Y)")
        self.btn_reset = QPushButton("重置 (R)")
        self.btn_reset.setToolTip("丢弃手动加的线/擦除结果，恢复成算法刚描出来的样子")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_reset.clicked.connect(self.reset_lines)
        for b in (self.btn_undo, self.btn_redo, self.btn_reset):
            bar2.addWidget(b)
        bar2.addStretch(1)

        self.cmb_out_mode = QComboBox()
        self.cmb_out_mode.addItem("原图 + 线条", "overlay")
        self.cmb_out_mode.addItem("透明底 + 线条(PNG)", "transparent")
        self.cmb_out_mode.setToolTip(
            "导出格式。**默认「透明底 + 线条」**，与主窗口共用同一个记忆值：\n"
            "在这里改了，主窗口下次也是它（关掉程序也保留）。"
        )
        # 初值 = 本次会话/上次记住的模式（不是硬编码的 overlay）
        self._select_out_mode(self.output_mode or app_settings.output_mode())
        self.output_mode = str(self.cmb_out_mode.currentData())
        self.cmb_out_mode.currentIndexChanged.connect(self._on_out_mode_changed)
        bar2.addWidget(QLabel("导出："))
        bar2.addWidget(self.cmb_out_mode)

        self.btn_export = QPushButton("输出最终图片…")
        self.btn_export.clicked.connect(self.export_final)
        bar2.addWidget(self.btn_export)
        layout.addLayout(bar2)

        self.canvas = ImageCanvas()
        self.canvas.clicked.connect(self._on_click)
        self.canvas.right_clicked.connect(self._on_right_click)
        self.canvas.dragged.connect(self._on_drag)
        self.canvas.drag_released.connect(self._on_drag_release)
        self.canvas.cursor_moved.connect(self._on_move)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        layout.addWidget(self.canvas, 1)
        # 状态栏标签必须在 _set_tool -> _refresh 之前建好
        self.lbl_status = QLabel()
        layout.addWidget(self.lbl_status)
        self.setCentralWidget(central)

        self._make_base_image()
        self.canvas.setMouseTracking(True)
        # 新线线宽初值 = 当前选中图层（或第一个图层）的线宽
        ids = [i for i, lay in enumerate(self.stack.layers) if lay.visible] or [0]
        self.sp_new_width.setValue(float(self.layer_style(ids[0]).width))
        self.sp_brush.setValue(60.0)
        self._set_tool("pan")

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("编辑(&E)")
        for text, seq, slot in (
            ("撤销", "Ctrl+Z", self.undo),
            ("重做", "Ctrl+Y", self.redo),
            ("重置", "R", self.reset_lines),
            ("输出最终图片", "Ctrl+S", self.export_final),
        ):
            act = QAction(text, self)
            act.setShortcut(QKeySequence(seq))
            act.triggered.connect(slot)
            m.addAction(act)

    # ================================================================== 工具
    def _select_out_mode(self, mode: str) -> None:
        """按模式值选中导出格式（屏蔽信号，避免"初始化即写盘"）。"""
        self.cmb_out_mode.blockSignals(True)
        self.cmb_out_mode.setCurrentIndex(1 if mode == "transparent" else 0)
        self.cmb_out_mode.blockSignals(False)

    def _on_out_mode_changed(self, _idx: int) -> None:
        """导出格式改动：立即记住，与主窗口共用同一个值。"""
        self.output_mode = str(self.cmb_out_mode.currentData())
        app_settings.set_output_mode(self.output_mode)

    def _set_tool(self, tool: str) -> None:
        self.tool = tool
        self._pending_pt = None
        for name, btn in (("pan", self.btn_pan), ("line", self.btn_add), ("erase", self.btn_erase)):
            btn.setChecked(name == tool)
        cur = {"pan": Qt.ArrowCursor, "line": Qt.CrossCursor, "erase": Qt.BlankCursor}
        self.canvas.setCursor(cur.get(tool, Qt.ArrowCursor))
        self.canvas.clear_overlay()
        self._cursor_img = None
        self._update_status()

    # ================================================================== 底图与重绘
    def _make_base_image(self) -> QImage:
        import cv2

        img = self.image
        self._preview_scale = min(1.0, PREVIEW_MAX_SIDE / max(img.width, img.height))
        if self._preview_scale < 1.0:
            small = cv2.resize(
                img.bgr,
                (int(img.width * self._preview_scale), int(img.height * self._preview_scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = img.bgr
        self._base_bgr = np.ascontiguousarray(small)
        self._base_cache = bgr_to_qimage(self._base_bgr)
        return self._base_cache

    def _scaled_lines(self) -> list[Line]:
        s = self._preview_scale
        if s == 1.0:
            return self.lines
        return [
            Line(ln.points * s, kind=ln.kind, crystal_id=ln.crystal_id, layer_id=ln.layer_id,
                 attrs=dict(ln.attrs))
            for ln in self.lines
        ]

    def _refresh(self) -> None:
        """重画底图 + 线条 + 光标提示（放大时按可见区域用原图重画）。"""
        crop = self._crop_rect()
        if crop is not None:
            self._view_mode = "zoom"
            self._render_highres(crop)
        else:
            self._view_mode = "full"
            self._render_full()
        self._update_status()

    # ------------------------------------------------------------------ 渲染
    def _magnification(self) -> float:
        """原图 1 个像素在屏幕上占几个像素（交给画布统一算）。"""
        return self.canvas.magnification()

    def _crop_rect(self) -> tuple[int, int, int, int] | None:
        """放大到 1:1 以上时返回要按原图重画的区域；否则 None。"""
        if self._magnification() <= self._hi_zoom:
            return None
        c = self.canvas
        poly = c.mapToScene(c.viewport().rect())
        r = poly.boundingRect()
        x0, y0 = c.to_image(r.left(), r.top())
        x1, y1 = c.to_image(r.right(), r.bottom())
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        pad = 0.12 * max(x1 - x0, y1 - y0)
        ix0 = int(max(0, np.floor(x0 - pad)))
        iy0 = int(max(0, np.floor(y0 - pad)))
        ix1 = int(min(self.image.width, np.ceil(x1 + pad)))
        iy1 = int(min(self.image.height, np.ceil(y1 + pad)))
        if ix1 - ix0 < 32 or iy1 - iy0 < 32:
            return None
        return ix0, iy0, ix1, iy1

    def _render_full(self) -> None:
        import cv2

        s = self._preview_scale
        lines = [ln for ln in self._scaled_lines() if self._line_visible_in_editor(ln)]
        size = (self._base_bgr.shape[1], self._base_bgr.shape[0])
        out = np.ascontiguousarray(self._base_bgr.copy())
        if lines:
            groups = self._style_groups(lines, s)
            bgr, _a = render_groups(size, groups, base_bgr=self._base_bgr,
                                    mode="overlay", scale=1.0)
            out = np.ascontiguousarray(bgr)
        qimg = bgr_to_qimage(out)
        self._paint_hints(qimg, (0, 0), s)
        self.canvas.set_image(qimg, keep_view=True, original_size=self.image.size)
        # 底图换了，旧的光标叠加层已经作废：清掉并按新位置重画，
        # 否则会在原地留下一个"幽灵圆圈"（要等下次重画才消失）。
        self._redraw_cursor_overlay_after_render()
        self._zoom_timer.stop()

    def _render_highres(self, crop: tuple[int, int, int, int]) -> None:
        import cv2

        x0, y0, x1, y1 = crop
        cw, ch = x1 - x0, y1 - y0
        sub = self.image.bgr[y0:y1, x0:x1]
        s = self._preview_scale
        dw = max(1, int(round(cw * s)))
        dh = max(1, int(round(ch * s)))
        base = cv2.resize(sub, (dw, dh), interpolation=cv2.INTER_AREA) if s < 1.0 else sub

        shifted: list = []
        for ln in self.lines:
            if not self._line_visible_in_editor(ln):
                continue
            pts = np.asarray(ln.points, np.float32).copy()
            pts[:, 0] = (pts[:, 0] - x0) * s
            pts[:, 1] = (pts[:, 1] - y0) * s
            shifted.append(Line(pts, ln.kind, ln.crystal_id, ln.layer_id, dict(ln.attrs)))
        groups = self._style_groups(shifted, s)
        bgr, _a = render_groups((dw, dh), groups, base_bgr=base, mode="overlay",
                                reference_color=self.reference_color, scale=1.0)
        qimg = bgr_to_qimage(np.ascontiguousarray(bgr))
        self._paint_hints(qimg, (x0, y0), s)
        vsx, vsy = self.image.width * s, self.image.height * s
        self.canvas.set_scene_image(
            qimg,
            scene_origin=(x0 * s, y0 * s),
            scene_span=(vsx, vsy),
            image_rect=(x0, y0, x1, y1),
            scene_size=(vsx, vsy),
        )
        self._redraw_cursor_overlay_after_render()
        self._zoom_timer.stop()

    def _redraw_cursor_overlay_after_render(self) -> None:
        """底图重画后，把光标叠加层按新的坐标映射重新贴一次。

        不做这一步，擦除成功后旧圆圈会留在原地（要等下次重画才消失）。
        """
        self.canvas.clear_overlay()
        if self.tool == "erase" or self._pending_pt is not None:
            self._update_cursor_overlay()

    def _paint_hints(self, qimg, offset: tuple[int, int], s: float) -> None:
        """在底图上画**待连线提示**（第一个点的圆点 + 到光标的虚线）。

        注意：擦除圆圈**不在这里画**。它必须画在"叠加层"上——
        底图只在重画时才更新，把会跟着鼠标走的东西画进底图，就会在原地留下残影
        （实测表现为"擦除后原地多一个圈，要等下次重画才消失"）。
        """
        ox, oy = offset
        p = QPainter(qimg)
        p.setRenderHint(QPainter.Antialiasing)

        def to_disp(x: float, y: float) -> QPointF:
            return QPointF((x - ox) * s, (y - oy) * s)

        if self._pending_pt is not None:
            p0 = to_disp(*self._pending_pt)
            p.setPen(QPen(QColor(0, 255, 255), 2))
            p.setBrush(QColor(0, 255, 255))
            p.drawEllipse(p0, 6, 6)
            if self._cursor_img is not None:
                p.setPen(QPen(QColor(0, 255, 255, 170), 1, Qt.DashLine))
                p.drawLine(p0, to_disp(*self._cursor_img))
        p.end()

    def _on_zoom_changed(self, _scale: float) -> None:
        self._zoom_timer.start()

    def _refresh_for_zoom(self) -> None:
        """只在"该不该用高分辨率"变了的时候重画。"""
        want = self._crop_rect()
        if want is None:
            if self._view_mode != "full":
                self._refresh()
            return
        if self._view_mode == "zoom":
            r = self.canvas.rendered_image_rect()
            c = self.canvas
            poly = c.mapToScene(c.viewport().rect())
            rc = poly.boundingRect()
            v0 = c.to_image(rc.left(), rc.top())
            v1 = c.to_image(rc.right(), rc.bottom())
            if r is not None and r[0] <= min(v0[0], v1[0]) and r[1] <= min(v0[1], v1[1]) \
                    and r[2] >= max(v0[0], v1[0]) and r[3] >= max(v0[1], v1[1]):
                return
        self._refresh()

    def _update_status(self) -> None:
        cnt: dict[int, int] = {}
        for ln in self.lines:
            cnt[ln.layer_id] = cnt.get(ln.layer_id, 0) + 1
        parts = "，".join(f"{self.layer_name(k)} {v} 条" for k, v in sorted(cnt.items()))
        tip = {
            "pan": "浏览：Shift+左键 / 中键拖动平移，滚轮缩放",
            "line": "加线：左键点第一个点 → 再左键点第二个点（或右键）连成直线；Esc 取消",
            "erase": "擦除：按住左键拖动擦掉圈内线段",
        }[self.tool]
        self.lbl_status.setText(f"{tip}　|　共 {len(self.lines)} 条线（{parts or '无'}）")

    # ================================================================== 交互
    # 注意：``ImageCanvas`` 的 ``clicked`` / ``cursor_moved`` / ``dragged`` / ``right_clicked``
    # 发出的**已经是原图坐标**（画布内部做了 mapToScene + to_image）。
    # 这里千万不能再调 ``canvas.to_image``，否则会二次放大 1/s 倍（≈3 倍），
    # 表现就是"点一下，线跑到画面外"。
    def _on_move(self, ix: float, iy: float) -> None:
        """鼠标移动：**不再整幅重画**。

        红线画在底图上，重画一次要重新光栅化所有线条；鼠标一动就重画会非常卡。
        这里把"第一个点 + 到光标的虚线 + 擦除圆圈"做成一小块**叠加层**贴上去，
        代价只和圆圈大小有关。
        """
        self._cursor_img = (ix, iy)
        if self._erasing:
            self._erase_at(ix, iy)
            return
        if self.tool == "erase":
            self._update_cursor_overlay()
        elif self._pending_pt is not None:
            self._update_cursor_overlay()
        else:
            self.canvas.clear_overlay()

    def _update_cursor_overlay(self) -> None:
        """只在光标附近生成一小块含"提示点/虚线/圆圈"的叠加图。

        关键：叠加层是贴在**场景**里的图元，所以这一小块必须按**场景单位**画，
        坐标与半径都要用 ``canvas.to_scene`` 换算（不能直接拿原图像素当场景像素，
        否则放大时圆圈会偏、直径也不对）。代价只和圆圈大小有关，不涉及整幅重画。
        """
        if self._cursor_img is None and self._pending_pt is None:
            self.canvas.clear_overlay()
            return
        pts = [p for p in (self._pending_pt, self._cursor_img) if p is not None]
        if not pts:
            self.canvas.clear_overlay()
            return

        def sc(px: float, py: float) -> tuple[float, float]:
            return self.canvas.to_scene(px, py)

        # 以光标为中心、按原图半径 r 换算成场景半径
        cx0, cy0 = sc(*pts[0])
        cx1, cy1 = sc(*pts[-1])
        r_img = float(self.sp_brush.value())
        sx, _sy = sc(r_img, 0.0)
        ox0, _oy0 = sc(0.0, 0.0)
        r_scene = max(4.0, abs(sx - ox0))
        pad = r_scene + 8.0
        x0 = min(cx0, cx1) - pad
        y0 = min(cy0, cy1) - pad
        x1 = max(cx0, cx1) + pad
        y1 = max(cy0, cy1) + pad
        # 限制大小，避免第一个点离光标很远时生成巨图
        if x1 - x0 > 4000 or y1 - y0 > 4000:
            x0, y0 = cx1 - pad, cy1 - pad
            x1, y1 = cx1 + pad, cy1 + pad
        w = max(2, int(round(x1 - x0)))
        h = max(2, int(round(y1 - y0)))
        tile = np.zeros((h, w, 4), np.uint8)

        def to_tile(px: float, py: float) -> tuple[float, float]:
            return px - x0, py - y0

        if self._pending_pt is not None:
            px, py = to_tile(*sc(*self._pending_pt))
            if self._cursor_img is not None:
                cx2, cy2 = to_tile(*sc(*self._cursor_img))
                _draw_line(tile, px, py, cx2, cy2, (255, 255, 0, 200), 1.5, dash=8.0)
            _draw_disc(tile, px, py, 3.5, fill=(255, 255, 0, 255))
        if self.tool == "erase" and self._cursor_img is not None:
            cx2, cy2 = to_tile(*sc(*self._cursor_img))
            # 红色虚线圆圈、内部不填充（BGRA：B=80, G=80, R=235）
            _draw_disc(tile, cx2, cy2, r_scene, ring=(80, 80, 235, 255),
                       ring_w=1.5, dash=9.0)
        self.canvas.set_overlay(
            (int(round(x0)), int(round(y0)), int(round(x0)) + w, int(round(y0)) + h),
            tile,
        )

    def _on_click(self, ix: float, iy: float) -> None:
        if self.tool == "line":
            if self._pending_pt is None:
                self._pending_pt = (ix, iy)
            else:
                self._commit_line(self._pending_pt, (ix, iy))
            self._refresh()
            return
        if self.tool == "erase":
            self._push_undo_once()
            self._erasing = True
            self._erase_at(ix, iy)

    def _on_right_click(self, ix: float, iy: float) -> None:
        """右键 = 用当前光标位置结束这条线（方便"确定第二个点"）。"""
        if self.tool == "line" and self._pending_pt is not None:
            self._commit_line(self._pending_pt, (ix, iy))
            self._refresh()

    def _on_drag(self, ix: float, iy: float) -> None:
        self._cursor_img = (ix, iy)
        if self._erasing:
            self._erase_at(ix, iy)

    def _on_drag_release(self) -> None:
        self._erasing = False
        self._erased_any = False

    def _commit_line(self, a: tuple[float, float], b: tuple[float, float]) -> None:
        """把两个点连成一条直线段，加入当前选中图层。"""
        if float(np.hypot(b[0] - a[0], b[1] - a[1])) < 2.0:
            self._pending_pt = None
            return
        self._push_undo()
        lid = self._new_line_layer()
        pts = np.array([a, b], dtype=np.float32)
        self.lines.append(
            Line(pts, kind="outer", crystal_id=0, layer_id=lid,
                 attrs={"closed": False, "manual": True, "length": float(np.hypot(b[0]-a[0], b[1]-a[1]))})
        )
        self._pending_pt = None

    def _new_line_layer(self) -> int:
        data = self.cmb_layer.currentData()
        if data is not None:
            return int(data)
        for i, lay in enumerate(self.stack.layers):
            if lay.visible:
                return i
        return 0

    # ------------------------------------------------------------------ 擦除
    def _push_undo_once(self) -> None:
        """一次拖动只记一次撤销。"""
        if not self._erased_any:
            self._push_undo()
            self._erased_any = True

    def _erase_at(self, x: float, y: float) -> None:
        """擦掉圆圈内的线段（自由擦除）。

        关键点：轮廓线的顶点是**稀疏**的（一条几百像素的长边只有两个端点），
        只判断顶点的话，圆圈落在一条长边中间时什么都不会发生。所以先把每条线
        **加密**到"点距不超过一个圆周步长"，再逐点判断、保留圈外的点、
        把剩下的连续点各自拼成新线——于是圆圈扫过哪里就断开哪里，和橡皮一样。

        性能：先用线段包围盒做廉价排除；只有**真的擦掉了东西**才重绘画布。
        """
        r = float(self.sp_brush.value())
        r2 = r * r
        step = max(2.0, r * 0.4)
        out: list[Line] = []
        changed = False
        for ln in self.lines:
            p = np.asarray(ln.points, np.float64)
            if p.shape[0] < 2:
                continue
            x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
            y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
            if x < x0 - r or x > x1 + r or y < y0 - r or y > y1 + r:
                out.append(ln)
                continue
            dense = self._densify(p, step)
            keep = (dense[:, 0] - x) ** 2 + (dense[:, 1] - y) ** 2 > r2
            if keep.all():
                out.append(ln)
                continue
            changed = True
            idx = np.nonzero(keep)[0]
            if idx.size == 0:
                continue
            splits = np.nonzero(np.diff(idx) > 1)[0]
            bounds = [0] + [int(k) + 1 for k in splits] + [idx.size]
            for a, b in zip(bounds[:-1], bounds[1:]):
                sel = idx[a:b]
                if sel.size < 2:
                    continue
                out.append(
                    Line(dense[sel].astype(np.float32), ln.kind, ln.crystal_id, ln.layer_id,
                         {**ln.attrs, "closed": False})
                )
        if changed:
            self.lines = out
            self._refresh()

    @staticmethod
    def _densify(p: np.ndarray, step: float) -> np.ndarray:
        """在每条线段上按 ``step`` 插入中间点（端点保留）。"""
        seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
        out = [p[:1]]
        for i in range(p.shape[0] - 1):
            n = int(seg[i] // step)
            if n >= 1:
                t = np.linspace(0.0, 1.0, n + 2)[1:-1]
                out.append(p[i] + t[:, None] * (p[i + 1] - p[i]))
            out.append(p[i + 1: i + 2])
        return np.vstack(out)

    # ================================================================== 编辑操作
    @staticmethod
    def _clone(lines: list[Line]) -> list[Line]:
        return [
            Line(ln.points.copy(), ln.kind, ln.crystal_id, ln.layer_id, dict(ln.attrs))
            for ln in lines
        ]

    def _push_undo(self) -> None:
        self._undo.append(self._clone(self.lines))
        if len(self._undo) > 64:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, snapshot: list[Line]) -> None:
        self.lines = self._clone(snapshot)
        self._pending_pt = None
        self._refresh()

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(self._clone(self.lines))
        self._restore(self._undo.pop())

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(self._clone(self.lines))
        self._restore(self._redo.pop())

    def reset_lines(self) -> None:
        self._push_undo()
        self.lines = self._clone(
            [ln for ln in self.stack.all_lines() if self.stack.layers[ln.layer_id].visible]
        )
        self._pending_pt = None
        self._refresh()

    # ================================================================== 键盘
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_Escape:
            self._pending_pt = None
            self._refresh()
        elif key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()
        elif key == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            self.redo()
        elif key == Qt.Key_R:
            self.reset_lines()
        elif key == Qt.Key_1:
            self._set_tool("pan")
        elif key == Qt.Key_2:
            self._set_tool("line")
        elif key == Qt.Key_3:
            self._set_tool("erase")
        else:
            super().keyPressEvent(event)

    # ================================================================== 导出
    def export_final(self) -> None:
        img = self.image
        stem = Path(img.path).stem if img.path else "result"
        # 起始目录 = 上次导出目录（没有则退回上次打开图片的目录）
        default = os.path.join(app_settings.initial_dir("export"), f"{stem}_edited.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "输出最终图片", default, "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif)"
        )
        if not path:
            return

        if not os.path.splitext(path)[1]:
            path += ".png"
        try:
            mode = self.cmb_out_mode.currentData()
            # 透明底只能存 PNG：选了 JPEG 会被合成为白底，等于白设，这里直接纠正
            if mode == "transparent" and os.path.splitext(path)[1].lower() in (".jpg", ".jpeg"):
                path = os.path.splitext(path)[0] + ".png"
                notify(self, f"透明底只能存 PNG，已改存\n{os.path.basename(path)}", 2600)
            # 输出用**全部可见图层**（不受顶部筛选影响），按各图层样式绘制
            lines_all = [ln for ln in self.lines if self.stack.layers[ln.layer_id].visible]
            groups = self._style_groups(lines_all, 1.0)
            bgr, alpha = render_groups(
                img.size,
                groups,
                base_bgr=img.bgr if mode == "overlay" else None,
                mode=mode,
                reference_color=self.reference_color,
                scale=1.0,
                supersample=SS_DEFAULT,   # 导出才做抗锯齿（4x 超采样）
            )
            save_image(path, bgr, alpha=alpha, expected_shape=(img.height, img.width))
        except Exception as exc:  # noqa: BLE001
            error(self, "导出失败", str(exc))
            return
        # 与主窗口共用"上次导出目录"
        app_settings.remember_export(path)
        notify_success(self, "保存成功！")
