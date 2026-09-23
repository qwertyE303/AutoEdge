"""主窗口。

工作流（与使用者的实际用法一致）：

1. 打开图片；
2. 在**当前图层**上「圈定区域」（可圈多个，重叠时取并集）；
3. **必须**用「绿点」在晶体上点几下（红点用来标掉被误判的基底）——
   实测绿点对精度影响极大，没有绿点不予分析；
4. 点「预览」才开始分析并渲染线条（不是每动一下就重算）；
5. 需要不同阈值时**新建图层**：每个图层有独立的选区/提示点/阈值/线条样式，
   最终显示与输出是所有可见图层的叠加；
6. 「生成初稿」进入编辑窗口，可对**任意图层**的线条删补，最后输出尺寸与原图一致的成品。
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from autoedge.config import LineStyle, SegConfig
from autoedge.debuglog import default_log_path, write_debug_log
from autoedge.imgio import ImageData, imread, save_image
from autoedge.pipeline import AnalysisResult, LayerStack, render_groups
from autoedge.render import SS_DEFAULT
from autoedge.roi import HintPoint, RoiSet
from autoedge_gui.canvas import ImageCanvas, bgr_to_qimage
from autoedge_gui.edit_window import EditWindow
from autoedge_gui.notifications import error, notify, notify_success
from autoedge_gui.layer_panel import LayerPanel
from autoedge_gui.params_panel import ParamPanel

__all__ = ["MainWindow", "run"]

PREVIEW_MAX_SIDE = 1600


class _AnalyzeWorker(QObject):
    """后台分析线程：分析一个图层或全部可见图层。"""

    done = Signal(object, str)
    failed = Signal(str)

    def __init__(self, stack: LayerStack, index: int, all_layers: bool) -> None:
        super().__init__()
        self.stack = stack
        self.index = index
        self.all_layers = all_layers

    def run(self) -> None:
        try:
            if self.all_layers:
                results = self.stack.preview_all()
                self.done.emit(results, "")
            else:
                res = self.stack.preview(self.index)
                self.done.emit([res], "")
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoEdge — 层状晶体描边")
        self.resize(1700, 1000)

        self.image: ImageData | None = None
        self.stack: LayerStack | None = None
        self.edit_window: EditWindow | None = None
        self._thread: QThread | None = None
        self._worker: _AnalyzeWorker | None = None
        self._busy = False
        self._syncing_panel = False

        self._build_ui()
        self._build_menu()
        self._update_workflow_state()

    # ================================================================== 便捷访问
    @property
    def _layer(self):
        return None if self.stack is None else self.stack.active

    @property
    def _analyzer(self):
        lay = self._layer
        return None if lay is None else lay.analyzer

    @property
    def _result(self) -> AnalysisResult | None:
        lay = self._layer
        return None if lay is None else lay.result

    @property
    def roi(self) -> RoiSet:
        lay = self._layer
        if lay is None or lay.analyzer is None or lay.analyzer.roi is None:
            return RoiSet()
        return lay.analyzer.roi

    # ================================================================== 界面
    def _build_ui(self) -> None:
        self.canvas = ImageCanvas()
        self.canvas.cursor_moved.connect(self._on_cursor)
        self.canvas.clicked.connect(self._on_canvas_clicked)
        self.canvas.polygon_committed.connect(self._on_polygon_committed)
        self.canvas.polygon_in_progress.connect(lambda _pts: self._refresh_overlay())
        self.canvas.hint_remove_requested.connect(self._on_hint_remove)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)

        #: 放大到超过这个倍率（显示图/原图）时，按可见区域用原图重画，保证细节清晰
        self._hi_zoom = 1.05
        #: 当前显示的口径："full"（整幅预览）或 "zoom"（按可见区域重画的高清块）
        self._view_mode = "full"
        self._zoom_refresh_timer = QTimer(self)
        self._zoom_refresh_timer.setSingleShot(True)
        self._zoom_refresh_timer.setInterval(90)
        self._zoom_refresh_timer.timeout.connect(self._refresh_for_zoom)

        self.panel = ParamPanel(self._default_preset())
        self.panel.analysis_changed.connect(self._on_params_changed)
        self.panel.render_changed.connect(self._on_render_changed)
        self.panel.preset_load_requested.connect(self._load_preset)
        self.panel.preset_save_requested.connect(self._save_preset)
        self.panel.preset_delete_requested.connect(self._delete_preset)

        self.layer_panel = LayerPanel()
        self.layer_panel.current_changed.connect(self._on_layer_changed)
        self.layer_panel.visibility_changed.connect(self._on_layer_visibility)
        self.layer_panel.request_add.connect(self._add_layer)
        self.layer_panel.request_duplicate.connect(self._duplicate_layer)
        self.layer_panel.request_remove.connect(self._remove_layer)
        self.layer_panel.request_move.connect(self._move_layer)
        self.layer_panel.btn_preview_all.clicked.connect(
            lambda: self.start_analyze(all_layers=True)
        )

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_preview_area())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([360, 1000, 300])
        self.setCentralWidget(splitter)

        bar = QStatusBar()
        self.setStatusBar(bar)
        self.lbl_info = QLabel("就绪：请先「打开图片」")
        self.lbl_last = QLabel("")
        self.lbl_pos = QLabel("")
        self.lbl_line = QLabel("")
        self.lbl_log = QLabel("")
        #: 输出模式常驻显示：模式是全局记忆项（会跨会话保留），
        #: 摆在状态栏一眼可见，避免"忘了自己上次设成什么"而导错图
        self.lbl_mode = QLabel("")
        bar.addWidget(self.lbl_info, 1)
        bar.addPermanentWidget(self.lbl_mode)
        bar.addPermanentWidget(self.lbl_log)
        bar.addPermanentWidget(self.lbl_last)
        bar.addPermanentWidget(self.lbl_line)
        bar.addPermanentWidget(self.lbl_pos)
        self._update_mode_label()

    def _build_left(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(6, 6, 6, 6)

        # ---- 第一步：圈定区域 ----
        box1 = QGroupBox("① 圈定要描边的区域（可圈多个，只作用于当前图层）")
        v1 = QVBoxLayout(box1)
        self.btn_roi = QPushButton("圈定区域")
        self.btn_roi.setCheckable(True)
        self.btn_roi.clicked.connect(lambda: self._set_mode("roi"))
        self.btn_pan = QPushButton("浏览（拖动/缩放）")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setChecked(True)
        self.btn_pan.clicked.connect(lambda: self._set_mode("pan"))
        self.grp_mode = QButtonGroup(self)
        self.grp_mode.setExclusive(True)
        self.grp_mode.addButton(self.btn_roi)
        self.grp_mode.addButton(self.btn_pan)
        row = QHBoxLayout()
        row.addWidget(self.btn_roi)
        row.addWidget(self.btn_pan)
        v1.addLayout(row)
        self.lbl_roi = QLabel("尚未圈定")
        self.lbl_roi.setWordWrap(True)
        v1.addWidget(self.lbl_roi)
        row2 = QHBoxLayout()
        b_undo = QPushButton("删除最后一个框")
        b_undo.clicked.connect(self._remove_last_roi)
        b_clear = QPushButton("清空全部")
        b_clear.clicked.connect(self._clear_roi)
        row2.addWidget(b_undo)
        row2.addWidget(b_clear)
        v1.addLayout(row2)
        outer.addWidget(box1)

        # ---- 第二步：提示点 ----
        box2 = QGroupBox("② 提示点（必须至少点一个绿点才能预览）")
        v2 = QVBoxLayout(box2)
        self.btn_hint_pos = QPushButton("绿点：这里是晶体")
        self.btn_hint_pos.setCheckable(True)
        self.btn_hint_pos.clicked.connect(lambda: self._set_mode("hint+"))
        self.btn_hint_neg = QPushButton("红点：这里不是晶体")
        self.btn_hint_neg.setCheckable(True)
        self.btn_hint_neg.clicked.connect(lambda: self._set_mode("hint-"))
        self.grp_mode.addButton(self.btn_hint_pos)
        self.grp_mode.addButton(self.btn_hint_neg)
        rowh = QHBoxLayout()
        rowh.addWidget(self.btn_hint_pos)
        rowh.addWidget(self.btn_hint_neg)
        v2.addLayout(rowh)
        self.lbl_hint = QLabel("绿点 0 个，红点 0 个")
        v2.addWidget(self.lbl_hint)
        b_hint_clear = QPushButton("清除提示点")
        b_hint_clear.clicked.connect(self._clear_hints)
        v2.addWidget(b_hint_clear)
        outer.addWidget(box2)

        # ---- 第三步：预览 ----
        box3 = QGroupBox("③ 预览（只重算当前图层）")
        v3 = QVBoxLayout(box3)
        self.btn_preview = QPushButton("▶ 预览")
        self.btn_preview.setStyleSheet("font-weight: bold;")
        self.btn_preview.clicked.connect(lambda: self.start_analyze(all_layers=False))
        v3.addWidget(self.btn_preview)
        v3.addWidget(QLabel("改完参数后点「预览」才会重新计算。\n换图层自动保留其他图层的结果。"))
        outer.addWidget(box3)

        outer.addWidget(self.panel, 1)

        # ---- 第四步：出图 ----
        box4 = QGroupBox("④ 出图")
        v4 = QVBoxLayout(box4)
        self.btn_draft = QPushButton("手动编辑")
        self.btn_draft.setToolTip("生成算法初稿并打开编辑窗口，可手动加线/擦线后再输出")
        self.btn_draft.clicked.connect(self.make_draft)
        v4.addWidget(self.btn_draft)
        self.btn_save = QPushButton("保存结果…")
        self.btn_save.clicked.connect(self.save_result)
        v4.addWidget(self.btn_save)
        outer.addWidget(box4)

        return w

    def _build_preview_area(self) -> QWidget:
        """画布 + 它自己的显示开关（显示晶体区域 / 显示选区与提示点）。"""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 0)
        bar.addWidget(QLabel("预览显示："))
        self.chk_show_mask = QCheckBox("显示晶体区域")
        self.chk_show_mask.setChecked(True)
        self.chk_show_mask.setToolTip("把算法判定为晶体的区域用半透明绿色盖出来（只看当前图层）")
        self.chk_show_mask.stateChanged.connect(lambda _s: self._refresh_overlay())
        bar.addWidget(self.chk_show_mask)
        self.chk_show_roi = QCheckBox("显示选区与提示点")
        self.chk_show_roi.setChecked(True)
        self.chk_show_roi.setToolTip("把圈定的选区多边形与绿点/红点画在预览图上")
        self.chk_show_roi.stateChanged.connect(lambda _s: self._refresh_overlay())
        bar.addWidget(self.chk_show_roi)
        bar.addStretch(1)
        v.addLayout(bar)
        v.addWidget(self.canvas, 1)
        return w

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.layer_panel)
        w.setMinimumWidth(250)
        w.setMaximumWidth(380)
        return w

    def _default_preset(self):
        from autoedge.config import Preset

        return Preset()

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("文件(&F)")
        a = QAction("打开图片…", self)
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self.open_image)
        m.addAction(a)

        # 最近打开的文件：每次展开时重建，文件被删/改名后条目会立即消失
        self.menu_recent = QMenu("打开最近文件", self)
        self.menu_recent.aboutToShow.connect(self._rebuild_recent_menu)
        m.addMenu(self.menu_recent)
        self._rebuild_recent_menu()

        m.addSeparator()
        a = QAction("保存结果…", self)
        a.setShortcut("Ctrl+S")
        a.triggered.connect(self.save_result)
        m.addAction(a)
        m.addSeparator()
        a = QAction("退出", self)
        a.triggered.connect(self.close)
        m.addAction(a)

        mv = self.menuBar().addMenu("视图(&V)")
        for text, seq, slot in (
            ("适应窗口", "Ctrl+0", lambda: self.canvas.fit_to_window()),
            ("100%", "Ctrl+1", lambda: self.canvas.zoom_to(1.0)),
        ):
            a = QAction(text, self)
            a.setShortcut(seq)
            a.triggered.connect(slot)
            mv.addAction(a)

        mh = self.menuBar().addMenu("帮助(&H)")
        a = QAction("操作说明", self)
        a.triggered.connect(self.show_help)
        mh.addAction(a)

    def _rebuild_recent_menu(self) -> None:
        """重建「打开最近文件」子菜单。

        每次展开时重建（不是只在打开图片后重建），这样文件被删掉/改名后
        条目会立刻消失，不会留下点了没反应的死项。列表里已不存在的文件
        顺手从设置里剔除，避免最近记录被失效路径占满。
        """
        from autoedge import settings as app_settings

        self.menu_recent.clear()
        entries = app_settings.recent_entries()
        gone = [p for p, ok in entries if not ok]
        for p in gone:
            app_settings.remove_recent(p)
        entries = [(p, ok) for p, ok in entries if ok]

        if not entries:
            act = self.menu_recent.addAction("（暂无最近文件）")
            act.setEnabled(False)
            return

        for i, (path, _ok) in enumerate(entries, start=1):
            act = QAction(f"{i}  {os.path.basename(path)}", self)
            act.setToolTip(path)
            act.triggered.connect(lambda _checked=False, p=path: self._load_image(p, remember=False))
            self.menu_recent.addAction(act)

        self.menu_recent.addSeparator()
        act = self.menu_recent.addAction("清除最近记录")
        act.triggered.connect(self._clear_recent)

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "操作说明",
            "① 圈定区域：点「圈定区域」后在图上逐点画多边形，右键或双击结束一个框。\n"
            "　 一块晶体圈一个紧框效果最好（框里留太多基底会拉偏基底色）。\n\n"
            "② 提示点：点「绿点：这里是晶体」后在晶体上点几下（**必须至少一个**）；\n"
            "　 基底上被误判成晶体的地方用红点标掉。\n\n"
            "③ 预览：点「▶ 预览」才会分析并画线，只重算当前图层。\n"
            "　 预览上方两个开关控制显示：「显示晶体区域」（算法判定为晶体的绿色区域）、\n"
            "　 「显示选区与提示点」。\n\n"
            "④ 图层（右侧）：每个图层有独立的选区/提示点/阈值/线条样式，最终叠加输出。\n"
            "　 例如：图层1 用高阈值描明显轮廓，再「新建」图层2 用低阈值描单层区。\n"
            "　 「预览全部图层」会依次重算所有可见图层（每层 1~10 秒）。\n\n"
            "⑤ 出图：点「手动编辑」打开编辑窗口，三个工具——\n"
            "　 「浏览」平移缩放；「加线」左键点两个点连成直线（右键也可结束，线宽可调）；\n"
            "　 「擦除」出现圆圈，按住左键拖动擦掉圈内线段（大小可调）。\n"
            "　 编辑完点「输出最终图片…」，尺寸与原图严格相同。\n\n"
            "⑥ 文件：菜单「文件 → 打开最近文件」列出最近 10 张打开过的图；\n"
            "　 打开图片与导出图片的对话框都会**记住上次用过的目录**（关掉程序也保留）。\n\n"
            "画布：滚轮缩放；Shift+左键或中键拖动平移；Ctrl+0 适应窗口；Ctrl+1 100%。",
        )

    # ================================================================== 模式
    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        if mode != "roi":
            self._refresh_overlay()
        hints = {
            "pan": "浏览模式：拖动/缩放",
            "roi": "圈定区域：左键逐点画多边形，右键或双击结束",
            "hint+": "绿点模式：在晶体上点击（右键删除最近的点）",
            "hint-": "红点模式：在不是晶体的地方点击（右键删除最近的点）",
        }
        self.lbl_info.setText(hints.get(mode, ""))

    # ================================================================== 打开图片
    def open_image(self) -> None:
        """「打开图片…」：起始目录 = 上次打开过图片的目录。"""
        from autoedge import settings as app_settings

        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开显微镜照片",
            app_settings.initial_dir("open"),
            "图像 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;所有文件 (*)",
        )
        if not path:
            return
        self._load_image(path)

    def _load_image(self, path: str, *, remember: bool = True) -> None:
        """载入图片并重建图层栈。

        Args:
            path: 图片路径。
            remember: 是否记进"最近文件"与"上次目录"。从最近文件菜单再次打开
                时传 ``False``——它已经在列表里，重复记只会白白搅动顺序。
        """
        from autoedge import settings as app_settings

        try:
            img = imread(path)
        except Exception as exc:  # noqa: BLE001
            # 最近文件里的图片常被移动/重命名：这时从列表里摘掉即可，
            # 不必弹对话框——它会当场从菜单消失，本身就是最直接的反馈。
            if not os.path.isfile(path) and path in app_settings.recent_files():
                app_settings.remove_recent(path)
                self.lbl_info.setText(f"这张图已经不在了，已从最近文件里移除：{path}")
                self._rebuild_recent_menu()
                return
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        self.image = img
        # 选区与提示点每次导入图片重新画，因此图层栈也重建
        self.stack = LayerStack(img, self.panel.collect().seg)
        self.stack.layers[0].style = self.panel.collect().outer
        self._sync_panel_from_layer()
        self.layer_panel.rebuild(self.stack.layers, self.stack.current)
        self.canvas.set_mode("roi")
        self.btn_roi.setChecked(True)
        self.setWindowTitle(f"AutoEdge — {os.path.basename(path)}  ({img.width}x{img.height})")
        self._render_base()
        self._update_workflow_state()
        self.lbl_info.setText(
            f"已载入 {img.width}x{img.height}：请先「圈定区域」，点几个绿点，然后点「▶ 预览」"
        )
        # 只有真正打开成功才记，打不开的文件不会污染最近列表
        if remember:
            app_settings.remember_file(path)
        self._rebuild_recent_menu()

    def _clear_recent(self) -> None:
        from autoedge import settings as app_settings

        app_settings.clear_recent()
        self._rebuild_recent_menu()
        self.lbl_info.setText("已清除最近文件记录")

    # ================================================================== 图层
    def _sync_panel_from_layer(self) -> None:
        lay = self._layer
        if lay is None or lay.analyzer is None:
            return
        self._syncing_panel = True
        self.panel.load_layer(lay.style, lay.analyzer.seg)
        self._syncing_panel = False

    def _flush_panel_to_layer(self) -> None:
        """把参数面板当前值写回**当前图层**。

        每个图层**各自独立**使用自己的参数与线条样式（阈值、最小面积、简化容差、
        线型/颜色/线宽/不透明度都是各层一份）。所以这里只写当前图层，
        绝不广播到其它图层——否则新建图层后一预览就会把别的图层改掉。

        "通用"只体现在两处：
        * 预设可以导入任意图层（由「预设」区触发，作用范围由使用者选择）；
        * **新建图层时继承当前图层的设置**，不会重置成默认值。

        输出模式不在这里：它是**全局记忆项**（见 :mod:`autoedge.settings`），
        不随图层变化。
        """
        lay = self._layer
        if lay is None or lay.analyzer is None:
            return
        style, seg = self.panel.apply_to_layer(lay.style, lay.analyzer.seg)
        lay.style, lay.analyzer.seg = style, seg
        lay.analyzer.update_seg(seg)

    def _on_layer_changed(self, index: int) -> None:
        if self.stack is None or not (0 <= index < len(self.stack.layers)):
            return
        self.stack.current = index
        self._sync_panel_from_layer()
        self._update_roi_label()
        self._refresh_overlay()
        lay = self.stack.layers[index]
        self.lbl_info.setText(
            f"当前图层：{lay.name}"
            + ("（参数已改，点「▶ 预览」重算本层）" if lay.stale else "")
        )

    def _on_layer_visibility(self, index: int, visible: bool) -> None:
        if self.stack is None or not (0 <= index < len(self.stack.layers)):
            return
        self.stack.layers[index].visible = bool(visible)
        self._refresh_overlay()

    def _add_layer(self) -> None:
        if self.stack is None:
            QMessageBox.information(self, "请先打开图片", "还没有载入任何图片。")
            return
        self._flush_panel_to_layer()
        lay = self.stack.add_layer()
        self._sync_panel_from_layer()
        self.layer_panel.rebuild(self.stack.layers, self.stack.current)
        self.showNormal()
        self.lbl_info.setText(
            f"已新建「{lay.name}」——请重新圈定区域、点绿点、调阈值，然后点「▶ 预览」"
        )

    def _duplicate_layer(self) -> None:
        if self.stack is None:
            return
        self._flush_panel_to_layer()
        new = self.stack.duplicate_layer()
        if new is None:
            return
        self._sync_panel_from_layer()
        self.layer_panel.rebuild(self.stack.layers, self.stack.current)
        self.lbl_info.setText(f"已复制为「{new.name}」（含选区、提示点与阈值）")

    def _remove_layer(self) -> None:
        if self.stack is None:
            return
        if len(self.stack.layers) <= 1:
            QMessageBox.information(self, "无法删除", "至少要保留一个图层。")
            return
        name = self._layer.name if self._layer else ""
        if self.stack.remove_layer():
            self._sync_panel_from_layer()
            self.layer_panel.rebuild(self.stack.layers, self.stack.current)
            self._update_roi_label()
            self._refresh_overlay()
            self.lbl_info.setText(f"已删除图层「{name}」")

    def _move_layer(self, delta: int) -> None:
        if self.stack is None:
            return
        self._flush_panel_to_layer()
        if self.stack.move_layer(int(delta)):
            self.layer_panel.rebuild(self.stack.layers, self.stack.current)
            self._refresh_overlay()

    # ================================================================== 预设
    #: 预设只含"参数 + 线条样式"；载入时套用到所有图层（可自行再逐层微调）。
    def _apply_settings_to_all_layers(self, seg, style) -> int:
        """把参数与线条样式套用到**所有图层**（只在载入预设时用）。"""
        if self.stack is None:
            return 0
        n = 0
        for lay in self.stack.layers:
            if lay.analyzer is None:
                continue
            lay.analyzer.update_seg(SegConfig(**seg.to_dict()).normalized())
            lay.style = LineStyle(
                enabled=style.enabled,
                style=style.style,
                color=style.color,
                width=style.width,
                alpha=style.alpha,
            )
            lay.stale = True
            n += 1
        return n

    def _save_preset(self) -> None:
        """把面板当前的参数 + 线条样式存成预设。

        **不需要先打开图片**——预设就是一套通用设置，任何时候都能存。
        """
        name, ok = QInputDialog.getText(self, "保存预设", "预设名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            from autoedge.presets import save_preset

            # 先取面板当前值；有图层时以当前图层为准（面板已与图层同步）
            preset = self.panel.collect()
            seg, style = preset.seg, preset.outer
            lay = self._layer
            if lay is not None and lay.analyzer is not None:
                seg, style = lay.analyzer.seg, lay.style
            path = save_preset(name, seg, style)
        except Exception as exc:  # noqa: BLE001
            error(self, "保存预设失败", str(exc))
            return
        self.panel.refresh_presets(select=name)
        self.lbl_info.setText(f"已保存预设「{name}」→ {path}")
        notify(self, f"已保存预设「{name}」")

    def _load_preset(self, name: str) -> None:
        """载入预设并套用到所有图层。"""
        if self.stack is None:
            notify(self, "请先打开图片")
            return
        from autoedge.presets import load_preset

        data = load_preset(name)
        if data is None:
            error(self, "载入预设失败", f"读不到预设「{name}」。")
            self.panel.refresh_presets()
            return
        n = self._apply_settings_to_all_layers(data.seg, data.style)
        # 面板显示当前图层的新值
        self._syncing_panel = True
        try:
            self.panel.load_layer(data.style, data.seg)
        finally:
            self._syncing_panel = False
        self.layer_panel.rebuild(self.stack.layers, self.stack.current)
        self._refresh_overlay()
        self.lbl_info.setText(f"已载入预设「{name}」到 {n} 个图层 —— 点「▶ 预览」重新分析")
        notify(self, f"已载入预设「{name}」")

    def _delete_preset(self, name: str) -> None:
        from autoedge.presets import delete_preset

        if not name:
            return
        ans = QMessageBox.question(self, "删除预设", f"确定删除预设「{name}」吗？")
        if ans != QMessageBox.Yes:
            return
        if delete_preset(name):
            self.panel.refresh_presets()
            notify(self, f"已删除预设「{name}」")
        else:
            error(self, "删除失败", f"删不掉「{name}」。")

    # ================================================================== 参数变化
    def _on_params_changed(self) -> None:
        """分析类参数变化。

        **不立刻清空已有结果**——只标记该图层"待预览"，画面保持不动，
        等使用者再点「预览」时统一刷新。
        """
        if self._syncing_panel or self.stack is None:
            return
        lay = self._layer
        if lay is None:
            return
        self._flush_panel_to_layer()
        lay.stale = True
        self.layer_panel.rebuild(self.stack.layers, self.stack.current)
        self.lbl_info.setText("参数已修改 —— 点「▶ 预览」重新分析当前图层")

    def _mark_stale(self) -> None:
        """把当前图层标为"待预览"（不动 ROI，也不清别人的结果）。"""
        lay = self._layer
        if lay is None:
            return
        lay.stale = True
        if self.stack is not None:
            self.layer_panel.rebuild(self.stack.layers, self.stack.current)
        self._update_workflow_state()

    def _invalidate_result(self, why: str) -> None:
        """选区或提示点改变：作废该图层结果并提示重新预览。

        **不调用 set_roi(None)** —— 那会把使用者刚点的提示点一起丢掉。
        分割缓存由 :meth:`Analyzer.segment` 按 ROI 指纹自行判断是否失效，
        这里只需要标记"待预览"。
        """
        self._mark_stale()
        self.lbl_info.setText(f"{why} —— 点「▶ 预览」重新分析")

    def _on_render_changed(self) -> None:
        if self._syncing_panel:
            return
        self._flush_panel_to_layer()
        # 输出模式也走这个信号：模式是全局项，状态栏要跟着变
        self._update_mode_label()
        self._refresh_overlay()

    # ================================================================== 分析
    def start_analyze(self, all_layers: bool = False) -> None:
        if self.stack is None or self._analyzer is None:
            QMessageBox.information(self, "请先打开图片", "还没有载入任何图片。")
            return
        lay = self._layer
        roi = lay.analyzer.roi
        if not all_layers:
            if roi is None or not roi.polygons:
                QMessageBox.critical(
                    self,
                    "还没有圈定区域",
                    f"当前图层「{lay.name}」还没有圈定任何区域。\n\n"
                    "请先点「圈定区域」，在图上逐点画出要描边的晶体（右键或双击结束）。",
                )
                return
            if not roi.positive_points():
                QMessageBox.critical(
                    self,
                    "还没有绿点",
                    f"当前图层「{lay.name}」还没有绿点，无法分析。\n\n"
                    "请点「绿点：这里是晶体」，在晶体上点几下（建议 3~5 个），再预览。\n"
                    "绿点用于锁定晶体位置，是最关键的一步。",
                )
                return
        else:
            missing = [
                l.name
                for l in self.stack.layers
                if l.visible
                and (l.analyzer is None or l.analyzer.roi is None or not l.analyzer.roi.positive_points())
            ]
            if missing:
                QMessageBox.critical(
                    self,
                    "有图层还没有选区或绿点",
                    "以下图层缺少选区或绿点，无法分析：\n　"
                    + "\n　".join(missing)
                    + "\n\n每个图层都必须圈定区域并至少点一个绿点。",
                )
                return
        if self._busy:
            return
        self._flush_panel_to_layer()
        for l in self.stack.layers:
            if l.analyzer is not None:
                l.analyzer.set_roi(l.analyzer.roi)

        self._busy = True
        self.btn_preview.setEnabled(False)
        self.btn_preview.setText("分析中…")
        self.layer_panel.btn_preview_all.setEnabled(False)

        self._thread = QThread(self)
        self._worker = _AnalyzeWorker(self.stack, self.stack.current, all_layers)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_analyzed)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_analyzed(self, results: object, _msg: str) -> None:
        self._busy = False
        self.btn_preview.setEnabled(True)
        self.btn_preview.setText("▶ 预览")
        self.layer_panel.btn_preview_all.setEnabled(True)
        self._stop_thread()
        res_list: list[AnalysisResult] = list(results) if results else []  # type: ignore[arg-type]
        if not res_list:
            self.lbl_info.setText("分析完成但没有结果")
            return
        lay = self._layer
        result = lay.result if lay is not None else res_list[0]
        if result is None:
            result = res_list[-1]
        try:
            t = result.info.get("timing", {})
            seg = result.seg
            thr = f"{seg.threshold:.1f}" if seg is not None else "—"
            if len(res_list) > 1:
                summary = (
                    f"已分析 {len(res_list)} 个图层 | 当前「{lay.name if lay else ''}」："
                    f"晶体 {len(result.crystals)} 块 / 外轮廓 {len(result.outer_lines)} 条"
                    f" | 阈值 {thr} | 本层耗时 {t.get('total', 0):.2f}s"
                )
            else:
                summary = (
                    f"{result.size[0]}x{result.size[1]} | 图层「{lay.name if lay else ''}」："
                    f"晶体 {len(result.crystals)} 块 | 外轮廓 {len(result.outer_lines)} 条 | "
                    f"阈值 {thr} | 耗时 {t.get('total', 0):.2f}s"
                )
            self.lbl_info.setText(summary)
            self.lbl_last.setText(
                f"上次结果：{len(result.crystals)} 块晶体 / {len(result.outer_lines)} 条外轮廓"
            )
            self._refresh_overlay()
        except Exception:  # noqa: BLE001
            self.lbl_info.setText("渲染失败（详见弹窗）")
            QMessageBox.critical(self, "渲染失败", traceback.format_exc())
            return
        if self.stack is not None:
            self.layer_panel.rebuild(self.stack.layers, self.stack.current)
        self._update_workflow_state()

        # 每次预览都写一份诊断日志，出问题时直接把文件发回即可核对
        try:
            layers_info = []
            if self.stack is not None:
                for i, l in enumerate(self.stack.layers):
                    layers_info.append(
                        (
                            l.name,
                            l.analyzer.roi if l.analyzer is not None else RoiSet(),
                            l.analyzer.seg if l.analyzer is not None else self.panel.collect().seg,
                            len(l.lines),
                            l.visible,
                            i == self.stack.current,
                        )
                    )
            log_path = write_debug_log(
                default_log_path(),
                image_size=(self.image.width, self.image.height) if self.image else (0, 0),
                image_path=self.image.path if self.image else None,
                canvas=self.canvas,
                roi=self.roi,
                preset=self.panel.collect(),
                result=result,
                layers=layers_info,
            )
            self.lbl_log.setText(f"诊断日志：{log_path}")
        except Exception:  # noqa: BLE001
            self.lbl_log.setText("诊断日志写入失败")

        if result.seg is not None and result.seg.count == 0:
            QMessageBox.warning(
                self,
                "没有检测到晶体",
                "在当前区域里没有检测到晶体。可以尝试：\n"
                "· 把「描边阈值」调低（越低越灵敏）\n"
                "· 在晶体上多补几个绿点\n"
                "· 检查圈定的范围是否盖住了晶体\n\n"
                f"诊断：模式={result.seg.info.get('mode')}，"
                f"框内像素={result.seg.info.get('roi_pixels')}，"
                f"阈值 ΔE={result.seg.threshold:.2f}，"
                f"阈值后连通域={result.seg.info.get('components_after_threshold')}",
            )

    def _on_failed(self, msg: str) -> None:
        self._busy = False
        self.btn_preview.setEnabled(True)
        self.btn_preview.setText("▶ 预览")
        self.layer_panel.btn_preview_all.setEnabled(True)
        self._stop_thread()
        self.lbl_info.setText("分析失败")
        QMessageBox.critical(self, "分析失败", msg)

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self._worker = None

    # ================================================================== 渲染
    def _preview_scale(self) -> float:
        if self.image is None:
            return 1.0
        return min(1.0, PREVIEW_MAX_SIDE / max(self.image.width, self.image.height))

    def _render_base(self) -> None:
        """只画底图（尚未分析时）。"""
        if self.image is None:
            return
        import cv2

        s = self._preview_scale()
        if s < 1.0:
            small = cv2.resize(
                self.image.bgr,
                (int(self.image.width * s), int(self.image.height * s)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = self.image.bgr
        self.canvas.set_image(
            bgr_to_qimage(np.ascontiguousarray(small)),
            keep_view=False,
            original_size=self.image.size,
            keep_pending=False,
        )
        self._refresh_overlay()

    def _refresh_overlay(self, keep_view: bool = True) -> None:
        """把线条/遮罩/选区画到画布上。

        两种口径：

        * **整幅预览**（默认）——整张原图按 ``PREVIEW_MAX_SIDE`` 降采样，线条同比缩；
        * **高分辨率块**（放大到 1:1 以上时）——只取**视口对应的一小块原图**，
          按原像素渲染。这样放到很大时看到的仍是真实像素，而不是把预览摊大。
        """
        if self.image is None:
            return
        import cv2

        crop = self._crop_rect() if keep_view else None
        if crop is not None:
            self._view_mode = "zoom"
            self._render_highres(crop)
            return

        self._view_mode = "full"
        s = self._preview_scale()
        # 预览一律"原图 + 线条"：输出模式只影响导出，不影响预览
        groups = [] if self.stack is None else self.stack.style_groups()
        bgr, _alpha = render_groups(
            self.image.size, groups, base_bgr=self.image.bgr, mode="overlay",
            reference_color=self.panel.reference_color(), scale=s,
        )
        # 检测区域遮罩：只画当前图层（多层遮罩叠在一起会看不清）
        if self.chk_show_mask.isChecked():
            ana = self._analyzer
            if ana is not None and ana._seg is not None:
                mask = ana._seg.labels > 0
                if s < 1.0:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                bgr = self._tint_mask(bgr, mask)
        qimg = bgr_to_qimage(np.ascontiguousarray(bgr))
        if self.chk_show_roi.isChecked():
            self._paint_annotations(qimg)
        self.canvas.set_image(qimg, keep_view=keep_view, original_size=self.image.size)

    @staticmethod
    def _tint_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        tint = np.zeros_like(bgr)
        tint[:, :, 1] = 255
        return np.where(mask[:, :, None], (0.65 * bgr + 0.35 * tint).astype(np.uint8), bgr)

    def _render_highres(self, crop: tuple[int, int, int, int]) -> None:
        """按原图 1:1 渲染 ``crop`` 这块区域，再放回画布（坐标映射自动重算）。"""
        import cv2

        from autoedge.geometry import Line as _Line

        x0, y0, x1, y1 = crop
        cw, ch = x1 - x0, y1 - y0
        sub = self.image.bgr[y0:y1, x0:x1]
        s = self._preview_scale()
        disp_w = max(1, int(round(cw * s)))
        disp_h = max(1, int(round(ch * s)))

        if s < 1.0:
            base = cv2.resize(sub, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            base = sub

        # 线条：先从整图坐标平移到本块坐标，再按 s 缩到显示尺寸
        groups = [] if self.stack is None else self.stack.style_groups()
        scaled: list = []
        for lines, style in groups:
            shifted = []
            for ln in lines:
                pts = np.asarray(ln.points, np.float32).copy()
                pts[:, 0] = (pts[:, 0] - x0) * s
                pts[:, 1] = (pts[:, 1] - y0) * s
                shifted.append(_Line(pts, ln.kind, ln.crystal_id, ln.layer_id, dict(ln.attrs)))
            scaled.append((shifted, self._scale_style(style, s)))

        bgr, _a = render_groups(
            (disp_w, disp_h), scaled, base_bgr=base, mode="overlay",
            reference_color=self.panel.reference_color(), scale=1.0,
        )

        if self.chk_show_mask.isChecked():
            ana = self._analyzer
            if ana is not None and ana._seg is not None:
                full = ana._seg.labels > 0
                m = np.array(full[y0:y1, x0:x1], dtype=bool)
                if s < 1.0:
                    m = cv2.resize(m.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                bgr = self._tint_mask(bgr, m)

        qimg = bgr_to_qimage(np.ascontiguousarray(bgr))
        if self.chk_show_roi.isChecked():
            self._paint_annotations(qimg)
        # 场景口径恒为"整幅预览"（跨度 = 原图 × s），与 to_image/to_scene 严格互逆；
        # 切片只画了可见那一块，坐标体系不变，放大时也不会有任何跳变。
        vsx, vsy = self.image.width * s, self.image.height * s
        self.canvas.set_scene_image(
            qimg,
            scene_origin=(x0 * s, y0 * s),
            scene_span=(vsx, vsy),
            image_rect=(x0, y0, x1, y1),
            scene_size=(vsx, vsy),
        )
        self._zoom_refresh_timer.stop()

    @staticmethod
    def _scale_style(style, s: float):
        from autoedge.config import LineStyle

        return LineStyle(
            enabled=style.enabled,
            style=style.style,
            color=style.color,
            width=max(1, int(round(style.width * s))),
            alpha=style.alpha,
        )

    # ================================================================== 高分辨率放大
    def visible_image_rect(self) -> tuple[float, float, float, float] | None:
        """当前视口对应原图上的哪一块（``x0, y0, x1, y1``，原图像素）。"""
        if self.image is None:
            return None
        c = self.canvas
        poly = c.mapToScene(c.viewport().rect())
        r = poly.boundingRect()
        x0, y0 = c.to_image(r.left(), r.top())
        x1, y1 = c.to_image(r.right(), r.bottom())
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def _on_zoom_changed(self, _scale: float) -> None:
        """缩放/平移后延迟重画：放大时按可见区域用原图重画，保证细节清晰。"""
        if self.image is None:
            return
        self._zoom_refresh_timer.start()

    def _magnification(self) -> float:
        """原图 1 个像素在屏幕上占几个像素（交给画布统一算）。"""
        return self.canvas.magnification()

    def _crop_rect(self) -> tuple[int, int, int, int] | None:
        """需要按原图重画时，返回要渲染的原图区域；否则返回 ``None``。

        ``None`` 表示用整幅降采样预览就够了（还没放大到需要细节的程度）。
        """
        if self.image is None:
            return None
        if self._magnification() <= self._hi_zoom:
            return None
        rect = self.visible_image_rect()
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        W, H = self.image.width, self.image.height
        # 留一点余量，避免拖动时立刻又要重画
        pad = 0.12 * max(x1 - x0, y1 - y0)
        x0 = int(max(0, np.floor(x0 - pad)))
        y0 = int(max(0, np.floor(y0 - pad)))
        x1 = int(min(W, np.ceil(x1 + pad)))
        y1 = int(min(H, np.ceil(y1 + pad)))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return x0, y0, x1, y1

    def _refresh_for_zoom(self) -> None:
        """只在"该不该用高分辨率"这件事变了的时候重画，避免频繁重算。"""
        if self.image is None or self._busy:
            return
        want = self._crop_rect()
        if want is None:
            if self._view_mode != "full":
                self._view_mode = "full"
                self._refresh_overlay(keep_view=True)
            return
        # 已经在高分辨率模式下，且当前视口仍落在上次渲染的范围内 -> 不用重画
        if self._view_mode == "zoom":
            r = self.canvas.rendered_image_rect()
            v = self.visible_image_rect()
            if r is not None and v is not None:
                if r[0] <= v[0] and r[1] <= v[1] and r[2] >= v[2] and r[3] >= v[3]:
                    return
        self._refresh_overlay(keep_view=True)

    def _paint_annotations(self, qimg: QImage) -> None:
        """画选区与提示点：当前图层高亮，其他图层淡显。"""
        p = QPainter(qimg)
        p.setRenderHint(QPainter.Antialiasing)
        layers = self.stack.layers if self.stack is not None else []
        cur = self.stack.current if self.stack is not None else 0
        for i, lay in enumerate(layers):
            if lay.analyzer is None:
                continue
            roi = lay.analyzer.roi
            if roi is None:
                continue
            active = i == cur
            if active:
                pen_c, fill_c, hint_w = QColor(255, 200, 0), QColor(255, 200, 0, 40), 2
            else:
                pen_c, fill_c, hint_w = QColor(150, 150, 150), QColor(150, 150, 150, 20), 1
            p.setPen(QPen(pen_c, 2 if active else 1))
            p.setBrush(fill_c)
            for poly in roi.polygons:
                qp = QPolygonF([QPointF(*self.canvas.to_scene(x, y)) for x, y in poly])
                p.drawPolygon(qp)
            for h in roi.hints:
                color = QColor(0, 255, 0) if h.positive else QColor(255, 0, 0)
                if not active:
                    color = QColor(color.red(), color.green(), color.blue(), 110)
                p.setPen(QPen(color, hint_w))
                p.setBrush(color)
                hx, hy = self.canvas.to_scene(h.x, h.y)
                p.drawEllipse(QPointF(hx, hy), 5 if active else 4, 5 if active else 4)

        pending = self.canvas.pending_polygon()
        if pending:
            p.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            qp = QPolygonF([QPointF(*self.canvas.to_scene(x, y)) for x, y in pending])
            p.drawPolyline(qp)
            for x, y in pending:
                sx, sy = self.canvas.to_scene(x, y)
                p.drawEllipse(QPointF(sx, sy), 3, 3)
        p.end()

    # ================================================================== ROI/提示点
    def _ensure_roi(self) -> RoiSet:
        lay = self._layer
        assert lay is not None and lay.analyzer is not None
        if lay.analyzer.roi is None:
            lay.analyzer.roi = RoiSet()
        return lay.analyzer.roi

    def _on_polygon_committed(self, points: list) -> None:
        if self._layer is None:
            return
        roi = self._ensure_roi()
        roi.add_polygon([(float(x), float(y)) for x, y in points])
        self._update_roi_label()
        self._invalidate_result(f"图层「{self._layer.name}」已加入一个框")
        self._refresh_overlay()
        self.lbl_info.setText("已加入一个框 —— 可继续圈，或点「▶ 预览」")

    def _on_canvas_clicked(self, x: float, y: float) -> None:
        mode = self.canvas.mode
        if self._layer is None:
            return
        roi = self._ensure_roi()
        if mode == "hint+":
            roi.hints.append(HintPoint(x, y, True))
        elif mode == "hint-":
            roi.hints.append(HintPoint(x, y, False))
        else:
            return
        self._update_roi_label()
        self._invalidate_result("已添加提示点")
        self._refresh_overlay()

    def _on_hint_remove(self, x: float, y: float) -> None:
        """右键：删除离点击位置最近的提示点（够不着就删最后加的那个）。"""
        roi = self.roi
        if not roi.hints:
            self.lbl_info.setText("没有可删除的提示点")
            return
        dists = [float(np.hypot(h.x - x, h.y - y)) for h in roi.hints]
        i = int(np.argmin(dists))
        idx = i if dists[i] <= 24.0 else len(roi.hints) - 1
        removed = roi.hints.pop(idx)
        n_pos = len(roi.positive_points())
        self.lbl_info.setText(
            f"已删除一个{'绿点' if removed.positive else '红点'}"
            f"（剩 绿{n_pos} 红{len(roi.hints) - n_pos}）"
        )
        self._update_roi_label()
        self._invalidate_result("已删除提示点")
        self._refresh_overlay()

    def _remove_last_roi(self) -> None:
        if self._layer is None:
            return
        self._ensure_roi().remove_last()
        self._update_roi_label()
        self._invalidate_result("已删除一个框")
        self._refresh_overlay()

    def _clear_roi(self) -> None:
        if self._layer is None:
            return
        self._ensure_roi().clear()
        self._update_roi_label()
        self._invalidate_result("已清空选区")
        self._refresh_overlay()

    def _clear_hints(self) -> None:
        if self._layer is None:
            return
        self._ensure_roi().hints.clear()
        self._update_roi_label()
        self._invalidate_result("已清除提示点")
        self._refresh_overlay()

    def _update_roi_label(self) -> None:
        roi = self.roi
        n_pos = len(roi.positive_points())
        n_neg = len(roi.hints) - n_pos
        name = self._layer.name if self._layer else "—"
        self.lbl_roi.setText(
            f"[{name}] " + (f"已圈定 {len(roi.polygons)} 个框" if roi.polygons else "尚未圈定")
        )
        self.lbl_hint.setText(f"绿点 {n_pos} 个，红点 {n_neg} 个")
        self._update_workflow_state()

    def _update_workflow_state(self) -> None:
        has_img = self.image is not None
        has_any = bool(self.stack is not None and any(l.visible and l.lines for l in self.stack.layers))
        self.btn_preview.setEnabled(has_img and not self._busy)
        self.layer_panel.btn_preview_all.setEnabled(has_img and not self._busy)
        self.btn_draft.setEnabled(has_img and has_any)
        self.btn_save.setEnabled(has_img and has_any)

    # ================================================================== 出图
    def _prepare_full_groups(self, use_edited: bool = False):
        """导出用的分组：优先用编辑窗口里人工修订过的线条。"""
        if self.stack is None:
            return []
        return self.stack.style_groups()

    def save_result(self) -> None:
        if self.image is None or self.stack is None:
            QMessageBox.information(self, "还没有结果", "请先点「▶ 预览」。")
            return
        groups = self._prepare_full_groups()
        if not groups:
            QMessageBox.information(self, "还没有结果", "请先点「▶ 预览」。")
            return
        from autoedge import settings as app_settings

        stem = Path(self.image.path or "result").stem
        # 起始目录 = 上次导出目录（没有则退回上次打开图片的目录）
        default = os.path.join(app_settings.initial_dir("export"), f"{stem}_out.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", default, "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif)"
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".png"
        try:
            mode = self.panel.output_mode()
            # 透明底只能存 PNG：选了 JPEG 会被合成为白底，等于白设，这里直接纠正
            if mode == "transparent" and os.path.splitext(path)[1].lower() in (".jpg", ".jpeg"):
                path = os.path.splitext(path)[0] + ".png"
                notify(self, f"透明底只能存 PNG，已改存\n{os.path.basename(path)}", 2600)
            bgr, alpha = render_groups(
                self.image.size,
                groups,
                base_bgr=self.image.bgr if mode == "overlay" else None,
                mode=mode,
                reference_color=self.panel.reference_color(),
                scale=1.0,
                supersample=SS_DEFAULT,   # 导出才做抗锯齿（4x 超采样）
            )
            save_image(path, bgr, alpha=alpha, expected_shape=(self.image.height, self.image.width))
        except Exception as exc:  # noqa: BLE001
            error(self, "保存失败", str(exc))
            return
        # 记导出目录：下次导出对话框从这里开始（跨会话保留）
        app_settings.remember_export(path)
        notify_success(self, "保存成功！")

    def make_draft(self) -> None:
        if self.image is None or self.stack is None:
            QMessageBox.information(self, "还没有结果", "请先点「▶ 预览」。")
            return
        if not any(l.lines for l in self.stack.layers):
            QMessageBox.information(self, "还没有结果", "请先点「▶ 预览」。")
            return
        if self.edit_window is not None:
            self.edit_window.close()
        self.edit_window = EditWindow(
            image=self.image,
            stack=self.stack,
            output_mode=self.panel.output_mode(),
            reference_color=self.panel.reference_color(),
            parent=None,
        )
        self.edit_window.show()
        self.edit_window.raise_()

    # ================================================================== 状态栏
    def _update_mode_label(self) -> None:
        """状态栏常驻显示当前输出模式（模式是全局的，这里随时可见）。"""
        mode = self.panel.output_mode()
        text = "透明底 + 线条" if mode == "transparent" else "原图 + 线条"
        self.lbl_mode.setText(f"输出：{text}")
        self.lbl_mode.setToolTip(
            "导出格式（在左侧「输出」里修改）。这是全局设置，会记住到下次启动。\n"
            "透明底只能存 PNG：若选了 JPEG 会自动改存 PNG。"
        )

    def _on_cursor(self, x: float, y: float) -> None:
        self.lbl_pos.setText(f"x={x:.0f}  y={y:.0f}")


def run() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    return int(app.exec())
