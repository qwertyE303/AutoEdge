"""图层管理面板（右侧，PS 风格）。

每个图层拥有自己的一套：选区、提示点、描边阈值、线条样式。
最终显示与输出是"所有可见图层的叠加"，因此可以先用高阈值描明显轮廓，
再新建一个图层用低阈值去描单层区。

本面板只负责"显示图层列表 + 发出用户意图信号"，不改动分析逻辑。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["LayerPanel"]


def _swatch(color: tuple[int, int, int], w: int = 14, h: int = 14) -> QPixmap:
    pm = QPixmap(w, h)
    pm.fill(QColor(*color))
    return pm


class LayerPanel(QWidget):
    """右侧图层列表。

    信号：
    * :attr:`current_changed` —— 当前图层改变（参数面板需同步）
    * :attr:`visibility_changed` —— 某图层显示/隐藏改变
    * :attr:`request_add` / :attr:`request_duplicate` / :attr:`request_remove`
    * :attr:`request_move` —— 上移(-1)/下移(+1)
    """

    current_changed = Signal(int)
    visibility_changed = Signal(int, bool)
    request_add = Signal()
    request_duplicate = Signal()
    request_remove = Signal()
    request_move = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("图层（当前图层的选区/提示点/阈值独立）")
        title.setWordWrap(True)
        layout.addWidget(title)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setToolTip("点选切换当前图层；勾选框控制是否参与叠加显示与输出")
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list, 1)

        row1 = QHBoxLayout()
        self.btn_add = QPushButton("新建")
        self.btn_dup = QPushButton("复制")
        self.btn_del = QPushButton("删除")
        for b, tip in (
            (self.btn_add, "新建一个空图层（可换选区/换阈值）"),
            (self.btn_dup, "复制当前图层（含选区、提示点与阈值）"),
            (self.btn_del, "删除当前图层"),
        ):
            b.setToolTip(tip)
            row1.addWidget(b)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_up = QPushButton("上移")
        self.btn_down = QPushButton("下移")
        self.btn_preview_all = QPushButton("预览全部图层")
        self.btn_preview_all.setToolTip("依次重新分析所有可见图层（每层 1~10 秒，会花些时间）")
        for b in (self.btn_up, self.btn_down):
            row2.addWidget(b)
        layout.addLayout(row2)
        layout.addWidget(self.btn_preview_all)

        self.btn_add.clicked.connect(self.request_add)
        self.btn_dup.clicked.connect(self.request_duplicate)
        self.btn_del.clicked.connect(self.request_remove)
        self.btn_up.clicked.connect(lambda: self.request_move.emit(-1))
        self.btn_down.clicked.connect(lambda: self.request_move.emit(+1))

    # ------------------------------------------------------------------ 刷新
    def rebuild(self, layers, current: int) -> None:
        """按图层列表重建显示。

        Args:
            layers: :class:`autoedge.pipeline.Layer` 列表。
            current: 当前图层序号。
        """
        self._updating = True
        self.list.clear()
        for i, lay in enumerate(layers):
            tags = []
            if lay.stale:
                tags.append("待预览")
            n = len(lay.lines)
            if n:
                tags.append(f"{n} 条线")
            ana = lay.analyzer
            if ana is not None and ana.roi is not None and not ana.roi.empty:
                tags.append(f"框{len(ana.roi.polygons)}")
                nh = len(ana.roi.hints)
                if nh:
                    tags.append(f"点{nh}")
            text = f"{lay.name}" + (f"　[{' / '.join(tags)}]" if tags else "")
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if lay.visible else Qt.Unchecked)
            item.setIcon(_swatch(lay.style.color))
            item.setData(Qt.UserRole, i)
            self.list.addItem(item)
        if 0 <= current < self.list.count():
            self.list.setCurrentRow(current)
        self._updating = False
        self.btn_del.setEnabled(len(layers) > 1)

    # ------------------------------------------------------------------ 事件
    def _on_row_changed(self, row: int) -> None:
        if self._updating or row < 0:
            return
        self.current_changed.emit(row)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        idx = item.data(Qt.UserRole)
        self.visibility_changed.emit(int(idx), item.checkState() == Qt.Checked)
