"""极简提示：只在屏幕中间浮一行字，1.5 秒后自动消失。

保存成功、导图完成这类"知道了就行"的反馈用它，不再弹带按钮的对话框。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QWidget

__all__ = ["notify", "notify_success", "error"]

#: 保持引用，否则窗口会立刻被回收
_ALIVE: list[QWidget] = []


def notify(parent: QWidget | None, text: str, msec: int = 1500) -> None:
    """在屏幕中间浮一行字，``msec`` 毫秒后自动消失。"""
    app = QApplication.instance()
    if app is None:
        return
    lab = QLabel(f"✓  {text}")
    lab.setAlignment(Qt.AlignCenter)
    lab.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
    lab.setAttribute(Qt.WA_ShowWithoutActivating, True)
    lab.setStyleSheet(
        "QLabel { background-color: rgba(28,28,28,235); color: #eaeaea;"
        " border: 1px solid rgba(255,255,255,60); border-radius: 8px;"
        " padding: 14px 26px; font-size: 15px; }"
    )
    lab.adjustSize()
    screen = parent.screen() if parent is not None else app.primaryScreen()
    if screen is not None:
        geo = screen.availableGeometry()
    else:  # 兜底
        geo = app.primaryScreen().availableGeometry()
    lab.move(geo.center() - lab.rect().center())
    lab.show()
    _ALIVE.append(lab)
    QTimer.singleShot(msec, lambda: _close(lab))


def _close(lab: QWidget) -> None:
    try:
        lab.close()
    finally:
        if lab in _ALIVE:
            _ALIVE.remove(lab)


def notify_success(parent: QWidget | None, text: str = "保存成功！") -> None:
    """保存/导出成功的统一提示。"""
    notify(parent, text)


def error(parent: QWidget | None, title: str, text: str) -> None:
    """出错提示——这个需要使用者读内容，保留对话框。"""
    QMessageBox.critical(parent, title, text)
