"""GUI 启动入口。

用法::

    python -m autoedge_gui            # 从源码目录启动
    AutoEdge.exe                      # 打包后的可执行文件
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    # 让 PyInstaller 单文件/单目录模式下都能找到包
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # noqa: BLE001
        print(f"无法加载 PySide6: {exc}", file=sys.stderr)
        return 2

    from autoedge_gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("AutoEdge")
    app.setOrganizationName("AutoEdge")
    # 记住的设置（最近文件 / 上次打开与导出目录 / 输出模式）写成 ini 文件而不是注册表：
    # ``%APPDATA%\AutoEdge\AutoEdge.ini``，纯文本、可手改、出问题好排查。
    # 必须在**任何 QSettings 被构造之前**设置。
    QSettings.setDefaultFormat(QSettings.IniFormat)
    win = MainWindow()
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
