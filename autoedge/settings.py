"""界面小设置的持久化（最近文件、上次打开/导出目录、输出模式、导出 DPI）。

存放位置由 Qt 决定：``%APPDATA%\\AutoEdge\\AutoEdge.ini``（纯文本，可手改）。

**为什么用 QSettings 而不是自己写 JSON**：像 ``presets\\`` 那样把文件放在
可执行文件旁边，在 "exe 装在 Program Files" 这类只读目录下会写不进去；
QSettings 落在用户配置目录，永远可写，而且每次 setValue 立即落盘，
不需要在关窗口/退出时做"记得保存"的收尾逻辑。

与应用其它部分的唯一约定是 **必须让 QSettings 用 IniFormat**
（见 :func:`autoedge_gui.__main__.main`）：默认格式在 Windows 上是注册表，
那样这个文件就不存在了，出问题也不便直接查看。
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from PySide6.QtCore import QSettings

__all__ = [
    "APP_NAME",
    "DEFAULT_DPI",
    "DPI_MAX",
    "DPI_MIN",
    "MAX_RECENT",
    "clear_recent",
    "config_path",
    "dpi_linked",
    "export_dpi",
    "initial_dir",
    "last_export_dir",
    "last_open_dir",
    "output_mode",
    "recent_entries",
    "recent_files",
    "remember_file",
    "remember_export",
    "remove_recent",
    "set_dpi_linked",
    "set_export_dpi",
    "set_last_export_dir",
    "set_last_open_dir",
    "set_output_mode",
    "settings",
]

#: 应用名（决定 ini 文件名；与 ``QApplication.setApplicationName`` 保持一致）
APP_NAME = "AutoEdge"
#: 最近文件最多记这么多个
MAX_RECENT = 10
#: 输出模式（与 ``autoedge.config.OutputMode`` 一致；这里写成字面量以免 config 反向依赖本模块）
_MODES = ("overlay", "transparent")

#: 导出 DPI 的默认值（原图是 300，导出必须与之一致，否则下游按错误比例处理）
DEFAULT_DPI = 300.0
#: 导出 DPI 的合法范围（超出范围一律夹回来；非法值退回默认）
DPI_MIN = 1.0
DPI_MAX = 9999.0

_K_RECENT = "recent_files"
_K_OPEN_DIR = "last_open_dir"
_K_EXPORT_DIR = "last_export_dir"
_K_MODE = "output_mode"
_K_DPI_X = "export_dpi_x"
_K_DPI_Y = "export_dpi_y"
_K_DPI_LINK = "export_dpi_link"


def settings() -> QSettings:
    """取 :class:`QSettings` 实例（应用名未设置时兜底）。"""
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    if app is not None and not app.applicationName():
        app.setApplicationName(APP_NAME)
    return QSettings()


def config_path() -> str:
    """返回 ini 文件路径（仅用于显示/诊断）。"""
    return str(settings().fileName())


# ---------------------------------------------------------------------- 目录记忆
def last_open_dir() -> str:
    """上次成功打开图片所在目录（未记录返回空串）。"""
    return str(settings().value(_K_OPEN_DIR, "", type=str))


def set_last_open_dir(path: str) -> None:
    d = _dir_of(path)
    if d:
        settings().setValue(_K_OPEN_DIR, d)


def last_export_dir() -> str:
    """上次成功导出图片所在目录（未记录返回空串）。"""
    return str(settings().value(_K_EXPORT_DIR, "", type=str))


def set_last_export_dir(path: str) -> None:
    d = _dir_of(path)
    if d:
        settings().setValue(_K_EXPORT_DIR, d)


def _dir_of(path: str) -> str:
    """把"文件路径或目录路径"统一成目录；取不到返回空串。"""
    if not path:
        return ""
    p = Path(path)
    return str(p if p.is_dir() else p.parent)


def _reference_dir() -> str:
    """首次运行时的兜底目录：源码目录上一级的 ``参考图``（打包后不存在）。

    注意路径层级：源码运行时 ``preset_dir()`` = ``...\\AutoEdge\\AutoEdge\\presets``，
    参考图在 ``...\\AutoEdge\\参考图``，也就是再往上一级。
    """
    from .presets import preset_dir

    try:
        base = preset_dir()
    except OSError:  # pragma: no cover - 极端只读环境
        return ""
    for cand in (base.parents[1] / "参考图", base.parent / "参考图"):
        if cand.is_dir():
            return str(cand)
    return ""


def initial_dir(kind: str = "open") -> str:
    """文件对话框的起始目录。

    回退链：**上次用过的目录 → 参考图目录 → 用户主目录**。

    回退不是可选项：记住的目录可能被删掉、被改名，或位于已拔出的移动盘，
    那种情况下若把失效路径直接交给 Qt，对话框会退化成"当前工作目录"，
    使用者看到的就是一个莫名其妙的目录。这里每级都验存在性。

    Args:
        kind: ``"open"``（打开图片）或 ``"export"``（导出图片）。
    """
    order = (
        (last_open_dir(), _reference_dir())
        if kind == "open"
        else (last_export_dir(), last_open_dir(), _reference_dir())
    )
    for cand in order:
        if cand and os.path.isdir(cand):
            return cand
    return str(Path.home())


# ---------------------------------------------------------------------- 输出模式
def output_mode() -> str:
    """输出模式：``"overlay"``（原图+线条）或 ``"transparent"``（透明底+线条）。

    默认 **transparent**（使用者要求导出默认带透明底，避免忘记改而导出错）。
    """
    v = str(settings().value(_K_MODE, "transparent", type=str))
    return v if v in _MODES else "transparent"


def set_output_mode(mode: str) -> None:
    if mode in _MODES:
        settings().setValue(_K_MODE, mode)


# ---------------------------------------------------------------------- 导出 DPI
def _clamp_dpi(value: object, fallback: float = DEFAULT_DPI) -> float:
    """把任意读到的值收敛成合法 DPI；非法值返回 ``fallback``。

    ini 是可以手改的，手改出 ``300,5``（小数点写成逗号）或负数时不能崩，
    也不能真的把 0 dpi 写进文件——那种文件比没有 DPI 更麻烦。
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(f) or f <= 0:
        return fallback
    return min(DPI_MAX, max(DPI_MIN, f))


def export_dpi() -> tuple[float, float]:
    """导出图片的 ``(水平, 垂直)`` DPI，默认 **300 × 300**。

    只决定写出文件的元数据（PNG 写 ``pHYs``、JPEG 改写 JFIF density），
    **不参与任何像素计算**，所以导出尺寸仍严格等于原图。
    主窗口与编辑窗口共用这一份值。
    """
    s = settings()
    x = _clamp_dpi(s.value(_K_DPI_X, DEFAULT_DPI, type=float))
    y = _clamp_dpi(s.value(_K_DPI_Y, DEFAULT_DPI, type=float))
    return x, y


def set_export_dpi(x: float, y: float) -> None:
    """记住导出 DPI（立即落盘，跨会话生效）。"""
    s = settings()
    s.setValue(_K_DPI_X, round(_clamp_dpi(x), 4))
    s.setValue(_K_DPI_Y, round(_clamp_dpi(y), 4))


def dpi_linked() -> bool:
    """水平/垂直是否联动（**默认 True**，与界面复选框初值一致）。"""
    return bool(settings().value(_K_DPI_LINK, True, type=bool))


def set_dpi_linked(linked: bool) -> None:
    settings().setValue(_K_DPI_LINK, bool(linked))


# ---------------------------------------------------------------------- 最近文件
def recent_files() -> list[str]:
    """最近打开过的图片，**最新在前**；不去重、不检查存在性。

    用 ini 的**数组**形式（``recent_files\\1``、``recent_files\\2``…）而不是
    ``setValue(列表)``：后者会被 store 成逗号分隔的一行，而 Windows 路径
    本身可能含逗号（如 ``C:\\a,b\\c.png``），读回来就会被拆成两条。
    """
    s = settings()
    n = s.beginReadArray(_K_RECENT)
    out: list[str] = []
    try:
        for i in range(n):
            s.setArrayIndex(i)
            item = str(s.value("path", "", type=str))
            if item and item not in out:
                out.append(item)
    finally:
        s.endArray()
    return out[:MAX_RECENT]


def _write_recent(paths: list[str]) -> None:
    s = settings()
    s.beginWriteArray(_K_RECENT, len(paths[:MAX_RECENT]))
    try:
        for i, p in enumerate(paths[:MAX_RECENT]):
            s.setArrayIndex(i)
            s.setValue("path", p)
    finally:
        s.endArray()


def recent_entries() -> list[tuple[str, bool]]:
    """``[(路径, 文件是否还在)]``，最新在前，供菜单直接使用。"""
    return [(p, os.path.isfile(p)) for p in recent_files()]


def remember_file(path: str) -> None:
    """打开成功后调用：记目录 + 把文件插到最近列表头部（去重、限量）。"""
    if not path:
        return
    try:
        full = str(Path(path).resolve())
    except OSError:  # pragma: no cover - 极端路径
        full = str(path)
    rest = [p for p in recent_files() if p != full]
    _write_recent([full, *rest])
    set_last_open_dir(full)


def remember_export(path: str) -> None:
    """导出成功后调用：记导出目录。"""
    set_last_export_dir(path)


def remove_recent(path: str) -> None:
    """从最近列表里去掉一条（文件已被移动/删除时用）。"""
    rest = [p for p in recent_files() if p != path]
    _write_recent(rest)


def clear_recent() -> None:
    """清空最近文件列表（不影响目录记忆）。"""
    settings().remove(_K_RECENT)
