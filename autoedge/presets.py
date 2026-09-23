"""通用预设：保存/载入**参数 + 线条样式**。

按需求，预设只包含这两块，不含选区/提示点/图层结构/输出模式：

* :class:`~autoedge.config.SegConfig` —— 描边阈值、最小面积、简化容差等参数；
* :class:`~autoedge.config.LineStyle` —— 线型、颜色、线宽、不透明度。

所有图层共用同一套设置，所以预设也是**全局通用**的：载入后应用到所有图层。

存放位置：可执行文件（打包后）或项目根目录（源码运行时）下的 ``presets\\<名字>.json``。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import LineStyle, SegConfig

__all__ = ["preset_dir", "PresetData", "list_presets", "save_preset", "load_preset", "delete_preset"]

_BAD_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def preset_dir() -> Path:
    """预设目录（不存在则创建）。"""
    if getattr(sys, "frozen", False):  # PyInstaller 打包后：exe 同目录
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parents[1]
    d = base / "presets"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path.home() / ".autoedge" / "presets"
        d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(name: str) -> str:
    """把预设名清洗成合法文件名。"""
    n = _BAD_CHARS.sub("_", (name or "").strip())
    return n[:64] or "未命名"


@dataclass
class PresetData:
    """一条预设：只有参数与线条样式。"""

    name: str = ""
    seg: SegConfig | None = None
    style: LineStyle | None = None

    def __post_init__(self) -> None:
        if self.seg is None:
            self.seg = SegConfig()
        if self.style is None:
            self.style = LineStyle()

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "seg": self.seg.to_dict(),
            "style": self.style.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "PresetData":
        return PresetData(
            name=str(d.get("name", "")),
            seg=SegConfig.from_dict(d.get("seg", {})),
            style=LineStyle.from_dict(d.get("style", {})),
        )


def list_presets() -> list[str]:
    """已保存的预设名（按名称排序）。"""
    d = preset_dir()
    return sorted(p.stem for p in d.glob("*.json"))


def save_preset(name: str, seg: SegConfig, style: LineStyle) -> Path:
    """保存预设，返回文件路径。"""
    data = PresetData(name=name, seg=seg.normalized(), style=style.normalized())
    path = preset_dir() / f"{safe_name(name)}.json"
    path.write_text(json.dumps(data.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_preset(name: str) -> PresetData | None:
    """载入预设；不存在或损坏返回 ``None``。"""
    path = preset_dir() / f"{safe_name(name)}.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return PresetData.from_dict(d)


def delete_preset(name: str) -> bool:
    """删除预设。"""
    path = preset_dir() / f"{safe_name(name)}.json"
    try:
        path.unlink()
        return True
    except OSError:
        return False
