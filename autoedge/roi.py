"""感兴趣区域（ROI）与人工提示点。

使用者先在图上粗略圈出要描边的晶体（可圈多个），算法**只在框内**分析，
这样基底上的小碎晶、四角亮弧、灰尘都不会被当成晶体。

* 多个框重叠时按"合并为一个连通的大框"处理（直接取并集）。
* 提示点：绿点 = 这里一定是晶体（强制纳入），红点 = 这里一定不是晶体（强制剔除）。
* 框的边界本身**不画线**（与"画面边缘不闭合"同理），线也不延伸出框外。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["HintPoint", "RoiSet"]


@dataclass
class HintPoint:
    """一个人工提示点。"""

    x: float
    y: float
    #: ``True`` = 是晶体；``False`` = 不是晶体
    positive: bool = True


@dataclass
class RoiSet:
    """一组 ROI 多边形 + 提示点。"""

    #: 每个元素是 ``(N, 2)`` 的多边形顶点（原图坐标）
    polygons: list[np.ndarray] = field(default_factory=list)
    hints: list[HintPoint] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """既没有框也没有提示点时才算空。

        注意：提示点也算内容 —— 若只看 ``polygons``，使用者"先点绿点、还没圈框"
        时会被判为空，随后的 set_roi(None) 会把刚点的点丢掉。
        """
        return not self.polygons and not self.hints

    def add_polygon(self, points: list[tuple[float, float]]) -> None:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] >= 3:
            self.polygons.append(arr)

    def remove_last(self) -> None:
        if self.polygons:
            self.polygons.pop()

    def clear(self) -> None:
        self.polygons.clear()
        self.hints.clear()

    # ------------------------------------------------------------------ 栅格化
    def mask(self, shape: tuple[int, int], pad: int = 0) -> np.ndarray:
        """把多边形并集栅格化成 ``uint8`` 掩码（1 = 框内）。

        Args:
            shape: ``(H, W)``。
            pad: 向外扩张的像素数（正数扩、负数缩）。
        """
        h, w = shape
        m = np.zeros((h, w), dtype=np.uint8)
        if not self.polygons:
            return m
        polys = [np.round(p).astype(np.int32).reshape(-1, 1, 2) for p in self.polygons]
        cv2.fillPoly(m, polys, 1)
        if pad > 0:
            k = 2 * int(pad) + 1
            m = cv2.dilate(m, np.ones((k, k), np.uint8))
        elif pad < 0:
            r = int(-pad)
            k = 2 * r + 1
            m = cv2.erode(m, np.ones((k, k), np.uint8))
        return m

    def bbox(self, shape: tuple[int, int], pad: int = 8) -> tuple[int, int, int, int] | None:
        """所有多边形的联合包围盒（外扩 ``pad`` 像素），``(x0, y0, x1, y1)`` 右开区间。"""
        if not self.polygons:
            return None
        pts = np.vstack(self.polygons)
        h, w = shape
        x0 = int(max(0, np.floor(pts[:, 0].min()) - pad))
        y0 = int(max(0, np.floor(pts[:, 1].min()) - pad))
        x1 = int(min(w, np.ceil(pts[:, 0].max()) + pad + 1))
        y1 = int(min(h, np.ceil(pts[:, 1].max()) + pad + 1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def positive_points(self) -> list[tuple[float, float]]:
        return [(p.x, p.y) for p in self.hints if p.positive]

    def negative_points(self) -> list[tuple[float, float]]:
        return [(p.x, p.y) for p in self.hints if not p.positive]

    def same_as(self, other: "RoiSet | None") -> bool:
        """与另一组 ROI 是否等价（用于避免"重设同样的 ROI 却清空缓存"）。

        多图层下这一点很关键：切到另一个图层预览时，绝不能因为
        "重新 set_roi" 就把**别的图层**已经算好的结果清掉。
        """
        if other is None:
            return False
        if len(self.polygons) != len(other.polygons) or len(self.hints) != len(other.hints):
            return False
        for a, b in zip(self.polygons, other.polygons, strict=True):
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-3):
                return False
        for a, b in zip(self.hints, other.hints, strict=True):
            if a.positive != b.positive or abs(a.x - b.x) > 1e-3 or abs(a.y - b.y) > 1e-3:
                return False
        return True

    # ------------------------------------------------------------------ 持久化
    def to_dict(self) -> dict:
        return {
            "polygons": [p.tolist() for p in self.polygons],
            "hints": [{"x": p.x, "y": p.y, "positive": p.positive} for p in self.hints],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "RoiSet":
        rs = RoiSet()
        for poly in d.get("polygons", []):
            rs.add_polygon([(float(x), float(y)) for x, y in poly])
        for hint in d.get("hints", []):
            rs.hints.append(
                HintPoint(float(hint["x"]), float(hint["y"]), bool(hint.get("positive", True)))
            )
        return rs
