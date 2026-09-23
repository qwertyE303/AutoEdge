"""折线几何工具。

**关键约束（使用者明确要求）**：绝不做曲线拟合。晶体的长直边必须保持为严格的直线段，
因为使用者要靠长边的倾角来确定晶体取向。因此这里只提供"去除像素阶梯 + Douglas-Peucker
简化"这类不改变几何的清理手段，不提供任何样条平滑与去抖滤波。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Line",
    "simplify_polyline",
    "remove_staircase",
    "merge_thin_excursions",
    "polyline_length",
    "orientation_deg",
    "is_closed",
    "dedupe_points",
]


@dataclass
class Line:
    """一条折线。

    Attributes:
        points: ``(N, 2)`` float32，按 ``(x, y)`` 排列，图像像素坐标。
        kind: ``"outer"``（外轮廓）或 ``"inner"``（内部分界线）。
        crystal_id: 所属晶体 id。
        attrs: 附加信息（长度、倾角等）。
    """

    points: np.ndarray
    kind: str = "outer"
    crystal_id: int = 0
    #: 所属图层序号（多图层叠加时用；单图层为 0）
    layer_id: int = 0
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    @property
    def length(self) -> float:
        return polyline_length(self.points)

    @property
    def closed(self) -> bool:
        return bool(self.attrs.get("closed", False))

    def bbox(self) -> tuple[float, float, float, float]:
        x, y = self.points[:, 0], self.points[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    def longest_segment(self) -> tuple[float, float]:
        """返回最长直线段的 ``(长度, 倾角-度)``。倾角范围 ``(-90, 90]``。"""
        if self.n < 2:
            return 0.0, 0.0
        p = self.points
        if self.closed:
            p = np.vstack([p, p[:1]])
        d = np.diff(p, axis=0)
        seg_len = np.hypot(d[:, 0], d[:, 1])
        if seg_len.size == 0:
            return 0.0, 0.0
        i = int(np.argmax(seg_len))
        return float(seg_len[i]), orientation_deg(d[i])


def dedupe_points(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """去掉连续重复点。"""
    if points.shape[0] < 2:
        return points
    keep = [0]
    for i in range(1, points.shape[0]):
        if np.hypot(*(points[i] - points[keep[-1]])) > tol:
            keep.append(i)
    return points[keep]


def remove_staircase(points: np.ndarray, max_step: float = 1.5, max_chord: float = 3.5) -> np.ndarray:
    """去除像素栅格化产生的"直角台阶"点，**不改变真实几何**。

    判定条件（三者同时满足才删除，因此真实拐角与真实台阶一定被保留）：

    1. 该点相对前后邻点的**转角**接近 90°（典型台阶形状）；
    2. 前后邻点的**连线长度**不超过 ``max_chord``（保证只跨越一个台阶，
       不会级联塌陷成长直线——这一点至关重要）；
    3. 该点到前后邻点连线的垂距不超过 ``max_step``。

    Args:
        points: ``(N, 2)`` 折线。
        max_step: 允许删除造成的最大几何偏差（像素）。
        max_chord: 允许跨越的最大弦长（像素），防止级联删除。

    Returns:
        清理后的折线。
    """
    pts = dedupe_points(np.asarray(points, dtype=np.float64))
    if pts.shape[0] < 3:
        return pts.astype(np.float32)

    n = pts.shape[0]
    alive = np.ones(n, dtype=bool)
    changed = True
    while changed:
        changed = False
        idx = np.nonzero(alive)[0]
        if idx.size < 3:
            break
        prev = idx[:-2]
        cur = idx[1:-1]
        nxt = idx[2:]
        a = pts[prev]
        b = pts[cur]
        c = pts[nxt]

        v1 = b - a
        v2 = c - b
        chord = c - a
        chord_len = np.hypot(chord[:, 0], chord[:, 1])
        cross = np.abs(chord[:, 0] * (b[:, 1] - a[:, 1]) - chord[:, 1] * (b[:, 0] - a[:, 0]))
        with np.errstate(invalid="ignore", divide="ignore"):
            perp = np.where(chord_len > 1e-9, cross / np.maximum(chord_len, 1e-9), 0.0)
        # 转角接近 90°：两个方向的点积相对各自长度都很小
        len1 = np.hypot(v1[:, 0], v1[:, 1])
        len2 = np.hypot(v2[:, 0], v2[:, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_sim = np.where(
                (len1 > 1e-9) & (len2 > 1e-9),
                (v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]) / np.maximum(len1 * len2, 1e-9),
                1.0,
            )
        drop = (
            (np.abs(cos_sim) < 0.45)
            & (chord_len <= float(max_chord))
            & (perp <= float(max_step))
        )
        if drop.any():
            alive[cur[drop]] = False
            changed = True

    out = pts[alive]
    if out.shape[0] < 3:
        return pts.astype(np.float32)
    return out.astype(np.float32)


def merge_thin_excursions(
    points: np.ndarray, max_span: float = 2.5, max_dev: float = 6.0
) -> np.ndarray:
    """合并"出去又马上回来"的薄片顶点（治长竖直弦的关键一步）。

    像素栅格化会在掩码边缘留下 1~2px 宽的毛刺。`findContours` 会忠实地沿毛刺
    走上去再走回来，于是轮廓上出现两条相距 1~2px、方向相反的平行长边；
    而这两条边的**端点位置几乎相同**——Douglas-Peucker 会把它们合并成一条长直线，
    结果就是一条**横穿晶体内部的长弦**（实测出现过 821px 的纯竖直段）。

    这里在简化**之前**先处理：若某个顶点相对前后邻点的位移很小
    （``|P[i-1] - P[i+1]| <= max_span``），说明这一段是"出去又回来"的薄片，
    把 P[i] 移到两侧邻点的中点上并删除该顶点。判定只看相邻两点，是纯局部操作，
    因此不会像全局 DP 那样把两条长平行边误合并。

    **防塌陷设计**（旧版实测会把点挪离原位最多 35px，进而被 DP 塌成一条长直线；
    WS2/0011 的 173,896px 掩码曾被压成 3 个点、7px 的三角形）：

    1. 同一轮内**相邻**的顶点不一起删——台阶毛刺是成对出现的，
       允许相邻同删会让"删点→邻居成新邻→再删"一路连锁下去；
    2. 每轮算完先试算，若合并结果偏离原始轮廓超过 ``max_dev``，
       就把本轮删除量减半重试，还超就整轮放弃——**绝不放任点被挪远**；
    3. 收敛后仍有独立复核，超限则整体回退到原始轮廓。

    Args:
        points: ``(N, 2)`` 折线。
        max_span: 允许合并的最大"往返跨度"（像素）。
        max_dev: 允许合并造成的最大几何偏离（像素）。

    Returns:
        清理后的折线。
    """
    pts_in = dedupe_points(np.asarray(points, dtype=np.float64))
    if pts_in.shape[0] < 5:
        return pts_in.astype(np.float32)
    closed = bool(np.allclose(pts_in[0], pts_in[-1]))
    pts = pts_in[:-1].copy() if closed else pts_in.copy()
    n = pts.shape[0]
    if n < 4:
        return pts_in.astype(np.float32)

    alive = np.ones(n, dtype=bool)
    span2 = float(max_span) ** 2
    limit = float(max_dev)

    def _trial(sel: np.ndarray, prev: np.ndarray, nxt: np.ndarray, idx: np.ndarray) -> bool:
        """试算本轮删除后的偏离是否可接受（不改动原数组）。"""
        if sel.size == 0:
            return False
        probe_pts = pts.copy()
        probe_pts[idx[sel]] = 0.5 * (pts[prev[sel]] + pts[nxt[sel]])
        probe_alive = alive.copy()
        probe_alive[idx[sel]] = False
        keep = dedupe_points(probe_pts[probe_alive])
        if keep.shape[0] < 3:
            return False
        probe = np.vstack([keep, keep[:1]]) if closed else keep
        return _polyline_max_deviation(pts_in, probe) <= limit

    for _ in range(20):
        idx = np.nonzero(alive)[0]
        if idx.size < 4:
            break
        m = idx.size
        prev = idx[(np.arange(m) - 1) % m]
        nxt = idx[(np.arange(m) + 1) % m]
        d = pts[nxt] - pts[prev]
        span = d[:, 0] ** 2 + d[:, 1] ** 2
        # 闸门 1：同一轮内不相邻才可删（沿环相邻的两个索引互斥）
        chosen: list[int] = []
        blocked: set[int] = set()
        for k in np.nonzero(span <= span2)[0]:
            ki = int(k)
            if ki in blocked:
                continue
            chosen.append(ki)
            blocked.add((ki + 1) % m)
            blocked.add((ki - 1) % m)
        if not chosen:
            break
        # 闸门 2：删除量过大就减半重试，绝不把点挪远
        sel = np.asarray(chosen, dtype=np.int64)
        while sel.size and not _trial(sel, prev, nxt, idx):
            sel = sel[: max(1, sel.size // 2)]
            if sel.size == 1 and not _trial(sel, prev, nxt, idx):
                sel = sel[:0]
        if sel.size == 0:
            break
        pts[idx[sel]] = 0.5 * (pts[prev[sel]] + pts[nxt[sel]])
        alive[idx[sel]] = False

    out = dedupe_points(pts[alive])
    if out.shape[0] < 3:
        return pts_in.astype(np.float32)

    # 闸门 3：收敛后独立复核
    probe = np.vstack([out, out[:1]]) if closed else out
    if _polyline_max_deviation(pts_in, probe) > limit + 1e-6:
        return pts_in.astype(np.float32)

    if closed and out.shape[0] >= 3:
        out = np.vstack([out, out[:1]])
    return out.astype(np.float32)


def _polyline_max_deviation(src: np.ndarray, ref: np.ndarray) -> float:
    """``src`` 上各点到 ``ref`` 折线的最大距离（像素）。"""
    if src.shape[0] == 0 or ref.shape[0] < 2:
        return float("inf")
    a = np.asarray(ref, dtype=np.float64)[:-1]
    b = np.asarray(ref, dtype=np.float64)[1:]
    ab = b - a
    seg2 = (ab * ab).sum(axis=1)
    seg2[seg2 <= 1e-12] = 1e-12
    o = np.asarray(src, dtype=np.float64)
    best = np.full(o.shape[0], np.inf, dtype=np.float64)
    for i in range(a.shape[0]):
        ap = o - a[i]
        t = np.clip((ap @ ab[i]) / seg2[i], 0.0, 1.0)
        proj = a[i] + t[:, None] * ab[i]
        np.minimum(best, np.hypot(o[:, 0] - proj[:, 0], o[:, 1] - proj[:, 1]), out=best)
    return float(best.max())


def simplify_polyline(
    points: np.ndarray, tolerance: float, closed: bool = False, thin_span: float = 2.5
) -> np.ndarray:
    """薄片合并 + 阶梯去除 + Douglas-Peucker 简化。

    与曲线拟合完全不同：简化后的顶点**一定落在原始折线上**（或薄片合并后的中点），
    长直线会被压缩成只有两个端点的严格直线段，倾角保持不变。**不做任何平滑**。

    Args:
        points: ``(N, 2)`` 折线。
        tolerance: 简化容差（像素）。0 表示只做阶梯去除。
        closed: 是否为闭合折线。为 True 时**返回的点列显式回到起点**
            （即 ``result[0] == result[-1]``），这样 ``polyline_length(closed=True)``
            、渲染、以及界面上的编辑操作都能用同一份数据，不必各自再补一次。
        thin_span: 薄片合并阈值（像素）；<=0 表示不做。

    Returns:
        简化后的折线。
    """
    pts = dedupe_points(np.asarray(points, dtype=np.float64))
    if pts.shape[0] < 3:
        return pts.astype(np.float32)
    if thin_span > 0:
        pts = merge_thin_excursions(
            pts, thin_span, max_dev=2.0 * float(thin_span) + 2.0 * float(tolerance)
        ).astype(np.float64)
        if pts.shape[0] < 3:
            return pts.astype(np.float32)
    pts = remove_staircase(pts, max_step=max(0.5, min(1.5, tolerance + 0.5)))

    if tolerance <= 0:
        return _close_if_needed(pts, closed)

    if closed:
        # 闭合折线用 contour 版；必须先去掉重复端点再做阶梯去除，
        # 否则 remove_staircase 会因首尾重复而失去正确的邻域关系。
        pts = pts[:-1] if np.allclose(pts[0], pts[-1]) else pts
        if pts.shape[0] < 4:
            return _close_if_needed(pts, closed)
        cleaned = remove_staircase(pts, max_step=max(0.5, min(1.5, tolerance + 0.5)))
        arr = np.vstack([cleaned, cleaned[:1]])
        out = _approx_polygon(arr, tolerance)
        if out.shape[0] >= 2 and np.allclose(out[0], out[-1]):
            out = out[:-1]
        if out.shape[0] < 3:
            return _close_if_needed(cleaned, closed)
        # 补成显式闭合（见 docstring）
        return _close_if_needed(out, closed)

    app = _approx_chain(pts, tolerance)
    return app.astype(np.float32)


def _close_if_needed(pts: np.ndarray, closed: bool) -> np.ndarray:
    """闭合折线显式补上"回到起点"，保证 ``pts[0] == pts[-1]``。"""
    pts = np.asarray(pts, dtype=np.float64)
    if not closed or pts.shape[0] < 3:
        return pts.astype(np.float32)
    if np.allclose(pts[0], pts[-1]):
        return pts.astype(np.float32)
    return np.vstack([pts, pts[:1]]).astype(np.float32)


def _approx_chain(pts: np.ndarray, tolerance: float) -> np.ndarray:
    """对开曲线做迭代式 Douglas-Peucker（栈实现，避免深递归）。"""
    n = pts.shape[0]
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    tol2 = float(tolerance) ** 2
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a = pts[i0]
        b = pts[i1]
        ab = b - a
        ab_len2 = float(ab[0] ** 2 + ab[1] ** 2)
        seg = pts[i0 + 1: i1]
        if ab_len2 < 1e-12:
            d2 = ((seg - a) ** 2).sum(axis=1)
        else:
            t = ((seg - a) @ ab) / ab_len2
            proj = a + t[:, None] * ab
            d2 = ((seg - proj) ** 2).sum(axis=1)
        if d2.size == 0:
            continue
        k = int(np.argmax(d2))
        if d2[k] > tol2:
            mid = i0 + 1 + k
            keep[mid] = True
            stack.append((i0, mid))
            stack.append((mid, i1))
    return pts[keep]


def _approx_polygon(pts: np.ndarray, tolerance: float) -> np.ndarray:
    """闭合折线的简化（OpenCV approxPolyDP）。"""
    import cv2

    arr = np.ascontiguousarray(pts.reshape(-1, 1, 2).astype(np.float32))
    out = cv2.approxPolyDP(arr, float(tolerance), True)
    return out.reshape(-1, 2).astype(np.float64)


def polyline_length(points: np.ndarray, closed: bool = False) -> float:
    """折线总长度（像素）。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    d = np.diff(pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def orientation_deg(vec: np.ndarray) -> float:
    """向量倾角，单位为度，范围 ``(-90, 90]``（把方向取反视为同一条边）。"""
    dx, dy = float(vec[0]), float(vec[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    ang = float(np.degrees(np.arctan2(dy, dx)))
    if ang <= -90.0:
        ang += 180.0
    elif ang > 90.0:
        ang -= 180.0
    return ang


def is_closed(points: np.ndarray, tol: float = 2.0) -> bool:
    """首尾点是否重合（闭合折线）。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        return False
    return bool(np.hypot(*(pts[0] - pts[-1])) <= tol)
