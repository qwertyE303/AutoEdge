"""晶体对象与外轮廓提取、整图分析。

外轮廓来自晶体掩码的连通域边界，经过"阶梯去除 + Douglas-Peucker 简化"，
顶点始终落在真实边界上，**长直边保持为严格直线**。

关于画面边缘：使用者要求"超出画面的部分不管它，不要沿着画面边缘闭合"，
因此当掩码触到画面边界时（``open_at_border=True``，默认），
沿画面边界的那一段会被裁掉，只保留晶体在画面内的真实边缘（开折线）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import SegConfig
from .geometry import (
    Line,
    dedupe_points,
    is_closed,
    orientation_deg,
    polyline_length,
    simplify_polyline,
)

__all__ = ["Crystal", "extract_outer_lines", "assign_crystal_ids", "build_crystals", "analyze_outer"]


@dataclass
class Crystal:
    """一块晶体。

    Attributes:
        crystal_id: 连通域 id（从 1 开始）。
        area: 像素面积。
        bbox: ``(x, y, w, h)`` 包围盒。
        lines: 该晶体的外轮廓折线（可能不止一条）。
        median_color: 晶体内部的 Lab 中位色，便于报告与调试。
    """

    crystal_id: int
    area: int
    bbox: tuple[int, int, int, int]
    lines: list[Line] = field(default_factory=list)
    median_color: np.ndarray | None = None


def _mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    """取二值掩码的**外**轮廓（忽略孔洞），返回 ``(N, 2)`` float32 列表。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: list[np.ndarray] = []
    for c in contours:
        if c.shape[0] < 3:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        out.append(pts)
    return out


def _component_slices(labels: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    """一次 ``connectedComponentsWithStats`` 拿到每个连通域的包围盒。

    **这是这里唯一可靠的取域方式**：以前用"从该连通域最上面那个像素起切片 + floodFill"
    的做法，一旦那个像素不在最左端（例如晶体最高点落在右半瓣上），切片就会把左侧
    整片区域切掉，轮廓于是沿着切边直上直下闭合成一条**穿过晶体的竖线**。
    实测 h-BN/0016 会因此丢掉 1,097,825 px（占该晶体 56%）。

    Returns:
        ``{连通域 id: (x0, y0, w, h)}``，全是左闭右开的绝对（整图）坐标。
    """
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(labels.astype(np.uint8), connectivity=8)
    out: dict[int, tuple[int, int, int, int]] = {}
    for i in range(1, int(n)):
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0 or ww <= 0 or hh <= 0:
            continue
        out[i] = (x0, y0, ww, hh)
    return out


def _is_closed_pts(pts: np.ndarray, tol: float = 1e-6) -> bool:
    """首尾点数值上是否重合（闭合轮廓）。"""
    return bool(pts.shape[0] >= 2 and np.hypot(*(pts[0] - pts[-1])) <= tol)


def _max_deviation(orig: np.ndarray, simp: np.ndarray) -> float:
    """``orig`` 上每个点到 ``simp`` 折线的最大距离（像素）。

    用来兜底校验简化结果有没有把轮廓"塌掉"——Douglas-Peucker 本身保证偏差不超过
    容差，但前面的薄片合并可能把点挪走，所以这里独立复核一次。
    """
    if orig.shape[0] == 0 or simp.shape[0] < 2:
        return float("inf")
    a = simp[:-1].astype(np.float64)
    b = simp[1:].astype(np.float64)
    ab = b - a
    seg2 = (ab * ab).sum(axis=1)
    seg2[seg2 <= 1e-12] = 1e-12
    best = np.full(orig.shape[0], np.inf, dtype=np.float64)
    o = orig.astype(np.float64)
    for i in range(a.shape[0]):
        ap = o - a[i]
        t = np.clip((ap @ ab[i]) / seg2[i], 0.0, 1.0)
        proj = a[i] + t[:, None] * ab[i]
        d = np.hypot(o[:, 0] - proj[:, 0], o[:, 1] - proj[:, 1])
        np.minimum(best, d, out=best)
    return float(best.max())


def _split_border_span(pts: np.ndarray, w: int, h: int, margin: int = 0) -> tuple[np.ndarray, bool]:
    """若轮廓沿画面边界走，则裁掉边界段，返回开折线与"是否被裁剪"标记。

    Args:
        pts: ``(N, 2)`` 闭合轮廓点。
        w: 图像宽度。
        h: 图像高度。
        margin: 判定为"贴边"的距离容差（像素）。

    Returns:
        ``(新的折线, 是否发生裁剪)``。若整条轮廓都贴边（例如细长晶体横穿整幅图），
        则原样返回并标记未裁剪，避免产出空结果。
    """
    on_border = (
        (pts[:, 0] <= margin)
        | (pts[:, 0] >= w - 1 - margin)
        | (pts[:, 1] <= margin)
        | (pts[:, 1] >= h - 1 - margin)
    )
    if not on_border.any():
        return pts, False
    if on_border.all():
        return pts, False

    # 找到第一个"不贴边"的点作为新起点，顺序重排，再用贴边点做切分
    idx_inside = np.nonzero(~on_border)[0]
    start = int(idx_inside[0])
    rolled = np.roll(pts, -start, axis=0)
    rolled_border = np.roll(on_border, -start)
    # 找出若干段连续的"贴边"区间，把内部段提取出来
    segments: list[np.ndarray] = []
    current: list[np.ndarray] = []
    n = rolled.shape[0]
    for i in range(n):
        if rolled_border[i]:
            if len(current) >= 2:
                segments.append(np.array(current, dtype=np.float32))
            current = []
        else:
            current.append(rolled[i])
    if len(current) >= 2:
        segments.append(np.array(current, dtype=np.float32))
    if not segments:
        return pts, False
    # 取最长的一段（闭合轮廓被画面切开时最多产生一段有效边缘）
    best = max(segments, key=lambda s: polyline_length(s))
    return best, True


def extract_outer_lines(
    labels: np.ndarray,
    cfg: SegConfig,
    *,
    open_at_border: bool = True,
) -> list[Line]:
    """从晶体标记图提取所有外轮廓折线。

    取域方式：``connectedComponentsWithStats`` 给出每个连通域的**精确包围盒**，
    只在该包围盒内做 ``findContours``。这样既保住了"不在全分辨率图上反复找轮廓"
    的性能优化，又不会像"从首个像素起 floodFill"那样把晶体切掉一半。

    每条轮廓最后都会做一次**兜底复核**（见 :func:`_max_deviation`）：简化结果若偏离
    原始轮廓超过容差，就退回简化前的点列。宁可留几个毛刺，也绝不输出被塌掉的轮廓。
    """
    h, w = labels.shape[:2]
    lines: list[Line] = []
    if labels.size == 0 or int(labels.max()) <= 0:
        return lines

    boxes = _component_slices(labels)
    # 复核门限：简化本身保证 ≤容差，薄片合并把点挪到邻点中点会再引入约 1×薄片跨度，
    # 这里留到 2 倍跨度 + 2 倍容差；超过就说明轮廓被塌掉了，退回未简化的点列。
    max_dev_allowed = max(1.0, 2.0 * float(cfg.thin_span) + 2.0 * float(cfg.simplify_tolerance))

    for cid, (bx, by, bw, bh) in boxes.items():
        patch = labels[by:by + bh, bx:bx + bw]
        comp = (patch == cid).astype(np.uint8)
        if not comp.any():
            continue
        for raw_local in _mask_contours(comp):
            if raw_local.shape[0] < 3:
                continue
            pts = raw_local + np.array([bx, by], dtype=np.float32)
            closed = is_closed(pts, tol=2.0)
            clipped = False
            if closed and open_at_border:
                pts, clipped = _split_border_span(pts, w, h)
                closed = is_closed(pts, tol=2.0) and not clipped
            if pts.shape[0] < 2:
                continue

            simp = simplify_polyline(pts, cfg.simplify_tolerance, closed=closed)
            if simp.shape[0] < 2:
                continue
            # 兜底：容差下简化会偏离原始轮廓超过门限时，逐步收紧容差重试，
            # 取"仍贴合轮廓的前提下最简"的那个结果。全程失败才退回原始点列
            # （宁可给出一条密但不走形的轮廓，也不给出一条塌掉的长直线）。
            if _max_deviation(pts, simp) > max_dev_allowed:
                best = None
                tol = float(cfg.simplify_tolerance)
                for _ in range(6):
                    tol *= 0.5
                    if tol < 0.2:
                        break
                    cand = simplify_polyline(pts, tol, closed=closed)
                    if cand.shape[0] >= 2 and _max_deviation(pts, cand) <= max_dev_allowed:
                        best = cand
                        break
                if best is not None:
                    simp = best
                else:
                    fallback = dedupe_points(np.asarray(pts, dtype=np.float64)).astype(np.float32)
                    if fallback.shape[0] >= 2:
                        simp = fallback
            # 注意：``simplify_polyline(closed=True)`` 返回的点列已显式回到起点，
            # 这里不必再补（闭合轮廓缺最后一段会让"线描到一半消失"）。
            if simp.shape[0] < 2:
                continue

            length = polyline_length(simp, closed=closed)
            if length < 4.0:
                continue
            longest_len, longest_ang = _longest_segment(simp, closed)
            lines.append(
                Line(
                    simp,
                    kind="outer",
                    crystal_id=cid,
                    attrs={
                        "closed": bool(closed),
                        "clipped_at_border": bool(clipped),
                        "length": float(length),
                        "longest_len": float(longest_len),
                        "longest_angle": float(longest_ang),
                    },
                )
            )
    return lines


def _longest_segment(pts: np.ndarray, closed: bool) -> tuple[float, float]:
    p = pts
    if closed and not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[:1]])
    if p.shape[0] < 2:
        return 0.0, 0.0
    d = np.diff(p, axis=0)
    seg_len = np.hypot(d[:, 0], d[:, 1])
    if seg_len.size == 0:
        return 0.0, 0.0
    i = int(np.argmax(seg_len))
    return float(seg_len[i]), float(orientation_deg(d[i]))


def assign_crystal_ids(lines: list[Line], labels: np.ndarray, max_checks: int = 3) -> None:
    """给每条线标注所属晶体 id（取折线上若干个采样点所在晶体）。"""
    if not lines:
        return
    h, w = labels.shape[:2]
    for ln in lines:
        votes: dict[int, int] = {}
        m = ln.n
        if m == 0:
            continue
        idxs = np.linspace(0, m - 1, num=min(max_checks, m)).astype(int)
        for i in idxs:
            x = int(np.clip(round(float(ln.points[i, 0])), 0, w - 1))
            y = int(np.clip(round(float(ln.points[i, 1])), 0, h - 1))
            v = int(labels[y, x])
            if v > 0:
                votes[v] = votes.get(v, 0) + 1
        if votes:
            ln.crystal_id = max(votes.items(), key=lambda kv: kv[1])[0]


def analyze_outer(
    labels: np.ndarray,
    cfg: SegConfig,
    lab: np.ndarray | None = None,
    *,
    open_at_border: bool = True,
) -> list[Crystal]:
    """生成晶体对象列表（含外轮廓折线）。"""
    n = int(labels.max())
    out: list[Crystal] = []
    outer = extract_outer_lines(labels, cfg, open_at_border=open_at_border)
    by_id: dict[int, list[Line]] = {}
    for ln in outer:
        by_id.setdefault(ln.crystal_id, []).append(ln)
    # 一次扫描统计所有晶体的面积与包围盒（比对每块晶体单独 nonzero 快得多）
    flat = labels.ravel()
    nz = np.nonzero(flat > 0)[0]
    if nz.size == 0:
        return out
    vals = flat[nz].astype(np.int64)
    area = np.bincount(vals, minlength=n + 1)
    yy = (nz // labels.shape[1]).astype(np.int64)
    xx = (nz % labels.shape[1]).astype(np.int64)
    min_y = np.full(n + 1, np.iinfo(np.int64).max, dtype=np.int64)
    max_y = np.full(n + 1, -1, dtype=np.int64)
    min_x = np.full(n + 1, np.iinfo(np.int64).max, dtype=np.int64)
    max_x = np.full(n + 1, -1, dtype=np.int64)
    np.minimum.at(min_y, vals, yy)
    np.maximum.at(max_y, vals, yy)
    np.minimum.at(min_x, vals, xx)
    np.maximum.at(max_x, vals, xx)

    med_colors: dict[int, np.ndarray] = {}
    if lab is not None:
        # 每块晶体的 Lab 中位色（报告/调试用）。按晶体分组统计，量小，开销可忽略。
        lab_flat = lab.reshape(-1, 3)
        for cid in range(1, n + 1):
            m = vals == cid
            if not m.any():
                continue
            med_colors[cid] = np.median(lab_flat[nz[m]], axis=0).astype(np.float32)

    for cid in range(1, n + 1):
        if area[cid] == 0:
            continue
        out.append(
            Crystal(
                crystal_id=cid,
                area=int(area[cid]),
                bbox=(
                    int(min_x[cid]),
                    int(min_y[cid]),
                    int(max_x[cid] - min_x[cid] + 1),
                    int(max_y[cid] - min_y[cid] + 1),
                ),
                lines=by_id.get(cid, []),
                median_color=med_colors.get(cid),
            )
        )
    del flat, nz, vals
    return out


def build_crystals(
    labels: np.ndarray, cfg: SegConfig, lab: np.ndarray | None = None
) -> list[Crystal]:
    """兼容入口：等价于 :func:`analyze_outer`。"""
    return analyze_outer(labels, cfg, lab)
