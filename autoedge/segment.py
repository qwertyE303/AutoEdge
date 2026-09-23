"""晶体提取：从图像中得到每块晶体的区域掩码。

判据选择（来自实测的可分性分析）：

* 单纯灰度在弱对比样本上 AUC 只有 0.63（单层 WS2/WSe2、暗楔形根本分不出来）；
* **"每个像素到基底主色的 Lab 色差"** 在强/弱对比样本上分别是 0.97 / 0.94。

因此以"到基底主色的色差"为主判据，配套四个稳健化处理：

1. **基底主色用框内单一稳健色**：ROI 内光照近似恒定，不需要空间缓变场；
   而"分块低分位 + 插值"在晶体附近会把基底色拉向晶体，把色差标尺整体压扁
   （实测晶体内部 ΔE 被压到 1~4，本该是 127），并使边界处出现长达 70~100 px 的
   假过渡带 —— 阈值取多少就决定边界缩进去多少，这是"描边比晶体小一圈"的主因。
2. **阈值：绿点锚定 > 三角法（Zack）**，并额外算一个**低阈值**：
   一个 ROI 里不同厚度区的色差可以差一个数量级（实测 118 / 125 / 10），
   单阈值必然二选一（要么漏薄区、要么多画）。
3. **滞后生长（高低双阈值）**：高阈值取"确定的晶体"作种子，低阈值负责铺满，
   只保留与种子连通的成分 —— 于是基底噪声碎块被自动丢弃，
   而"与厚区相连的薄区"被纳入。实测这一条把右侧薄区的覆盖率从 0.2% 提到 93.7%。
4. **边界脊线锚定**：晶体的物理边缘是"颜色变化最快处"，即 |∇ΔE| 的脊线。
   掩码边界沿法线吸附到脊线上，使边界不随阈值漂移（阈值只决定"哪些地方算晶体"，
   不决定"边界画在哪")。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .roi import RoiSet

__all__ = ["SegmentationResult", "extract_crystals"]


@dataclass
class SegmentationResult:
    """晶体分割结果。

    Attributes:
        labels: ``(H, W)`` int32，0 为背景，每个晶体一个正整数 id（原图分辨率）。
        threshold: 实际使用的色差阈值。
        method: 阈值来源（``anchor`` / ``triangle``）。
        stats: 统计信息。
        roi_mask: ``(H, W)`` uint8，实际参与分析的区域（无 ROI 时全 1）。
        info: 诊断信息。
    """

    labels: np.ndarray
    threshold: float
    threshold_low: float = 0.0
    method: str = ""
    stats: dict = field(default_factory=dict)
    roi_mask: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.labels.max())


# ---------------------------------------------------------------------------- 工具
def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """填补掩码内部孔洞（晶体内部的暗坑不应算作背景的洞）。"""
    pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(pad, ff, (0, 0), 1)
    holes = (pad[1:-1, 1:-1] == 0).astype(np.uint8)
    if not holes.any():
        return mask
    return np.where(holes > 0, np.uint8(1), mask)


def _triangle_threshold(hist: np.ndarray) -> int:
    """三角法自动阈值，适合"巨大峰 + 稀疏长尾"的分布。

    从直方图峰值向最远端拉一条直线，取离该直线最远的直方图点作为阈值。
    """
    n = int(hist.size)
    if n == 0 or hist.sum() <= 0:
        return 0
    peak = int(np.argmax(hist))
    far = n - 1
    while far > peak and hist[far] <= 0:
        far -= 1
    if far <= peak:
        return 0
    x1, y1 = float(far), float(hist[far])
    x2, y2 = float(peak), float(hist[peak])
    dx, dy = x2 - x1, y2 - y1
    norm = max(1e-9, float(np.hypot(dx, dy)))
    best_d, best_i = -1.0, 0
    for i in range(peak + 1, far + 1):
        d = abs(dy * (i - x1) - dx * (hist[i] - y1)) / norm
        if d > best_d:
            best_d, best_i = d, i
    return best_i


def substrate_color(
    bgr: np.ndarray,
    block_div: float = 6.0,
    valid: np.ndarray | None = None,
    window: int = 0,
    percentile: float = 25.0,
    *,
    flat: bool = True,
) -> np.ndarray:
    """估计"基底主色"。

    默认（``flat=True``）返回**区域内单一稳健色**：取"亮度低 ``percentile`` 分位"
    的那批像素的颜色中位数。ROI 内光照近似恒定，单一色不需要任何滤波核，
    也不会像空间缓变场那样把边界摊平。

    为什么要放弃空间缓变场（``flat=False`` 的旧行为）：分块低分位 + 双线性插值
    在"晶体远大于块"时，块内低分位仍会取到晶体像素，插值出来的场在空间上是一条
    斜坡 —— 实测基底色场在晶体内部被拉到 205（= 晶体自己的灰度），
    于是晶体内部 ΔE 从应有的 127 掉到 1~4，边界处出现 70~100 px 的假过渡带。

    Args:
        bgr: ``(h, w, 3)`` uint8 或 float32。
        block_div: 缓变场模式下窗口边长 = 区域长边 / 该值（``flat=False`` 时使用）。
        valid: ``(h, w)`` bool，只在这些像素里统计（一般是 ROI 内）。
        window: 显式指定缓变场窗口边长（像素）；0 表示自动。
        percentile: 低分位百分比。
        flat: 是否返回单一色（推荐 True）。

    Returns:
        ``(h, w, 3)`` float32 的基底色（单一色时为常量场）。
    """
    h, w = bgr.shape[:2]
    sub = np.clip(bgr.astype(np.float32), 0, 255).astype(np.uint8)
    if valid is None:
        valid = np.ones((h, w), dtype=bool)
    if not valid.any():
        valid = np.ones((h, w), dtype=bool)

    if flat:
        gray_all = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gp = gray_all[valid]
        t = float(np.percentile(gp, percentile))
        sel = sub[valid & (gray_all <= t)].astype(np.float32)
        if sel.shape[0] < 16:
            sel = sub[valid].astype(np.float32)
        c = np.median(sel, axis=0).astype(np.float32)
        return np.broadcast_to(c, (h, w, 3)).copy()

    win = int(window) if window > 0 else int(
        np.clip(round(min(h, w) / 5.0), 48, 256)
    )
    win = max(8, min(win, min(h, w) // 2 if min(h, w) >= 16 else max(4, min(h, w))))
    if win % 2 == 0:
        win += 1

    # 分块：块内取低分位亮度的那部分像素，用其颜色中位数代表该块基底
    bs = max(8, win)
    ny = max(1, int(np.ceil(h / bs)))
    nx = max(1, int(np.ceil(w / bs)))
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sub_f = sub.astype(np.float32)
    base_small = np.zeros((ny, nx, 3), dtype=np.float32)
    filled = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        y0, y1 = j * bs, min(h, (j + 1) * bs)
        for i in range(nx):
            x0, x1 = i * bs, min(w, (i + 1) * bs)
            vm = valid[y0:y1, x0:x1].ravel()
            if vm.sum() < 16:
                continue
            patch = sub_f[y0:y1, x0:x1].reshape(-1, 3)[vm]
            gp = gray[y0:y1, x0:x1].ravel()[vm]
            t = float(np.percentile(gp, percentile))
            sel = patch[gp <= t]
            if sel.shape[0] < 8:
                sel = patch
            base_small[j, i] = np.median(sel, axis=0)
            filled[j, i] = True
    if not filled.any():
        base_small[:] = np.median(sub_f[valid], axis=0) if valid.any() else 0.0
        filled[:] = True
    elif not filled.all():
        # 未填充的块用最近的已填充块顶替，避免插值出黑块
        idx = np.argwhere(filled)
        for j in range(ny):
            for i in range(nx):
                if filled[j, i]:
                    continue
                d = np.abs(idx[:, 0] - j) + np.abs(idx[:, 1] - i)
                k = idx[int(np.argmin(d))]
                base_small[j, i] = base_small[k[0], k[1]]

    if (ny, nx) == (1, 1):
        return np.broadcast_to(base_small[0, 0], (h, w, 3)).copy()
    return cv2.resize(base_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def distance_from_base(bgr: np.ndarray, base_rgb: np.ndarray) -> np.ndarray:
    """每个像素到基底主色的 Lab 色差。"""
    img_u8 = np.clip(bgr, 0, 255).astype(np.uint8)
    base_u8 = np.clip(base_rgb, 0, 255).astype(np.uint8)
    lab_img = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_base = cv2.cvtColor(base_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.linalg.norm(lab_img - lab_base, axis=2)


def choose_threshold(
    dist: np.ndarray,
    valid: np.ndarray,
    *,
    sensitivity: float = 1.0,
    anchors: list[tuple[float, float]] | None = None,
    negatives: list[tuple[float, float]] | None = None,
    offset: tuple[int, int] = (0, 0),
    low_factor: float = 0.5,
) -> tuple[float, float, str, dict]:
    """决定**高/低两个**色差阈值（滞后阈值）。

    高阈值（种子）：只有"确定的晶体"才过关。
    低阈值（生长）：铺满用，只要与种子连通即可纳入 —— 于是"与厚区相连的
    薄区/单层区"能被包含，而基底上的孤立噪声碎块因为不连种子被丢弃。

    高阈值的优先级（越靠前越有依据）：

    1. **绿点 + 红点**：直接用"使用者声明的晶体处"与"声明的基底处"的实测色差
       取中间偏基底的值。这是最可靠的方式。
    2. **只有绿点**：取绿点处色差的中位数的一个比例。
    3. **都没有**：三角法谷底。

    低阈值以**最弱的那个绿点**为准（乘 ``low_factor``）：
    实测同一 ROI 内不同厚度区的色差可以差一个数量级（118 / 125 / 10），
    用中位数会被强区绑架，薄区永远进不来。

    三者都以"分布左半边"估计的噪声尺度设下限。
    另外会给出 ``signal_weak`` 标记：绿点与红点几乎无差别时说明该区域
    信号低于可分辨水平，此时任何分割都不可靠，界面会给出警告。

    Returns:
        ``(thr_high, thr_low, method, info)``
    """
    info: dict = {}
    d = dist[valid]
    if d.size < 16:
        return 0.0, 0.0, "empty", info

    bg_level = float(np.percentile(d, 20))
    # 噪声尺度只能用分布的左半边估计：右侧是晶体，用 MAD 会被撑大数倍
    lo, hi = float(np.percentile(d, 5)), float(np.percentile(d, 45))
    sigma = max(0.3, (hi - lo) / 2.0)

    # 直方图只统计到 p99，避免个别极端像素把"最远端"拉到无意义的位置
    d_hi = float(np.percentile(d, 99.0))
    dmax = max(12.0, d_hi)
    bins = np.linspace(0.0, dmax, 256)
    hist, _ = np.histogram(np.minimum(d, dmax), bins=bins)
    hist = hist.astype(np.float64)
    k = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    hist = np.convolve(hist, k / k.sum(), mode="same")
    tri_bin = _triangle_threshold(hist)
    tri_val = float(bins[min(tri_bin, bins.size - 1)])

    # 下限只用来挡噪声。实测三角法给出的谷底（约 ΔE 20）已远高于噪声，
    # 若下限取到 3σ 以上会反客为主把阈值顶高，造成大面积漏检。
    floor = bg_level + 1.6 * sigma
    thr = max(floor, tri_val)
    thr_low = max(floor, thr * float(low_factor))
    method = "triangle"
    info.update({"triangle": tri_val, "floor": floor, "bg_level": bg_level, "sigma": sigma})

    def _sample(points: list[tuple[float, float]]) -> list[float]:
        hh, ww = dist.shape
        out_vals: list[float] = []
        for px, py in points:
            x = int(round(px)) - offset[0]
            y = int(round(py)) - offset[1]
            if 0 <= x < ww and 0 <= y < hh:
                out_vals.append(float(dist[y, x]))
        return out_vals

    pos_vals = _sample(anchors) if anchors else []
    neg_vals = _sample(negatives) if negatives else []

    # 记录每个提示点的"坐标 + 实测色差"，让使用者能核对程序读到了哪里。
    # 这一步是排查"点的地方和读到的位置不一致"这类问题的关键证据。
    if anchors:
        info["pos_detail"] = [
            {"x": float(px), "y": float(py), "de": round(float(v), 2)}
            for (px, py), v in zip(anchors, pos_vals, strict=False)
        ]
    if negatives:
        info["neg_detail"] = [
            {"x": float(px), "y": float(py), "de": round(float(v), 2)}
            for (px, py), v in zip(negatives, neg_vals, strict=False)
        ]

    # 优先级 1：绿点 + 红点 —— 直接用使用者声明的两类位置的实测差距定阈值
    if pos_vals and neg_vals:
        p_med = float(np.median(pos_vals))
        n_med = float(np.median(neg_vals))
        info["pos_median"] = p_med
        info["neg_median"] = n_med
        info["hint_gap"] = p_med - n_med
        if p_med > n_med + 0.5:
            # 取两者之间、偏红点一侧（宁可多画一点也不能漏）
            thr = max(floor, n_med + 0.35 * (p_med - n_med))
            method = "hint-gap"
        # 绿点与红点几乎无差别 -> 该区域信号低于可分辨水平
        info["signal_weak"] = bool((p_med - n_med) < max(1.0, 2.5 * sigma))

    # 优先级 2：只有绿点
    if method != "hint-gap" and pos_vals:
        anchor = float(np.median(pos_vals))
        # 系数取小：绿点落在亮区时中位色差偏大，不压下来会大面积漏检
        thr = max(floor, 0.25 * anchor)
        method = "anchor"
        info["anchor_dist"] = anchor
        info["anchor_min"] = float(np.min(pos_vals))

    thr = min(thr, bg_level + 0.98 * (d_hi - bg_level))
    thr = max(thr * float(sensitivity), floor)

    # ---- 低阈值（生长用）：以**最弱的那个绿点**为准 ----
    # 实测同一个 ROI 里不同厚度区的色差可以差一个数量级（118 / 125 / 10），
    # 若低阈值也跟着中位数走，薄区永远进不来。
    if pos_vals:
        weak = float(np.min(pos_vals))
        info["anchor_weak"] = weak
        thr_low = max(floor, weak * float(low_factor))
    else:
        thr_low = max(floor, thr * float(low_factor))
    thr_low = min(thr_low, thr)

    info["threshold"] = thr
    info["threshold_low"] = thr_low
    return thr, thr_low, method, info


def measure_hint_distances(
    bgr: np.ndarray, roi: RoiSet
) -> tuple[list[float], list[float], np.ndarray]:
    """量出提示点处的实测色差（用于给"描边阈值"滑块一个建议起始值）。

    做法与正式分割完全一致：框内单一稳健基底色 → Lab 色差。

    Args:
        bgr: ``(H, W, 3)`` 原图。
        roi: ROI 与提示点。

    Returns:
        ``(绿点处色差列表, 红点处色差列表, 框内色差分布的一维数组)``。
        若框为空则返回 ``([], [], array([]))``。
    """
    h, w = bgr.shape[:2]
    if roi is None or roi.empty:
        return [], [], np.zeros(0, np.float32)
    box = roi.bbox((h, w), pad=0)
    if box is None:
        return [], [], np.zeros(0, np.float32)
    x0, y0, x1, y1 = box
    sub = np.ascontiguousarray(bgr[y0:y1, x0:x1])
    valid = roi.mask((h, w), pad=0)[y0:y1, x0:x1].astype(bool)
    if not valid.any():
        return [], [], np.zeros(0, np.float32)
    base = substrate_color(sub, valid=valid, flat=True)
    dist = distance_from_base(sub, base)
    hh, ww = dist.shape

    def sample(pts: list[tuple[float, float]]) -> list[float]:
        out: list[float] = []
        for px, py in pts:
            ix, iy = int(round(px)) - x0, int(round(py)) - y0
            if 0 <= ix < ww and 0 <= iy < hh:
                out.append(float(dist[iy, ix]))
        return out

    return (
        sample(roi.positive_points()),
        sample(roi.negative_points()),
        dist[valid].astype(np.float32),
    )


def _keep_components(
    mask: np.ndarray, min_px: int, must_include: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """保留面积达标的连通域。

    Args:
        mask: ``(h, w)`` uint8。
        min_px: 面积下限（像素）。
        must_include: ``(h, w)`` bool。**只要某个连通域含有这些点，就无条件保留**——
            使用者点的绿点是"这里一定是晶体"的强声明，即使该连通域很小也不能丢。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    forced: set[int] = set()
    if must_include is not None and must_include.any():
        ys, xs = np.nonzero(must_include)
        for lb in np.unique(labels[ys, xs]):
            if lb > 0:
                forced.add(int(lb))
    out = np.zeros_like(mask)
    kept = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_px or i in forced:
            out[labels == i] = 1
            kept += 1
    return out, kept


def _hysteresis(
    dist: np.ndarray,
    valid: np.ndarray,
    thr_high: float,
    thr_low: float,
    *,
    min_px: int,
    must_include: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """滞后阈值：高阈值定种子，低阈值生长，只保留与种子连通的成分。

    这样"与厚区相连的薄区"能被纳入，而基底上孤立的噪声碎块（不与任何种子
    连通）被自动丢弃 —— 于是可以放心把低阈值压到噪声下限附近。

    Args:
        dist: ``(h, w)`` 色差场。
        valid: ``(h, w)`` bool，参与分析的区域。
        thr_high: 高阈值（种子）。
        thr_low: 低阈值（生长）。
        min_px: 连通域面积下限。
        must_include: 强制保留的像素（绿点邻域）。

    Returns:
        ``(mask, kept_count)``
    """
    weak = ((dist >= thr_low) & valid).astype(np.uint8)
    strong = ((dist >= thr_high) & valid).astype(np.uint8)
    if not weak.any():
        return np.zeros_like(weak), 0
    if not strong.any():
        # 没有种子时退回单阈值，但不能把整片低阈值区域都当成晶体
        return _keep_components(weak, min_px, must_include)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    forced: set[int] = set()
    for lb in np.unique(labels[strong > 0]):
        if lb > 0:
            forced.add(int(lb))
    if must_include is not None and must_include.any():
        ys, xs = np.nonzero(must_include)
        for lb in np.unique(labels[ys, xs]):
            if lb > 0:
                forced.add(int(lb))
    if not forced:
        return np.zeros_like(weak), 0
    # 面积达标、或"含有绿点声明"的连通域一律保留
    keep = {i for i in forced if int(stats[i, cv2.CC_STAT_AREA]) >= min_px}
    if not keep:
        keep = set(forced)
    out = np.isin(labels, list(keep)).astype(np.uint8)
    return out, len(keep)


def _snap_boundary_to_ridge(
    mask: np.ndarray,
    dist: np.ndarray,
    *,
    grad_sigma: float = 1.5,
    peak_ratio: float = 1.6,
    cut_ratio: float = 0.6,
    max_out: int = 14,
    max_in: int = 40,
) -> tuple[np.ndarray, int, int]:
    """把掩码边界吸附到色差脊线（沿法线**双向**搜索，只在存在明显更强的峰时移动）。

    晶体边缘是"颜色变化最快处"，即 ΔE 沿法线方向的极大值。掩码边界由阈值决定，
    会随阈值漂移（阈值越低越往外）；这里把边界重新定位到脊线上，
    使**阈值只决定"哪些地方算晶体"，不决定"边界画在哪"**。

    判定规则（只在证据充分时才动，避免在平坦基底上乱跑）：

    1. 沿法线向内 ``max_in``、向外 ``max_out`` 采样 ΔE，找极大值位置；
    2. 若"峰值 ≥ ``peak_ratio`` × 当前边界处的 ΔE" —— 说明近旁确实存在一条
       明显更强的脊线，则把边界移到峰值处；否则保持不动（基底内部的噪声起伏
       不构成脊线，掩码边界本来就不该动）；
    3. 移动方向若为向外，再从峰值继续走到 ΔE 掉到 ``cut_ratio`` × 峰值处，
       把"色差不高但确属晶体"的过渡带补上（这是薄区被完整包住的关键）；
    4. 移动方向若为向内，把外侧那一段从掩码里去掉（回收虚胖的边界）。

    Args:
        mask: ``(h, w)`` uint8 掩码。
        dist: ``(h, w)`` 色差场。
        grad_sigma: 计算梯度前的平滑尺度。
        peak_ratio: 判定"存在明显更强的脊线"的倍数阈值。
        cut_ratio: 向外续接的终止比例（相对峰值）。
        max_out: 向外搜索的最大步数。
        max_in: 向内搜索的最大步数。

    Returns:
        ``(new_mask, moved_count, added_px)``
    """
    h, w = mask.shape
    if not mask.any():
        return mask, 0, 0

    ys, xs = np.nonzero(mask)
    pad = max(max_out, max_in) + 3
    y0 = max(0, int(ys.min()) - pad)
    x0 = max(0, int(xs.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x1 = min(w, int(xs.max()) + pad + 1)
    m = np.ascontiguousarray(mask[y0:y1, x0:x1])
    de = np.ascontiguousarray(dist[y0:y1, x0:x1])
    hh, ww = de.shape

    de_s = cv2.GaussianBlur(de, (0, 0), grad_sigma)
    gx = cv2.Sobel(de_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(de_s, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) + 1e-6
    nx = gx / mag
    ny = gy / mag

    boundary = (m > 0) & (cv2.erode(m, np.ones((3, 3), np.uint8)) == 0)
    bys, bxs = np.nonzero(boundary)
    n_b = bys.size
    if n_b == 0:
        return mask, 0, 0

    # 双向：steps<0 为法线负方向，steps>0 为正方向，0 为当前边界
    steps = np.concatenate(
        [np.arange(-max_in, 0, dtype=np.float32), np.arange(0, max_out + 1, dtype=np.float32)]
    )
    ns = steps.size
    zero = int(max_in)

    from scipy.ndimage import map_coordinates

    dirx = nx[bys, bxs]
    diry = ny[bys, bxs]

    def sample(sign: float) -> np.ndarray:
        px = bxs[:, None].astype(np.float32) + (sign * steps)[None, :] * dirx[:, None]
        py = bys[:, None].astype(np.float32) + (sign * steps)[None, :] * diry[:, None]
        np.clip(px, 0, ww - 1, out=px)
        np.clip(py, 0, hh - 1, out=py)
        v = map_coordinates(
            de,
            [py.ravel().astype(np.float64), px.ravel().astype(np.float64)],
            order=1,
            mode="nearest",
        ).reshape(n_b, ns)
        return v, px, py

    vp, pxp, pyp = sample(+1.0)
    vn, pxn, pyn = sample(-1.0)

    here = vp[:, zero]
    kp = np.argmax(vp, axis=1)
    kn = np.argmax(vn, axis=1)
    peak_p = vp[np.arange(n_b), kp]
    peak_n = vn[np.arange(n_b), kn]
    use_pos = peak_p >= peak_n

    idx = np.arange(n_b)
    vals = np.where(use_pos[:, None], vp, vn)
    px_all = np.where(use_pos[:, None], pxp, pxn)
    py_all = np.where(use_pos[:, None], pyp, pyn)
    peak_i = np.where(use_pos, kp, kn)
    peak_v = np.where(use_pos, peak_p, peak_n)

    moved_mask = peak_v >= np.maximum(here, 1e-6) * peak_ratio  # 证据充分才移动
    out_i = peak_i > zero      # 峰在外侧 -> 外扩
    in_i = (peak_i < zero) & moved_mask  # 峰在内侧 -> 回收

    add = np.zeros((hh, ww), np.uint8)
    cut = np.zeros((hh, ww), np.uint8)

    # 外扩：从当前边界一路补到"ΔE 掉到 cut_ratio×峰值"处
    for i in np.nonzero(out_i & moved_mask)[0]:
        v = vals[i]
        end = int(peak_i[i])
        for j in range(int(peak_i[i]) + 1, ns):
            if v[j] < cut_ratio * peak_v[i] or j - peak_i[i] > max_out:
                break
            end = j
        js = np.arange(zero, end + 1)
        ax = np.rint(px_all[i, js]).astype(np.int32)
        ay = np.rint(py_all[i, js]).astype(np.int32)
        ok = (ax >= 0) & (ax < ww) & (ay >= 0) & (ay < hh)
        add[ay[ok], ax[ok]] = 1

    # 回收：把当前边界到外侧那一段擦掉
    for i in np.nonzero(in_i)[0]:
        js = np.arange(zero, ns)
        ax = np.rint(px_all[i, js]).astype(np.int32)
        ay = np.rint(py_all[i, js]).astype(np.int32)
        ok = (ax >= 0) & (ax < ww) & (ay >= 0) & (ay < hh)
        cut[ay[ok], ax[ok]] = 1

    moved = int((moved_mask & (out_i | in_i)).sum())

    out = mask.copy()
    sub = out[y0:y1, x0:x1]
    if cut.any():
        sub[cut > 0] = 0
    if add.any():
        sub |= add
    out[y0:y1, x0:x1] = sub
    return out, moved, int(add.sum())


def _clean_and_label(
    mask: np.ndarray,
    *,
    min_area: int,
    max_area: int,
    open_radius: int,
    fill_holes: bool,
    negative_points: list[tuple[float, float]],
) -> tuple[np.ndarray, int, int]:
    """形态学清理 + 连通域打标（面积过滤 + 红点剔除）。

    性能要点：只在掩码的**包围盒**内做形态学与连通域运算，
    避免在 4928x3264 全图上反复分配大数组。
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros((h, w), np.int32), 0, 0
    pad = max(4, int(open_radius) * 3 + 2)
    y0 = max(0, int(ys.min()) - pad)
    x0 = max(0, int(xs.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    sub = np.ascontiguousarray(mask[y0:y1, x0:x1])

    if open_radius > 0:
        r = int(open_radius)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, k)
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, k)
    if fill_holes:
        sub = _fill_holes(sub)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    out = np.zeros((h, w), dtype=np.int32)
    keep = 0
    rejected = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or (max_area and area > max_area):
            continue
        if negative_points:
            hit = False
            for px, py in negative_points:
                ix = int(np.clip(round(px) - x0, 0, sub.shape[1] - 1))
                iy = int(np.clip(round(py) - y0, 0, sub.shape[0] - 1))
                if labels[iy, ix] == i:
                    hit = True
                    break
            if hit:
                rejected += 1
                continue
        keep += 1
        oy, ox = np.nonzero(labels == i)
        out[oy + y0, ox + x0] = keep
    return out, keep, rejected


# ---------------------------------------------------------------------------- 主入口
def extract_crystals(
    bgr: np.ndarray,
    *,
    threshold: float = 12.0,
    seed_factor: float = 2.5,
    min_area: int = 100,
    max_area: int = 0,
    open_radius: int = 1,
    fill_holes: bool = True,
    roi: RoiSet | None = None,
    apply_ridge_snap: bool = True,
    ridge_peak_ratio: float = 1.6,
    ridge_cut_ratio: float = 0.6,
    ridge_max_out: int = 14,
    ridge_max_in: int = 40,
) -> SegmentationResult:
    """提取晶体区域。

    流程：单一基底色 → 色差场 ΔE → 双阈值（滞后生长）→ 边界脊线锚定。

    阈值只有一个主控参数（``threshold``，界面上的"描边阈值"）：
    它同时是滞后生长的**低阈值**；高阈值（种子）由 ``threshold × seed_factor``
    导出，只用于"哪些弱区值得纳入"的判断，不暴露给使用者。

    Args:
        bgr: ``(H, W, 3)`` uint8 原图。
        threshold: 描边阈值（Lab 色差 ΔE）。越小越灵敏。
        seed_factor: 高阈值（种子）= 描边阈值 × 该值。
        min_area: 面积下限（像素）。
        max_area: 面积上限（像素），0 表示不限制。
        open_radius: 形态学开运算半径（像素）。
        fill_holes: 是否填补晶体内部孔洞。
        roi: ROI 与提示点。给出且非空时**只在框内分析**（推荐）。
        apply_ridge_snap: 是否把边界吸附到色差脊线。
        ridge_peak_ratio: 判定"存在明显更强的脊线"的倍数。
        ridge_cut_ratio: 向外续接终止比例（相对脊线处 ΔE）。
        ridge_max_out: 脊线向外搜索/外接的最大距离（像素）。
        ridge_max_in: 脊线向内搜索的最大距离（像素）。

    Returns:
        :class:`SegmentationResult`
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"extract_crystals 需要 (H, W, 3) 的 BGR 图像，实际 {bgr.shape}")
    h, w = bgr.shape[:2]
    use_roi = roi is not None and not roi.empty

    if use_roi:
        assert roi is not None
        roi_mask = roi.mask((h, w), pad=0)
        box = roi.bbox((h, w), pad=0)
        if box is None:
            return SegmentationResult(
                labels=np.zeros((h, w), np.int32), threshold=0.0, method="roi-empty"
            )
        x0, y0, x1, y1 = box
        sub_bgr = np.ascontiguousarray(bgr[y0:y1, x0:x1])
        sub_valid = roi_mask[y0:y1, x0:x1].astype(bool)

        # 基底色与阈值都只使用框内数据，避免被框外大片暗基底与四角亮弧带偏。
        # 基底色用**单一稳健色**：缓变场会在晶体附近把基底色拉向晶体，
        # 把色差标尺压扁（实测晶体内部 ΔE 从 127 掉到 1~4）。
        base = substrate_color(sub_bgr, valid=sub_valid, flat=True)
        dist = distance_from_base(sub_bgr, base)
        thr_low = float(max(0.5, threshold))
        thr = float(max(thr_low, thr_low * max(1.0, seed_factor)))
        method = "manual"
        info: dict = {
            "threshold": thr,
            "threshold_low": thr_low,
            "seed_factor": float(seed_factor),
            "base_color": [round(float(v), 1) for v in base[0, 0]],
        }
        # 记录每个提示点的实测色差，便于核对标注落点与取值
        hh0, ww0 = dist.shape
        for tag, pts in (("pos_detail", roi.positive_points()), ("neg_detail", roi.negative_points())):
            rows = []
            for px, py in pts:
                ix, iy = int(round(px)) - x0, int(round(py)) - y0
                if 0 <= ix < ww0 and 0 <= iy < hh0:
                    rows.append(
                        {"x": float(px), "y": float(py), "de": round(float(dist[iy, ix]), 2)}
                    )
            if rows:
                info[tag] = rows

        # 绿点所在连通域无条件保留（使用者明确声明"这里是晶体"）
        anchor_mask = np.zeros((h, w), np.uint8)
        for px, py in roi.positive_points():
            cv2.circle(anchor_mask, (int(round(px)), int(round(py))), 6, 1, -1)
        anchor_mask = anchor_mask[y0:y1, x0:x1].astype(bool)

        # 滞后阈值：高阈值种子 + 低阈值生长（连通域下限用固定值，
        # 不再随框面积放大 —— 否则大框里的小晶体碎块会被整体砍掉）
        min_px = max(1, int(min_area))
        sub_mask, comps = _hysteresis(
            dist, sub_valid, thr, thr_low, min_px=min_px, must_include=anchor_mask
        )
        if apply_ridge_snap:
            sub_mask_snapped, moved, added = _snap_boundary_to_ridge(
                sub_mask,
                dist,
                peak_ratio=ridge_peak_ratio,
                cut_ratio=ridge_cut_ratio,
                max_out=ridge_max_out,
                max_in=ridge_max_in,
            )
            # 回收边界可能切出小碎屑，按面积再筛一次（只保留大块）
            if moved:
                sub_mask_snapped, _ = _keep_components(sub_mask_snapped, min_px)
        else:
            sub_mask_snapped, moved, added = sub_mask, 0, 0
        info.update(
            {
                "mode": "roi",
                "roi_box": (x0, y0, x1, y1),
                "roi_pixels": int(sub_valid.sum()),
                "roi_count": len(roi.polygons),
                "valid_pixels": int(sub_valid.sum()),
                "min_component_px": min_px,
                "components_after_threshold": comps,
                "ridge_moved": int(moved),
                "ridge_added_px": int(added),
            }
        )
        full_mask = np.zeros((h, w), np.uint8)
        full_mask[y0:y1, x0:x1] = sub_mask_snapped
        pos = roi.positive_points()
        neg = roi.negative_points()
        for px, py in pos:
            cv2.circle(full_mask, (int(round(px)), int(round(py))), 8, 1, -1)
        info["hints_positive"] = len(pos)
        info["hints_negative"] = len(neg)

        out, kept, rejected = _clean_and_label(
            full_mask,
            min_area=max(1, int(min_area)),
            max_area=max_area,
            open_radius=open_radius,
            fill_holes=fill_holes,
            negative_points=neg,
        )
        info["rejected_negative"] = rejected
        areas = [int(a) for a in np.bincount(out.ravel())[1:] if a > 0]
        return SegmentationResult(
            labels=out,
            threshold=float(thr),
            threshold_low=float(thr_low),
            method=method,
            stats={
                "kept": kept,
                "rejected_negative": rejected,
                "rejected_corner": 0,
                "areas": areas,
                "largest": max(areas) if areas else 0,
            },
            roi_mask=roi_mask,
            info=info,
        )

    # ---------------- 无 ROI：全自动模式（兜底） ----------------
    base = substrate_color(bgr, flat=True)
    dist = distance_from_base(bgr, base)
    thr_low = float(max(0.5, threshold))
    thr = float(max(thr_low, thr_low * max(1.0, seed_factor)))
    method = "manual"
    info: dict = {"threshold": thr, "threshold_low": thr_low, "mode": "auto"}
    min_px = max(1500, int(min_area))
    mask, comps = _hysteresis(dist, np.ones((h, w), dtype=bool), thr, thr_low, min_px=min_px)
    if apply_ridge_snap:
        mask, moved, added = _snap_boundary_to_ridge(
            mask,
            dist,
            peak_ratio=ridge_peak_ratio,
            cut_ratio=ridge_cut_ratio,
            max_out=ridge_max_out,
            max_in=min(ridge_max_in, 12),
        )
        if moved:
            mask, _ = _keep_components(mask, min_px)
    else:
        moved, added = 0, 0
    info.update(
        {
            "mode": "auto",
            "roi_pixels": int(h * w),
            "components_after_threshold": comps,
            "ridge_moved": int(moved),
            "ridge_added_px": int(added),
        }
    )
    out, kept, rejected = _clean_and_label(
        mask,
        min_area=max(1, int(min_area)),
        max_area=max_area,
        open_radius=open_radius,
        fill_holes=fill_holes,
        negative_points=[],
    )
    areas = [int(a) for a in np.bincount(out.ravel())[1:] if a > 0]
    return SegmentationResult(
        labels=out,
        threshold=float(thr),
        threshold_low=float(thr_low),
        method=method,
        stats={
            "kept": kept,
            "rejected_negative": rejected,
            "rejected_corner": 0,
            "areas": areas,
            "largest": max(areas) if areas else 0,
        },
        roi_mask=np.ones((h, w), np.uint8),
        info=info,
    )
