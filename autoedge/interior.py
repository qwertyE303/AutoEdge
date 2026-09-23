"""内部分界线的提取。

物理背景：同一块层状晶体不同区域的**厚度突变**会在光学显微镜下产生色差，
这些色差区域的公共边界就是需要描的内部分界线。厚度连续渐变（没有突变）的地方
不应出现线条——这由使用者实时调节的"色差阈值"控制。

实现思路（全部为可解释的传统方法）：

1. 在降采样图上做 SLIC 超像素分割，得到候选的"厚度区"。
2. 对每个区域取区域内的 Lab 中位色（排除晶体外像素）。
3. 相邻区域的中位色差 ΔE 超过阈值 -> 它们的公共边界是真实分界线，保留；
   否则丢弃。阈值可由 Otsu 自动给出下限，使用者再手动加严。
4. 把保留边界骨架化并串成有序折线。
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from .geometry import Line, polyline_length

__all__ = ["build_region_labels", "extract_interior_lines", "region_color_distances"]


def _downscale(a: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return a
    return a[::scale, ::scale]


def build_region_labels(
    lab: np.ndarray,
    labels: np.ndarray,
    *,
    scale: int = 6,
    compactness: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """对晶体区域做超像素分割。

    Args:
        lab: ``(H, W, 3)`` uint8 Lab 图。
        labels: ``(H, W)`` int32 晶体标记图。
        scale: 超像素目标尺度（像素，原图分辨率）。内部按 2 倍降采样计算以提速，
            坐标再映射回原图，几何误差不超过 2 像素。
        compactness: SLIC 紧凑度。

    Returns:
        ``(region_labels, med_colors, n_regions)``。``region_labels`` 为
        ``(H, W)`` int32（0 表示晶体外），``med_colors`` 为 ``(n_regions + 1, 3)``
        float32 的 Lab 中位色。
    """
    from skimage.segmentation import slic

    h, w = labels.shape[:2]
    # 降采样以控制 SLIC 的开销：目标是让"超像素在原图上的边长 ≈ scale 像素"
    ds = 4 if scale >= 3 else 2
    if min(h, w) // ds < 32:
        ds = 1
    lab_s = np.ascontiguousarray(_downscale(lab, ds))
    size_on_small = max(2, int(round(float(scale) / ds)))
    n_segments = int(round(lab_s.shape[0] * lab_s.shape[1] / float(size_on_small**2)))
    n_segments = int(np.clip(n_segments, 16, 20000))
    seg = slic(
        lab_s,
        n_segments=n_segments,
        compactness=float(compactness),
        sigma=1.0,
        start_label=0,
        channel_axis=2,
    ).astype(np.int32)

    # 只保留晶体内部的像素参与颜色统计
    labels_s = _downscale(labels, ds)
    inside = labels_s > 0
    n_seg = int(seg.max()) + 1
    med = np.zeros((n_seg, 3), dtype=np.float32)
    counts = np.zeros(n_seg, dtype=np.int64)
    flat_seg = seg.ravel()
    flat_in = inside.ravel()
    flat_lab = lab_s.reshape(-1, 3)
    if flat_in.any():
        idx = flat_seg[flat_in]
        np.add.at(counts, idx, 1)
        for c in range(3):
            sums = np.bincount(idx, weights=flat_lab[flat_in, c].astype(np.float64), minlength=n_seg)
            with np.errstate(invalid="ignore", divide="ignore"):
                med[:, c] = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    # 把所有"完全在晶体外"的超像素并入 0（背景），避免它们造出假边界
    med[counts == 0] = 0.0
    reg = seg + 1  # 0 留给背景
    reg[~inside] = 0
    return reg.astype(np.int32), med, n_seg


def region_color_distances(
    reg_small: np.ndarray,
    med: np.ndarray,
    scale: int,
    labels_small: np.ndarray,
) -> tuple[dict[tuple[int, int], float], np.ndarray]:
    """计算所有相邻区域对的色差 ΔE（全向量化，支持数万区域）。

    Args:
        reg_small: ``(h, w)`` int32 区域标记（降采样后，0 为背景）。
        med: ``(n + 1, 3)`` 区域 Lab 中位色。
        scale: 降采样倍数。
        labels_small: ``(h, w)`` int32 晶体标记（降采样后）。

    Returns:
        ``(pair_delta, sample_xy)``：``pair_delta[(a, b)] = ΔE``；
        ``sample_xy`` 为 ``(n + 1, 2)`` 的 ``(x, y)`` 原图坐标。
    """
    n = int(reg_small.max())
    base = np.int64(n) + 2
    codes: list[np.ndarray] = []
    for a_row, b_row in (
        (reg_small[:, :-1], reg_small[:, 1:]),
        (reg_small[:-1, :], reg_small[1:, :]),
    ):
        m = (a_row != b_row) & (a_row > 0) & (b_row > 0)
        if m.any():
            av = a_row[m].astype(np.int64)
            bv = b_row[m].astype(np.int64)
            lo = np.minimum(av, bv)
            hi = np.maximum(av, bv)
            codes.append(lo * base + hi)
    if not codes:
        return {}, np.zeros((n + 1, 2), dtype=np.int32)

    all_codes = np.unique(np.concatenate(codes))
    ia = (all_codes // base).astype(np.int64)
    ib = (all_codes % base).astype(np.int64)
    diff = med[ia] - med[ib]
    dist = np.sqrt((diff.astype(np.float64) ** 2).sum(axis=1))
    pairs = {  # 区域对数量可达数万，用 zip 构造字典比逐项索引快
        (int(a), int(b)): float(d)
        for a, b, d in zip(ia.tolist(), ib.tolist(), dist.tolist(), strict=False)
    }

    # 每个区域的代表点：用行优先第一个像素的近似（取每行首次出现位置）
    sample_xy = np.zeros((n + 1, 2), dtype=np.int32)
    first = np.full(n + 1, -1, dtype=np.int64)
    flat = reg_small.ravel()
    idx = np.nonzero(flat > 0)[0]
    if idx.size:
        vals = flat[idx].astype(np.int64)
        # 逆序赋值，保证保留每个区域最小的线性下标
        first[vals[::-1]] = idx[::-1]
    valid = first > 0
    ys = (first[valid] // reg_small.shape[1]).astype(np.int32)
    xs = (first[valid] % reg_small.shape[1]).astype(np.int32)
    sample_xy[np.nonzero(valid)[0]] = np.stack([xs * scale, ys * scale], axis=1)
    return pairs, sample_xy


def _refine_x(full: np.ndarray, x: int, y: int, radius: int = 6) -> int:
    """在水平方向把边界位置吸附到本地梯度最强处。"""
    h, w = full.shape
    x0 = max(1, x - radius)
    x1 = min(w - 2, x + radius)
    if x1 <= x0:
        return int(np.clip(x, 0, w - 1))
    row = full[y, x0 - 1: x1 + 2].astype(np.float32)
    g = np.abs(np.diff(row))
    return int(x0 + int(np.argmax(g)))


def _refine_y(full: np.ndarray, x: int, y: int, radius: int = 6) -> int:
    """在垂直方向把边界位置吸附到本地梯度最强处。"""
    h, w = full.shape
    y0 = max(1, y - radius)
    y1 = min(h - 2, y + radius)
    if y1 <= y0:
        return int(np.clip(y, 0, h - 1))
    col = full[y0 - 1: y1 + 2, x].astype(np.float32)
    g = np.abs(np.diff(col))
    return int(y0 + int(np.argmax(g)))


def _upsample_mask(small: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把降采样得到的边界掩码用最近邻放大回原尺寸。

    最近邻放大不会像双线性那样模糊线条，代价只是边界有最多 ``ds`` 像素的偏移，
    随后由 :func:`extract_interior_lines` 的梯度细化把位置吸附回真实边缘。
    """
    h, w = shape
    if small.shape == (h, w):
        return small.astype(np.uint8)
    return cv2.resize(small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _skeleton_polylines(mask: np.ndarray, min_len: float) -> list[np.ndarray]:
    """把 1 像素宽的线条掩码串成有序折线。"""
    from skimage.morphology import skeletonize

    skel = skeletonize(mask > 0)
    if not skel.any():
        return []

    ys, xs = np.nonzero(skel)
    coords = {(int(x), int(y)) for x, y in zip(xs, ys, strict=False)}
    nb = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
    )

    def neighbors(p: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = p
        return [(x + dx, y + dy) for dx, dy in nb if (x + dx, y + dy) in coords]

    degree = {p: len(neighbors(p)) for p in coords}
    visited: set[tuple[int, int]] = set()
    chains: list[list[tuple[int, int]]] = []

    def walk(start: tuple[int, int], first: tuple[int, int]) -> list[tuple[int, int]]:
        chain = [start, first]
        visited.add(start)
        prev, cur = start, first
        while True:
            if cur in visited:
                break
            visited.add(cur)
            if degree.get(cur, 0) != 2:
                break
            nxt = [p for p in neighbors(cur) if p != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
        return chain

    # 先从端点/交点出发，保证线条在分叉处正确断开
    ends = [p for p in coords if degree.get(p, 0) != 2]
    for p in ends:
        if p in visited:
            continue
        for q in neighbors(p):
            if q not in visited:
                chains.append(walk(p, q))
    # 剩下的都是纯环
    for p in coords:
        if p in visited:
            continue
        ns = [q for q in neighbors(p) if q not in visited]
        if ns:
            chains.append(walk(p, ns[0]))
        else:
            visited.add(p)

    out: list[np.ndarray] = []
    for chain in chains:
        if len(chain) < 2:
            continue
        pts = np.array(chain, dtype=np.float32)
        if polyline_length(pts) >= float(min_len):
            out.append(pts)
    return out


def _median_local(a: np.ndarray, r: int) -> np.ndarray:
    """用中值滤波实现局部中值（对孤立噪点与细线稳健）。"""
    import cv2

    if r <= 0:
        return a
    return cv2.medianBlur(a, 2 * int(r) + 1)


def _box_means(stacked: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """计算每个像素处 ``(2r+1)^2`` 窗口内各通道的**和**，以及掩码像素计数。

    用 ``scipy.ndimage.uniform_filter`` 且 ``normalize=False``：它给出窗口内的
    精确求和，且支持多通道一次完成，边界按零填充处理，语义明确。
    （``cv2.boxFilter`` 在 OpenCV 5.0 上对大核会返回尺寸与归一化都不正确的结果，
    实测 kernel=19 时按 15 计算，因此不能用。）

    Args:
        stacked: ``(H, W, C)`` float64，第 0 通道为掩码、其余为"掩码 x 颜色"。
        radius: 窗口半径。

    Returns:
        ``(sums, counts)``：均为 ``(H, W, C)`` / ``(H, W)``。
    """
    from scipy.ndimage import uniform_filter

    k = 2 * radius + 1
    # 注意：uniform_filter 的 normalize=False 仍然除以核面积，因此要乘回 k*k
    # 才能得到"窗口内求和"。加上通道轴长度 1，使滤波只作用在空间两轴上。
    scale = float(k * k)
    sums = (
        uniform_filter(stacked, size=(k, k, 1), mode="constant", cval=0.0) * scale
    )
    counts = (
        uniform_filter(
            stacked[:, :, 0].astype(np.float64), size=(k, k), mode="constant", cval=0.0
        )
        * scale
    )
    return sums, counts


def extract_step_lines(
    lab: np.ndarray,
    crystal_labels: np.ndarray,
    *,
    step_delta: float = 12.0,
    min_len: float = 12.0,
    sample_radius: int = 9,
    smooth_sigma: float = 0.8,
    edge_exclude: int = 10,
    min_component_pixels: int = 60,
    max_ramp: int = 6,
) -> tuple[list[np.ndarray], dict]:
    """用"脊线两侧邻域均值色差"提取内部分界线（主方法）。

    实测结论（决定了本算法的形式）：真实厚度台阶的**总色差可以很大（ΔE 30+）**，
    但过渡带**可能宽达 10~20 像素**，因此：

    * 相邻像素梯度（1px 基线）完全测不到；
    * 过大的宽基线会把边界拉偏。

    这里采用最稳健的形式：对每个候选像素，比较它**两侧各一个
    ``sample_radius`` 像素方形邻域的均值颜色**，色差超阈值即为台阶；
    再沿**垂直于台阶的方向**做非极大值抑制，把边界收敛到过渡带的中心线上。
    对陡台阶和缓变台阶都能给出完整、单条、位置居中的分界线。

    Args:
        lab: ``(H, W, 3)`` uint8 Lab 图。
        crystal_labels: ``(H, W)`` int32 晶体标记图。
        step_delta: 色差阈值（Lab ΔE）。
        min_len: 分界线最短长度（像素）。
        sample_radius: 采样方形邻域半径（像素）。取值应略大于过渡带半宽。
        smooth_sigma: 计算前的轻度平滑（抑制 JPEG 噪声）。
        edge_exclude: 距晶体外轮廓多少像素内不算内部分界线。
        min_component_pixels: 边界连通域最小像素数。
        max_ramp: 平坦侧邻域与脊线之间允许的最大间隔（像素）。

    Returns:
        ``(chains, info)``
    """
    import cv2

    from .geometry import polyline_length

    h, w = crystal_labels.shape[:2]
    # 台阶过渡带本身就有十几像素宽，因此在降采样图上检测再放大回去几乎不丢精度，
    # 但可以显著提速。放大后线条坐标自动回到原图尺度。
    ds = 3 if min(h, w) >= 1500 else (2 if min(h, w) >= 800 else 1)
    if ds > 1:
        work_lab = np.ascontiguousarray(lab[::ds, ::ds])
        work_labels = np.ascontiguousarray(crystal_labels[::ds, ::ds])
        wh, ww = work_labels.shape[:2]
    else:
        work_lab = lab
        work_labels = crystal_labels
        wh, ww = h, w

    sr = max(2, int(round(sample_radius / ds)))

    lab_f = (
        cv2.GaussianBlur(work_lab.astype(np.float32), (0, 0), smooth_sigma, smooth_sigma)
        if smooth_sigma > 0
        else work_lab.astype(np.float32)
    )
    inside = (work_labels > 0).astype(np.float32)
    stacked = np.empty((wh, ww, lab_f.shape[2] + 1), dtype=np.float32)
    stacked[:, :, 0] = inside
    for i in range(lab_f.shape[2]):
        stacked[:, :, i + 1] = lab_f[:, :, i] * inside

    # 先在整幅工作图上算一次"窗口和"（积分图，O(1)/像素），
    # 之后每个偏移量只是对积分图做平移索引，扫描多个偏移量几乎不增加开销。
    sums, _win_area = _box_means(stacked, sr)

    # 先在整幅工作图上算一次"窗口和"（积分图，O(1)/像素），
    # 之后每个偏移量只是对结果做平移索引，因此扫描多个偏移量几乎不增加开销。
    sums, win_area = _box_means(stacked, sr)
    del win_area
    k = 2 * sr + 1

    # 预填充零边界，使"平移窗口切片"变成开销极低的视图索引。
    max_off = sr + int(max(0, max_ramp))
    pad = max_off + sr
    padded = np.zeros((wh + 2 * pad, ww + 2 * pad, sums.shape[2]), dtype=np.float64)
    padded[pad: pad + wh, pad: pad + ww, :] = sums

    def side_means(shift_y: int, shift_x: int) -> tuple[np.ndarray, np.ndarray]:
        """返回 (该侧方形邻域的平均颜色, 该侧的覆盖率)，尺寸与输入相同。

        ``shift`` 表示采样方窗中心相对当前像素的偏移；方窗半径 ``sr``。
        窗口中心对齐脊线时两侧对称、色差为 0，因此必须把窗口推到某一侧的
        平坦区，才能测到完整台阶色差。用多个偏移量可同时覆盖
        "陡台阶"与"宽过渡带"两种情形。

        覆盖率 = 窗口内属于该侧的像素数 / 窗口面积（窗口面积恒为 k*k）。
        """
        # 输出 (y, x) 要取 input (y + shift_y) 处的窗口和；
        # 又因为 sums 是"中心对齐"的，窗口和再整体平移 sr 才是该处的值。
        top = pad + shift_y
        left = pad + shift_x
        win = padded[top: top + wh, left: left + ww, :]
        cov = win[:, :, 0] / float(k * k)
        mean = win[:, :, 1:] / np.maximum(win[:, :, 0:1], 1e-3)
        return mean.astype(np.float32), cov.astype(np.float32)

    info: dict = {
        "sample_radius": int(sample_radius),
        "work_radius": sr,
        "downscale": ds,
        "threshold": float(step_delta),
        "edge_exclude": int(edge_exclude),
    }

    # 扫描多个偏移量：窗口中心对齐脊线时两侧对称、色差为 0，
    # 必须把窗口推到某一侧的平坦区才能测到完整台阶色差。
    # 用多个偏移量可以同时覆盖"陡台阶"与"宽过渡带"两种情形。
    need = 0.85
    best: np.ndarray | None = None
    best_axis = np.zeros((wh, ww), dtype=np.uint8)
    offsets = [sr + int(m) for m in range(0, int(max(0, max_ramp)) + 1)]
    for axis_name, (sy0, sx0) in (("h", (0, 1)), ("v", (1, 0))):
        axis_id = 1 if axis_name == "h" else 2
        for o in offsets:
            sy, sx = sy0 * o, sx0 * o
            m_pos, cov_pos = side_means(sy, sx)
            m_neg, cov_neg = side_means(-sy, -sx)
            d = m_pos - m_neg
            delta = np.sqrt((d * d).sum(axis=2))
            delta[(cov_pos < need) | (cov_neg < need)] = 0.0
            if best is None:
                best = delta
                best_axis[delta > 0] = axis_id
            else:
                take = delta > best
                best = np.where(take, delta, best)
                best_axis[take] = axis_id
    assert best is not None
    info["best_max"] = float(best.max()) if best.size else 0.0
    info["best_p99"] = float(np.percentile(best, 99)) if best.size else 0.0

    # 沿两个方向分别做非极大值抑制（半径需不小于偏移扫描范围），再合并
    kr = max(2, sr // 2 + int(max(0, max_ramp)))
    nms_h = cv2.dilate(best, np.ones((1, 2 * kr + 1), np.float32))
    nms_v = cv2.dilate(best, np.ones((2 * kr + 1, 1), np.float32))
    keep = (best >= float(step_delta)) & (
        ((best_axis == 1) & (best >= nms_h - 1e-4)) | ((best_axis == 2) & (best >= nms_v - 1e-4))
    )
    info["kept_pixels"] = int(keep.sum())
    if not keep.any():
        return [], info

    boundary = keep.astype(np.uint8)
    # 排除靠近晶体外轮廓的部分（那一段由"外轮廓"单独负责）。
    # 在工作分辨率上用距离变换，避免在全分辨率上做大核运算。
    dist_in = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
    boundary[dist_in < float(max(1.0, edge_exclude / ds))] = 0
    info["after_edge_exclude"] = int(boundary.sum())
    if not boundary.any():
        return [], info

    # 去孤立噪点
    min_px = max(4, int(round(min_component_pixels / (ds * ds))))
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(boundary, connectivity=8)
    clean = np.zeros_like(boundary)
    for i in range(1, n_lbl):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_px:
            clean[lbl == i] = 1
    info["components"] = int(n_lbl - 1)
    info["after_size_filter"] = int(clean.sum())
    if not clean.any():
        return [], info

    if ds > 1:
        clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_NEAREST)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    chains = _skeleton_polylines(clean, min_len=float(min_len))
    cleaned = [c for c in chains if polyline_length(c) >= float(min_len)]
    info["chains"] = len(cleaned)
    return cleaned, info


def extract_interior_lines(
    lab: np.ndarray,
    crystal_labels: np.ndarray,
    full_gray: np.ndarray,
    *,
    scale: int = 6,
    color_delta: float = 12.0,
    min_len: float = 12.0,
    compactness: float = 10.0,
    auto_otsu: bool = True,
    refine: bool = True,
    method: str = "step",
    baseline: int = 9,
) -> tuple[list[Line], dict]:
    """提取全部内部分界线。

    Args:
        lab: ``(H, W, 3)`` uint8 Lab 图（已做亮度补偿）。
        crystal_labels: ``(H, W)`` int32 晶体标记图（原图分辨率）。
        full_gray: ``(H, W)`` float32 对比度图（region 方法用于边界细化）。
        scale: 超像素目标尺度（像素），仅 ``method="region"`` 使用。
        color_delta: 色差阈值（Lab ΔE）。
        min_len: 分界线最短长度（像素）。
        compactness: SLIC 紧凑度，仅 ``method="region"`` 使用。
        auto_otsu: 是否用 Otsu 自动提高阈值（仅 region 方法）。
        refine: 是否把边界吸附到真实梯度峰值（仅 region 方法）。
        method: ``"step"``（默认，宽基线色差 + 脊线，稳健）或 ``"region"``
            （超像素中位色比较，低对比度下易碎）。
        baseline: 宽基线半径（像素），仅 ``method="step"`` 使用。

    Returns:
        ``(lines, info)``
    """
    if method == "step":
        chains, info = extract_step_lines(
            lab,
            crystal_labels,
            step_delta=float(color_delta),
            min_len=float(min_len),
            sample_radius=int(baseline),
        )
        info["method"] = "step"
        return [Line(pts, kind="inner", crystal_id=0) for pts in chains], info

    h, w = crystal_labels.shape[:2]
    ds = 4 if scale >= 3 else 2
    if min(h, w) // ds < 32:
        ds = 1
    reg, med, n_seg = build_region_labels(lab, crystal_labels, scale=scale, compactness=compactness)
    labels_small = crystal_labels[::ds, ::ds]
    pairs, _ = region_color_distances(reg, med, ds, labels_small)

    deltas = np.array(list(pairs.values()), dtype=np.float32) if pairs else np.zeros(0, np.float32)
    thr = float(color_delta)
    info: dict = {"pair_count": int(deltas.size)}
    if auto_otsu and deltas.size >= 16:
        # 把 ΔE 线性映射到 0~255 后交给 Otsu 找"噪声/真实台阶"的分界
        u8 = np.clip(deltas / 60.0 * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
        otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_delta = float(otsu) / 255.0 * 60.0
        info["otsu_delta"] = otsu_delta
        thr = max(thr, otsu_delta)
    info["threshold"] = thr
    info["n_regions"] = n_seg

    # 找出所有"色差达标"的相邻区域对
    keep_pairs = {p for p, d in pairs.items() if d >= thr}
    info["kept_pairs"] = len(keep_pairs)
    if not keep_pairs:
        return [], info

    # 构造边界掩码：区域标记变化处即为边界；再用达标区域对过滤
    base = np.int64(reg.max()) + 2
    keep_keys = np.array(sorted(keep_pairs), dtype=np.int64)
    keep_codes = keep_keys[:, 0] * base + keep_keys[:, 1]

    boundary_small = np.zeros_like(reg, dtype=np.uint8)
    # 水平相邻
    ra, rb = reg[:, :-1], reg[:, 1:]
    m = (ra != rb) & (ra > 0) & (rb > 0)
    if m.any():
        lo = np.minimum(ra[m], rb[m]).astype(np.int64)
        hi = np.maximum(ra[m], rb[m]).astype(np.int64)
        sel = np.zeros_like(m, dtype=bool)
        sel[m] = np.isin(lo * base + hi, keep_codes)
        full = np.zeros_like(reg, dtype=bool)
        full[:, :-1] = sel
        boundary_small |= full.astype(np.uint8)

    # 垂直相邻
    ra2, rb2 = reg[:-1, :], reg[1:, :]
    m2 = (ra2 != rb2) & (ra2 > 0) & (rb2 > 0)
    if m2.any():
        lo2 = np.minimum(ra2[m2], rb2[m2]).astype(np.int64)
        hi2 = np.maximum(ra2[m2], rb2[m2]).astype(np.int64)
        sel2 = np.zeros_like(m2, dtype=bool)
        sel2[m2] = np.isin(lo2 * base + hi2, keep_codes)
        full2 = np.zeros_like(reg, dtype=bool)
        full2[:-1, :] = sel2
        boundary_small |= full2.astype(np.uint8)

    if not boundary_small.any():
        return [], info

    # 放大回原图分辨率
    boundary = _upsample_mask(boundary_small, (h, w))
    # 轻度膨胀以桥接降采样造成的断点，再细化
    boundary = cv2.dilate(boundary, np.ones((3, 3), np.uint8), iterations=1)

    if refine:
        ys, xs = np.nonzero(boundary)
        step = max(1, int(xs.size // 200000) + 1)
        didx = np.arange(0, xs.size, step)
        new_x = xs[didx].copy()
        new_y = ys[didx].copy()
        for i in range(didx.size):
            x, y = int(xs[didx[i]]), int(ys[didx[i]])
            gx_strength = 0.0
            gy_strength = 0.0
            if 0 < x < w - 1:
                gx_strength = abs(float(full_gray[y, x + 1]) - float(full_gray[y, x - 1]))
            if 0 < y < h - 1:
                gy_strength = abs(float(full_gray[y + 1, x]) - float(full_gray[y - 1, x]))
            if gx_strength >= gy_strength:
                new_x[i] = _refine_x(full_gray, x, y)
            else:
                new_y[i] = _refine_y(full_gray, x, y)
        snapped = np.zeros_like(boundary)
        np.add.at(snapped, (np.clip(new_y, 0, h - 1), np.clip(new_x, 0, w - 1)), 1)
        snapped = (snapped > 0).astype(np.uint8)
        # 合并原始边界，避免细化后出现断点
        boundary = cv2.dilate(np.maximum(snapped, boundary), np.ones((3, 3), np.uint8), iterations=1)

    chains = _skeleton_polylines(boundary, min_len=float(min_len))
    lines = [Line(pts, kind="inner", crystal_id=0) for pts in chains]
    info["chains"] = len(lines)
    info["method"] = "region"
    return lines, info


def assign_crystal_ids(lines: list[Line], crystal_labels: np.ndarray) -> None:
    """给每条内部分界线标注它所属的晶体 id（取折线中点所处的晶体）。"""
    h, w = crystal_labels.shape[:2]
    for ln in lines:
        if ln.n == 0:
            continue
        mid = ln.points[ln.n // 2]
        x = int(np.clip(round(float(mid[0])), 0, w - 1))
        y = int(np.clip(round(float(mid[1])), 0, h - 1))
        ln.crystal_id = int(crystal_labels[y, x])


def crystal_median_colors(lab: np.ndarray, crystal_labels: np.ndarray) -> dict[int, np.ndarray]:
    """统计每块晶体的 Lab 中位色（调试/报告用）。"""
    out: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, np.float32))
    n = int(crystal_labels.max())
    for i in range(1, n + 1):
        m = crystal_labels == i
        if not m.any():
            continue
        vals = lab[m]
        out[i] = np.median(vals, axis=0).astype(np.float32)
    return dict(out)
