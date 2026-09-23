"""预处理：光照归一化与对比度补偿。

显微镜照片有两个必须处理的特性：

1. **暗角与光照不均**：同一张图里基底亮度可能相差数倍。
   做法是用大核形态学开运算估计背景（亮晶体被抹掉，只剩基底），
   再把图像转成"相对背景的对比度"，这样黑基底照片和中灰基底照片能共用一套阈值。
2. **新旧照片整体亮度差异**：通过可选的亮度/对比度/伽马补偿解决。

所有输出都是 float32 且与输入同尺寸，不改变分辨率。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["PreprocessResult", "estimate_background", "prepare"]


@dataclass
class PreprocessResult:
    """预处理结果。

    Attributes:
        gray: ``(H, W)`` float32，光照归一化后的灰度对比度图（背景≈0，晶体明显为正）。
        lab: ``(H, W, 3)`` uint8，已补偿后的 Lab 图，用于计算区域色差。
        background: ``(H, W)`` float32，估计出的背景灰度。
        noise: float，背景噪声水平（对比度图的稳健估计）。
        background_level: float，背景绝对灰度中位数（用于判断新旧照片）。
        auto_sigma: float，本次实际使用的背景估计核半径（像素）。
    """

    gray: np.ndarray
    lab: np.ndarray
    background: np.ndarray
    noise: float
    background_level: float
    auto_sigma: float


def _auto_sigma(height: int, width: int) -> float:
    """自动背景核：取短边的约 1/12，并夹在合理范围内。

    核必须显著大于最大晶体尺寸才能把晶体完全抹掉，同时又要能跟随暗角的缓变。
    """
    return float(np.clip(min(height, width) / 12.0, 24.0, 512.0))


def estimate_background(gray: np.ndarray, sigma: float) -> np.ndarray:
    """估计缓变的光照背景。

    做法（关键性能点：**不在全分辨率上做大核形态学**，那会慢到几分钟）：

    1. ``INTER_AREA`` 降采样 —— 相当于盒式低通，亮晶体被邻域平均掉一部分；
    2. 用 ``p`` 分位秩滤波把剩下的亮晶体彻底压掉（基底占绝大多数像素）；
    3. 高斯平滑后 ``INTER_CUBIC`` 升回原尺寸。

    背景本身是缓变场，上述降采样-升采样不会损失有效信息。
    """
    h, w = gray.shape
    target = float(np.clip(sigma, 24.0, 512.0))
    ds = int(np.clip(round(target / 6.0), 2, 16))
    small = cv2.resize(gray, (max(2, w // ds), max(2, h // ds)), interpolation=cv2.INTER_AREA)
    # 局部最小值 + 中值：把残余的亮晶体压掉（基底占绝大多数像素，故中值≈基底水平）
    k = int(np.clip(round(target / ds) | 1, 3, 15))
    small = cv2.medianBlur(small, k)
    bg_small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, target / ds / 2.0))
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.maximum(bg, 1.0).astype(np.float32)


def _apply_tone(gray: np.ndarray, brightness: int, contrast: float, gamma: float) -> np.ndarray:
    if brightness:
        gray = gray + float(brightness)
    if contrast != 1.0:
        mean = float(gray.mean())
        gray = (gray - mean) * float(contrast) + mean
    if gamma != 1.0:
        g = np.clip(gray, 0.0, 255.0) / 255.0
        gray = (np.power(g, 1.0 / float(gamma)) * 255.0).astype(np.float32)
    return gray


def prepare(
    bgr: np.ndarray,
    *,
    normalize_illumination: bool = True,
    background_sigma: float = 0.0,
    brightness: int = 0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> PreprocessResult:
    """把 BGR 图像转换成"对比度图 + Lab 图"。

    Args:
        bgr: ``(H, W, 3)`` uint8。
        normalize_illumination: 是否做光照归一化（强烈建议开启）。
        background_sigma: 背景估计核半径（像素）；``0`` 表示自动。
        brightness: 手动亮度补偿，-100~100。
        contrast: 手动对比度，0.2~4.0。
        gamma: 伽马，0.2~4.0。

    Returns:
        :class:`PreprocessResult`
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"prepare 需要 (H, W, 3) 的 BGR 图像，实际 {bgr.shape}")

    h, w = bgr.shape[:2]
    sigma = float(background_sigma) if background_sigma and background_sigma > 0 else _auto_sigma(h, w)

    work = bgr
    if brightness or contrast != 1.0 or gamma != 1.0:
        work = _apply_tone(bgr.astype(np.float32), brightness, contrast, gamma)
        work = np.clip(work, 0, 255).astype(np.uint8)

    gray_u8 = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    if normalize_illumination:
        bg = estimate_background(gray_u8, sigma)
        # 相对对比度：0 表示与背景一致，正值表示比背景亮
        bg_f = np.maximum(bg.astype(np.float32), 1.0)
        gray = gray_u8.astype(np.float32) / bg_f * 255.0 - 255.0
    else:
        bg = np.full_like(gray_u8, int(np.median(gray_u8)), dtype=np.uint8)
        gray = gray_u8.astype(np.float32) - float(np.median(gray_u8))

    # 背景噪声：取对比度图的稳健标准差（1.4826 * MAD）
    mad = float(np.median(np.abs(gray - np.median(gray))))
    noise = max(1e-3, 1.4826 * mad)

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    background_level = float(np.median(gray_u8.astype(np.float32)))

    return PreprocessResult(
        gray=gray.astype(np.float32),
        lab=lab,
        background=bg.astype(np.float32),
        noise=noise,
        background_level=background_level,
        auto_sigma=float(sigma),
    )
