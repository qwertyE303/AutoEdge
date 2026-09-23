"""图像读写。

Windows 下 :func:`cv2.imread` 无法处理非 ASCII 路径（本项目路径含中文），
因此统一用 :func:`numpy.fromfile` + :func:`cv2.imdecode` 读写。

另外：手机/相机照片带 EXIF Orientation 时，``cv2.imdecode`` 不会自动旋转，
这里会显式读取并应用，保证"看到的方向"与"像素方向"一致。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

__all__ = ["ImageData", "imread", "imwrite", "save_image", "image_size"]


@dataclass
class ImageData:
    """读入的图像。

    Attributes:
        bgr: ``(H, W, 3)`` uint8，BGR 排列。
        alpha: ``(H, W)`` uint8 或 None。仅当源文件带 alpha 通道时存在。
        path: 源文件路径。
        orientation: 源文件 EXIF Orientation 值（1 表示正常）。
        exif: 原始 EXIF 字节，原样保留以便写出。
    """

    bgr: np.ndarray
    alpha: np.ndarray | None = None
    path: str | None = None
    orientation: int = 1
    exif: bytes | None = None

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def size(self) -> tuple[int, int]:
        """返回 ``(width, height)``。"""
        return self.width, self.height


def _read_exif(path: str) -> tuple[int, bytes | None]:
    """读取 EXIF Orientation 与原始 EXIF 字节；失败时返回 ``(1, None)``。"""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            orientation = int(exif.get(0x0112, 1) or 1)
            raw = exif.tobytes() if len(exif) else None
            return orientation, raw
    except Exception:  # noqa: BLE001 - EXIF 缺失不应影响主流程
        return 1, None


def _apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """把 EXIF Orientation 实际应用到像素上（与 PIL ``exif_transpose`` 等价）。"""
    if orientation == 1:
        return img
    # Orientation -> 逆时针旋转次数 / 是否需要镜像
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.rotate(cv2.transpose(img), cv2.ROTATE_180)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def imread(path: str, keep_alpha: bool = True) -> ImageData:
    """读取图像为 :class:`ImageData`。

    Args:
        path: 图像路径，支持非 ASCII 字符。
        keep_alpha: 是否保留 alpha 通道。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件无法解码。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"图像不存在: {path}")
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise ValueError(f"图像文件为空: {path}")
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"无法解码图像（格式不支持或文件损坏）: {path}")

    alpha = None
    if raw.ndim == 2:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        if keep_alpha:
            alpha = raw[:, :, 3].copy()
        bgr = raw[:, :, :3].copy()
    elif raw.shape[2] == 3:
        bgr = raw
    else:
        raise ValueError(f"不支持的通道数 {raw.shape[2]}: {path}")

    if bgr.dtype == np.uint16:
        bgr = (bgr.astype(np.float32) / 257.0).round().astype(np.uint8)
        if alpha is not None:
            alpha = (alpha.astype(np.float32) / 257.0).round().astype(np.uint8)

    orientation, exif = _read_exif(path)
    if orientation != 1:
        bgr = np.ascontiguousarray(_apply_orientation(bgr, orientation))
        if alpha is not None:
            alpha = np.ascontiguousarray(_apply_orientation(alpha, orientation))

    return ImageData(
        bgr=bgr, alpha=alpha, path=path, orientation=1, exif=exif
    )


def save_image(
    path: str,
    bgr: np.ndarray,
    alpha: np.ndarray | None = None,
    jpeg_quality: int = 95,
    expected_shape: tuple[int, int] | None = None,
    exif: bytes | None = None,
) -> None:
    """把图像写到磁盘。

    Args:
        path: 目标路径，扩展名决定格式（``.png`` / ``.jpg`` / ``.jpeg`` / ``.tif`` / ``.bmp``）。
        bgr: ``(H, W, 3)`` 或 ``(H, W)`` uint8。
        alpha: 可选 ``(H, W)`` uint8。仅 PNG/TIFF 支持。
        jpeg_quality: JPEG 质量 1~100。
        expected_shape: 若给出 ``(H, W)``，则强制校验尺寸一致，不一致直接报错。
        exif: 可选的 EXIF 字节（PNG/JPEG 均支持写入）。

    Raises:
        ValueError: 尺寸不符或格式不支持。
    """
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    if expected_shape is not None and (h, w) != (int(expected_shape[0]), int(expected_shape[1])):
        raise ValueError(
            f"输出尺寸与期望不一致：实际 {w}x{h}，期望 {expected_shape[1]}x{expected_shape[0]}"
        )

    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        if alpha is not None:
            # JPEG 不支持透明，按"线条叠加在底图上"的语义合成到白底
            alpha_f = (alpha.astype(np.float32) / 255.0)[..., None]
            white = np.full_like(bgr, 255, dtype=np.float32)
            bgr = (bgr.astype(np.float32) * alpha_f + white * (1 - alpha_f)).round().astype(np.uint8)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, jpeg_quality)))]
        payload = bgr
    elif ext in (".tif", ".tiff", ".bmp"):
        params = []
        payload = bgr if alpha is None else np.dstack([bgr, alpha])
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        payload = bgr if alpha is None else np.dstack([bgr, alpha])
    else:
        raise ValueError(f"不支持的输出格式: {ext}")

    ok, buf = cv2.imencode(ext if ext != ".jpeg" else ".jpg", payload, params)
    if not ok:
        raise ValueError(f"编码失败: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())

    if exif:
        try:
            with Image.open(path) as im:
                ex = im.getexif()
                ex.frombytes(exif)
                im.save(path, exif=ex, quality=jpeg_quality)
        except Exception:  # noqa: BLE001 - EXIF 写入失败不影响主流程
            pass


def image_size(path: str) -> tuple[int, int]:
    """只读取尺寸，不加载全部像素。返回 ``(width, height)``。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return int(im.width), int(im.height)
