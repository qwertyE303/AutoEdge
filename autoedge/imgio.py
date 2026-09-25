"""图像读写。

Windows 下 :func:`cv2.imread` 无法处理非 ASCII 路径（本项目路径含中文），
因此统一用 :func:`numpy.fromfile` + :func:`cv2.imdecode` 读写。

另外：手机/相机照片带 EXIF Orientation 时，``cv2.imdecode`` 不会自动旋转，
这里会显式读取并应用，保证"看到的方向"与"像素方向"一致。

导出 DPI 也在这里补：``cv2.imencode`` **不写任何分辨率元数据**（实测 PNG 里
没有 pHYs；JPEG 的 JFIF density 是 ``units=0, 1×1``，等于没有），于是
Photoshop 打开导出图一律按 72 ppi 处理，物理尺寸就错了。这里在**编码完成后
就地补写元数据**——不动 IDAT/DCT 数据，因此像素与不补时逐字节一致。
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

__all__ = ["ImageData", "imread", "imwrite", "save_image", "image_size"]

#: PNG 文件签名
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
#: 1 英寸 = 25.4 mm（PNG 的 pHYs 用"像素/米"，所以要换算）
_MM_PER_INCH = 25.4


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


def _dpi_to_ppm(dpi: float) -> int:
    """dpi → 像素/米（PNG ``pHYs`` 的单位）。"""
    return int(round(float(dpi) * 1000.0 / _MM_PER_INCH))


def _png_with_dpi(raw: bytes, dpi: tuple[float, float]) -> bytes:
    """在 ``IHDR`` 之后、``IDAT`` 之前插入 ``pHYs``（已存在则替换）。

    写法与 **Photoshop 自己导出的 PNG 完全一致**（实测手描参考图
    ``Gr05.png`` 等 8 张的 chunk 就是 ``IHDR → pHYs(X=11811, Y=11811, unit=1)
    → IDAT``，且不含 eXIf/tEXt）：单位用米、紧跟 IHDR、不写 EXIF。

    只重新拼装 chunk 头部，``IDAT`` 里的压缩数据原样搬运，所以**像素零改动**。
    结构不认识（没有 IHDR）时原样返回，绝不写坏文件。
    """
    if not raw.startswith(_PNG_SIG):
        return raw
    data = struct.pack(">IIB", _dpi_to_ppm(dpi[0]), _dpi_to_ppm(dpi[1]), 1)
    chunk = (
        struct.pack(">I", len(data))
        + b"pHYs"
        + data
        + struct.pack(">I", zlib.crc32(b"pHYs" + data) & 0xFFFFFFFF)
    )

    out = bytearray(_PNG_SIG)
    i = len(_PNG_SIG)
    inserted = False
    while i + 8 <= len(raw):
        length = struct.unpack(">I", raw[i : i + 4])[0]
        kind = raw[i + 4 : i + 8]
        end = i + 12 + length
        if kind == b"IHDR":
            out += raw[i:end]
            out += chunk
            inserted = True
        elif kind != b"pHYs":
            out += raw[i:end]
        i = end
        if kind == b"IEND":
            break
    return bytes(out) if inserted else raw


def _jpeg_with_dpi(raw: bytes, dpi: tuple[float, float]) -> bytes:
    """改写 JFIF ``APP0`` 里的密度（``units=1`` 英寸）；没有 APP0 就补一个。

    只改 4 个字节（units/Xdensity/Ydensity），**DCT 数据一个字节都不动**，
    所以不会产生"二次压缩"的画质损失。cv2 写出的 JPEG 实测是
    ``units=0, Xdensity=Ydensity=1``，等于"没有 DPI"，Photoshop 会退回 72。
    """
    if len(raw) < 4 or raw[0:2] != b"\xff\xd8":
        return raw
    x = max(1, min(65535, int(round(dpi[0]))))
    y = max(1, min(65535, int(round(dpi[1]))))

    i = 2
    while i + 4 <= len(raw):
        if raw[i] != 0xFF:
            break
        marker = raw[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # 无长度字段的标记
            i += 2
            continue
        if marker == 0xDA:  # SOS：图像数据开始，后面不可能再有 APP0
            break
        seg_len = struct.unpack(">H", raw[i + 2 : i + 4])[0]
        # JFIF APP0：FF E0 | len | 'JFIF\0' | ver(2) | units(1) | Xd(2) | Yd(2) | …
        if marker == 0xE0 and raw[i + 4 : i + 9] == b"JFIF\x00" and seg_len >= 16:
            buf = bytearray(raw)
            buf[i + 11] = 1  # units = 1（每英寸）
            buf[i + 12 : i + 14] = struct.pack(">H", x)
            buf[i + 14 : i + 16] = struct.pack(">H", y)
            return bytes(buf)
        i += 2 + seg_len

    app0 = (
        b"\xff\xe0"
        + struct.pack(">H", 16)
        + b"JFIF\x00"
        + b"\x01\x01"
        + b"\x01"
        + struct.pack(">H", x)
        + struct.pack(">H", y)
        + b"\x00\x00"
    )
    return raw[:2] + app0 + raw[2:]


def save_image(
    path: str,
    bgr: np.ndarray,
    alpha: np.ndarray | None = None,
    jpeg_quality: int = 95,
    expected_shape: tuple[int, int] | None = None,
    exif: bytes | None = None,
    dpi: tuple[float, float] | None = None,
) -> None:
    """把图像写到磁盘。

    Args:
        path: 目标路径，扩展名决定格式（``.png`` / ``.jpg`` / ``.jpeg`` / ``.tif`` / ``.bmp``）。
        bgr: ``(H, W, 3)`` 或 ``(H, W)`` uint8。
        alpha: 可选 ``(H, W)`` uint8。仅 PNG/TIFF 支持。
        jpeg_quality: JPEG 质量 1~100。
        expected_shape: 若给出 ``(H, W)``，则强制校验尺寸一致，不一致直接报错。
        exif: 可选的 EXIF 字节（PNG/JPEG 均支持写入）。
        dpi: 可选的 ``(水平, 垂直)`` DPI，只写进文件元数据（PNG 的 ``pHYs``、
            JPEG 的 JFIF density），**不改动任何像素**。TIFF/BMP 目前不支持
            （cv2 写 TIFF 时没有分辨率标签，补写要重排 IFD，收益不值风险）。

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

    # DPI 必须放在**最后**：上面那段 EXIF 是用 Pillow 把整个文件重存一遍的，
    # 先补 DPI 会被它冲掉（那条分支目前没有调用方在用，但顺序不能反）。
    if dpi is not None and ext in (".png", ".jpg", ".jpeg"):
        with open(path, "rb") as f:
            raw = f.read()
        patched = _png_with_dpi(raw, dpi) if ext == ".png" else _jpeg_with_dpi(raw, dpi)
        if patched != raw:
            with open(path, "wb") as f:
                f.write(patched)


def image_size(path: str) -> tuple[int, int]:
    """只读取尺寸，不加载全部像素。返回 ``(width, height)``。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return int(im.width), int(im.height)
