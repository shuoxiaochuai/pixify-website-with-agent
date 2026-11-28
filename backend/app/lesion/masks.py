"""真实标注（验证集 GT）的读取、对齐与 Dice。

标注多为 IDRiD 的 .tif，也可能是手绘 PNG/JPG。入库前统一转成灰度 PNG；
比较时走与推理完全相同的 Resize(642) → CenterCrop(640) 链，保证像素一一对应。
"""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.services.images import MAX_FILE_BYTES, MAX_PIXELS

SIZE = 640  # CAUNet.original_size，forward 里的插值尺寸写死了 640
PAD = 642  # 验证集 Resize(642) → CenterCrop(640)


class MaskRejected(Exception):
    """不能作为真实标注使用的文件。"""


def normalize(data: bytes) -> bytes:
    """任意可解码图片 → 灰度 PNG 字节，供入库与后续比较。"""
    if not data:
        raise MaskRejected("标注文件为空")
    if len(data) > MAX_FILE_BYTES:
        raise MaskRejected(f"标注文件超过 {MAX_FILE_BYTES // 1024 // 1024} MB 上限")
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_PIXELS:
                raise MaskRejected("标注像素总量过大")
            image.load()
            gray = image.convert("L")
    except MaskRejected:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise MaskRejected("标注文件已损坏或不是可读取的图片") from exc

    buffer = io.BytesIO()
    gray.save(buffer, format="PNG")
    return buffer.getvalue()


def align(data: bytes, size: tuple[int, int]) -> np.ndarray:
    """标注对齐到模型输入尺寸，返回 640×640 的布尔掩码。"""
    width, height = size
    with Image.open(io.BytesIO(data)) as image:
        gray = image.convert("L")
        if gray.size != size:
            # 标注与原图尺寸不一致时先缩放到原图，再走同一条裁剪链
            gray = gray.resize((width, height), Image.NEAREST)

    scale = PAD / min(width, height)
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    gray = gray.resize(resized, Image.NEAREST)
    left = max(0, int(round((resized[0] - SIZE) / 2)))
    top = max(0, int(round((resized[1] - SIZE) / 2)))
    gray = gray.crop((left, top, left + SIZE, top + SIZE))
    return np.asarray(gray) > 0


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Dice 系数。两边都为空时记 0，避免空预测在阈值遍历里刷高分。"""
    total = int(pred.sum()) + int(truth.sum())
    if total == 0:
        return 0.0
    overlap = np.logical_and(pred, truth).sum()
    return 2.0 * float(overlap) / total
