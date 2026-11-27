"""CAUNet 推理：预处理、算力判定、阈值化与着色。

预处理逐条对齐 DEAR-UNet 的验证链（DataSet.load_train_val_data + Transforms）：
PIL 短边 Resize 642（双线性）→ CenterCrop 640 → CLAHE(4.0, 8x8) → 只取绿通道 →
float32、范围 0~255（训练脚本对 uint8 张量直接 .float()，没有除 255）。
模型最后一层自带 Sigmoid，输出即概率；官方可视化脚本用 > 0.8 出二值图。

顶层不 import torch：无卡实例上 app 与 worker 的启动、其他工具都不该被 CUDA 库拖累。
"""

import io
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings

TYPES: tuple[str, ...] = ("EX", "HE", "MA", "SE")
LABELS: dict[str, str] = {
    "EX": "出血点",
    "HE": "硬性渗出",
    "MA": "微血管瘤",
    "SE": "软性渗出",
}
# 四类用互不混淆的颜色：红 / 橙 / 青 / 绿
COLORS: dict[str, tuple[int, int, int]] = {
    "EX": (235, 64, 52),
    "HE": (255, 168, 26),
    "MA": (38, 198, 218),
    "SE": (67, 210, 92),
}

SIZE = 640  # CAUNet.original_size，forward 里的插值尺寸写死了 640
PAD = 642  # 验证集 Resize(642) → CenterCrop(640)
CLAHE_CLIP = 4.0  # albumentations CLAHE 的默认值
CLAHE_TILE = (8, 8)

NO_GPU = "当前实例未开启 GPU，病灶分割需要显卡算力；开启后重试"
NO_RUNTIME = "当前实例未安装推理运行时（torch），病灶分割暂不可用"
NO_MODEL = "当前实例未配置病灶分割权重"

_lock = threading.Lock()
_models: dict[str, object] = {}


class Unavailable(Exception):
    """可用于展示的不可用原因。算力、运行时、权重缺失都走这里。"""


def model_paths() -> dict[str, Path]:
    settings = get_settings()
    configured = {
        "EX": settings.lesion_model_ex,
        "HE": settings.lesion_model_he,
        "MA": settings.lesion_model_ma,
        "SE": settings.lesion_model_se,
    }
    return {name: Path(raw).expanduser() for name, raw in configured.items() if raw}


def device_label() -> str:
    return "cpu" if get_settings().lesion_device == "cpu" else "cuda"


def _import_torch(*, required: bool = True):
    try:
        import torch
    except Exception as exc:  # ImportError 之外，CUDA 动态库缺失也会在这里炸
        if required:
            raise Unavailable(NO_RUNTIME) from exc
        return None
    return torch


def available() -> tuple[bool, str | None]:
    """能否执行分割。前端预检与执行前的硬门共用同一判断。"""
    paths = model_paths()
    missing = [name for name in TYPES if name not in paths or not paths[name].is_file()]
    if missing:
        return False, f"{NO_MODEL}（缺少 {'、'.join(missing)}）"
    torch = _import_torch(required=False)
    if torch is None:
        return False, NO_RUNTIME
    if get_settings().lesion_device != "cpu" and not torch.cuda.is_available():
        return False, NO_GPU
    return True, None


def status() -> dict:
    ok, reason = available()
    found = model_paths()
    return {
        "available": ok,
        "reason": reason,
        "device": device_label(),
        "models": {name: name in found and found[name].is_file() for name in TYPES},
    }


def prepare(data: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """图片字节 → 模型输入 (640, 640) float32，并返回原图宽高供还原。"""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = image.size
    if width < 1 or height < 1:
        raise Unavailable("图片尺寸异常，无法分割")

    scale = PAD / min(width, height)
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = image.resize(resized, Image.BILINEAR)
    left = max(0, int(round((resized[0] - SIZE) / 2)))
    top = max(0, int(round((resized[1] - SIZE) / 2)))
    image = image.crop((left, top, left + SIZE, top + SIZE))

    array = _clahe(np.asarray(image, dtype=np.uint8))
    # 训练时 green=True 只喂绿通道，且不做 /255 归一化
    green = np.ascontiguousarray(array[:, :, 1], dtype=np.float32)
    return green, (width, height)


def _clahe(rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise Unavailable(NO_RUNTIME) from exc
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(
        lab[:, :, 0]
    )
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def restore(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """640 掩码放回 642 画布的原裁剪位，再最近邻放回原图尺寸。"""
    width, height = size
    scale = PAD / min(width, height)
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    left = max(0, int(round((resized[0] - SIZE) / 2)))
    top = max(0, int(round((resized[1] - SIZE) / 2)))

    canvas = np.zeros((resized[1], resized[0]), dtype=np.uint8)
    canvas[top : top + SIZE, left : left + SIZE] = mask.astype(np.uint8)
    grown = Image.fromarray(canvas, "L").resize((width, height), Image.NEAREST)
    return np.asarray(grown) > 0


def _model(name: str, torch, device: str):
    model = _models.get(name)
    if model is None:
        from app.lesion.caunet import CAUNet

        paths = model_paths()
        if name not in paths:
            raise Unavailable(f"{NO_MODEL}（缺少 {name}）")
        checkpoint = torch.load(paths[name], map_location="cpu", weights_only=False)
        state = (
            checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        # 兼容旧代码误用 DataParallel 存下来的 module. 前缀；当前权重没有
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model = CAUNet()
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device)
        _models[name] = model
    return model


def predict(data: bytes, types: Sequence[str], threshold: float) -> dict[str, np.ndarray]:
    """逐类推理并返回原图尺寸的二值掩码。显存有限，四类串行。"""
    torch = _import_torch()
    device = device_label()
    if device == "cuda" and not torch.cuda.is_available():
        raise Unavailable(NO_GPU)

    sample, size = prepare(data)
    masks: dict[str, np.ndarray] = {}
    with _lock:
        with torch.inference_mode():
            batch = torch.from_numpy(sample)[None, None].to(device)
            for name in types:
                output = _model(name, torch, device)(batch)
                prob = output[0] if isinstance(output, tuple) else output
                prob = prob[0, 0].detach().to("cpu").float().numpy()
                masks[name] = restore(prob >= threshold, size)
        if device == "cuda":
            torch.cuda.empty_cache()
    return masks


def overlay(mask: np.ndarray, color: tuple[int, int, int]) -> bytes:
    """二值掩码 → 同尺寸 RGBA PNG，只有命中像素有颜色，其余全透明。"""
    height, width = mask.shape
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[mask] = (*color, 255)
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()
