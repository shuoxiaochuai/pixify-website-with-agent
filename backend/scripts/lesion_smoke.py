"""病灶分割冒烟：真实权重 + GPU + 与 IDRiD 真实掩码对比。

用法（服务器 backend 目录下）：
    .venv/bin/python scripts/lesion_smoke.py [图片] [EX 掩码]

不带参数用 IDRiD_55 的样例；掩码只覆盖 EX，其余三类只打像素数。阈值取 .env 里的
LESION_THRESHOLD，判断服务端配置是否真的生效。
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings
from app.lesion import engine

IMG = "/root/rivermind-data/Segmentation_dataset3/test_images/IDRiD_55.jpg"
GT = "/root/rivermind-data/Segmentation_dataset3/test_masks/EX/IDRiD_55_EX.tif"


def main() -> None:
    image = Path(sys.argv[1] if len(sys.argv) > 1 else IMG)
    gt_path = Path(sys.argv[2] if len(sys.argv) > 2 else GT)
    threshold = get_settings().lesion_threshold

    print("STATUS", engine.status(), flush=True)
    print(f"threshold={threshold} image={image}", flush=True)

    data = image.read_bytes()
    t0 = time.time()
    masks = engine.predict(data, list(engine.TYPES), threshold)
    first = time.time() - t0

    t1 = time.time()
    engine.predict(data, ["EX"], threshold)
    warm = time.time() - t1

    print(f"elapsed_first4={first:.2f}s warm_ex={warm:.2f}s")
    print("pred_pixels", {name: int(mask.sum()) for name, mask in masks.items()})

    if gt_path.is_file():
        gt = np.asarray(Image.open(gt_path).convert("L")) > 0
        pred = masks["EX"]
        if pred.shape != gt.shape:
            resized = Image.fromarray(gt.astype(np.uint8) * 255, "L").resize(
                (pred.shape[1], pred.shape[0]), Image.NEAREST
            )
            gt = np.asarray(resized) > 0
        dice = 2.0 * (pred & gt).sum() / max(1, int(pred.sum()) + int(gt.sum()))
        print(f"gt_pixels={int(gt.sum())} dice_ex={dice:.4f}")


if __name__ == "__main__":
    main()
