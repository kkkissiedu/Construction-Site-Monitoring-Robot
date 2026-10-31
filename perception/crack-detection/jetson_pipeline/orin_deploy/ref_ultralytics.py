"""
ref_ultralytics.py - mask IoU/Dice of the PyTorch yolo11s-seg model on the held-out set, same metric as eval_seg.py.

Usage:
    conda activate cuda_pt
    python ref_ultralytics.py
"""

# %% Imports
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO


# %% Config
HELDOUT = "heldout"
WEIGHTS = "models/yolo11s-seg_crack_finetuned.pt"
CONFS   = [0.25, 0.1, 0.05]
IMGSZ   = 640


# %% Helpers
def load_gt(path: str, hw: tuple) -> np.ndarray:
    m = np.zeros(hw, np.uint8)
    for line in open(path).read().strip().splitlines():
        v   = np.array(line.split()[1:], np.float32).reshape(-1, 2)
        pts = (v * [hw[1], hw[0]]).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def iou_dice(a: np.ndarray, b: np.ndarray) -> tuple:
    inter, ua, ub = int((a & b).sum()), int(a.sum()), int(b.sum())
    union = ua + ub - inter
    return (1.0 if union == 0 else inter / union), (1.0 if ua + ub == 0 else 2 * inter / (ua + ub))


# %% Main
def main() -> None:
    files = sorted(glob.glob(os.path.join(HELDOUT, "images", "*.png")))
    model = YOLO(WEIGHTS)
    for conf in CONFS:
        res = []
        miss = 0
        for f in files:
            img = cv2.imread(f)
            hw  = img.shape[:2]
            gt  = load_gt(os.path.join(HELDOUT, "labels", os.path.splitext(os.path.basename(f))[0] + ".txt"), hw)
            r   = model.predict(img, imgsz=IMGSZ, conf=conf, iou=0.7, retina_masks=True, verbose=False)[0]
            pm  = np.zeros(hw, np.uint8)
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy():
                    pm |= (cv2.resize(m, (hw[1], hw[0])) > 0.5).astype(np.uint8)
            else:
                miss += 1
            res.append(iou_dice(pm, gt))
        res = np.array(res)
        print(f"ultralytics .pt  conf={conf}  IoU {res[:,0].mean():.4f}  Dice {res[:,1].mean():.4f}  miss {miss}")


# %% Entry point
if __name__ == "__main__":
    main()
