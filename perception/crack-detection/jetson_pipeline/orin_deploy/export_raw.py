"""
export_raw.py — export yolo26 seg models to ONNX with raw heads (end2end=False) so TensorRT INT8 can build them.

Usage:
    conda activate cuda_pt
    python export_raw.py
"""

# %% Imports
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO


# %% Config
MODELS_DIR = Path("models")
NAMES      = ["yolo26n-seg_crack_finetuned", "yolo26s-seg_crack_finetuned"]
IMGSZ      = 640
OPSET      = 17


# %% Main
def main() -> None:
    for name in NAMES:
        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / f"{name}.pt"
            shutil.copy(MODELS_DIR / f"{name}.pt", pt)               # export writes next to the .pt
            out = YOLO(str(pt)).export(format="onnx", imgsz=IMGSZ, opset=OPSET, simplify=True, end2end=False)
            dst = MODELS_DIR / f"{name.replace('_finetuned', '_raw')}.onnx"
            shutil.move(str(out), str(dst))
            print(f"wrote {dst}")


# %% Entry point
if __name__ == "__main__":
    main()
