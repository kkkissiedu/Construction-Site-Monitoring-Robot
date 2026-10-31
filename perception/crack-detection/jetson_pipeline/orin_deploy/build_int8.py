#!/usr/bin/env python3
"""
build_int8.py - build a TensorRT INT8 engine (and calibration cache) for a YOLO-seg ONNX on the Jetson.

Usage:
    python3 build_int8.py MODEL.onnx OUT.engine IMGSZ [N_IMAGES] [PIN]   (PIN='proto' keeps the mask-prototype branch in FP16)

Uses only JetPack's own TensorRT Python API, NumPy and OpenCV. CUDA memory is handled through ctypes (libcudart).
Calibration images come from ~/crackseg/calib (letterboxed to IMGSZ, RGB, 0-1, CHW).
INT8 is allowed to fall back to FP16 per layer (mixed precision).
"""
import ctypes
import glob
import os
import sys
import time

import cv2
import numpy as np
import tensorrt as trt

ONNX_PATH  = sys.argv[1]
ENGINE_OUT = sys.argv[2]
IMGSZ      = int(sys.argv[3])
N_IMAGES   = int(sys.argv[4]) if len(sys.argv) > 4 else 200
PIN        = sys.argv[5] if len(sys.argv) > 5 else ""   # "proto" = keep the mask-prototype branch in FP16
CALIB_DIR  = os.path.expanduser("~/crackseg/calib")
CACHE_PATH = os.path.splitext(ENGINE_OUT)[0] + ".calib.cache"
WORKSPACE  = 2 << 30


def load_cudart() -> ctypes.CDLL:
    for name in ("libcudart.so.12", "libcudart.so", "/usr/local/cuda/lib64/libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("libcudart not found")


cudart = load_cudart()
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def cuda_ok(code: int) -> None:
    if code != 0:
        raise RuntimeError(f"CUDA error {code}")


def preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, files: list) -> None:
        super().__init__()
        self.files  = files
        self.index  = 0
        self.nbytes = 1 * 3 * IMGSZ * IMGSZ * 4
        self.device = ctypes.c_void_p()
        cuda_ok(cudart.cudaMalloc(ctypes.byref(self.device), self.nbytes))

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.index >= len(self.files):
            return None
        x = preprocess(self.files[self.index])
        self.index += 1
        if self.index % 25 == 0:
            print(f"  calibration image {self.index}/{len(self.files)}", flush=True)
        cuda_ok(cudart.cudaMemcpy(self.device, x.ctypes.data_as(ctypes.c_void_p), self.nbytes, 1))
        return [int(self.device.value)]

    def read_calibration_cache(self):
        if os.path.exists(CACHE_PATH):
            print("  reusing calibration cache", CACHE_PATH)
            with open(CACHE_PATH, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache) -> None:
        with open(CACHE_PATH, "wb") as f:
            f.write(cache)
        print("  wrote calibration cache", CACHE_PATH, flush=True)


def main() -> int:
    files = sorted(glob.glob(os.path.join(CALIB_DIR, "*")))[:N_IMAGES]
    print(f"model {os.path.basename(ONNX_PATH)} | imgsz {IMGSZ} | {len(files)} calibration images | TensorRT {trt.__version__}")
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser  = trt.OnnxParser(network, logger)
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            return 2
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    if PIN:
        pinned = []
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            floaty = all(layer.get_output(j).dtype in (trt.float32, trt.float16) for j in range(layer.num_outputs))
            if PIN in layer.name and layer.type not in (trt.LayerType.CONSTANT, trt.LayerType.CAST) and floaty:
                layer.precision = trt.float16
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float16)
                pinned.append(layer.name)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        print(f"  pinned {len(pinned)} layers containing '{PIN}' to FP16, e.g. {pinned[:3]}")
        for k in range(network.num_outputs):
            network.get_output(k).dtype = trt.float32
    config.int8_calibrator = Calibrator(files)
    t0   = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("  BUILD FAILED (see TensorRT messages above)")
        return 3
    with open(ENGINE_OUT, "wb") as f:
        f.write(plan)
    print(f"  OK: {ENGINE_OUT} ({os.path.getsize(ENGINE_OUT) / 1e6:.1f} MB) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())