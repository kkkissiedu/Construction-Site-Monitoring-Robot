# %%
"""bench_engine.py - time a TensorRT engine on the Jetson via the pycuda runtime.

Loads a serialised ``.engine``, runs a real image through it with device
buffers, and reports both the pure inference cycle and the full end-to-end path
(preprocess + infer) so the number reflects what a deployed pipeline sees, not
just the kernel.

Usage:
    OPENBLAS_CORETYPE=ARMV8 python3 bench_engine.py \
        --engine models/yolo11n-seg_fp16.engine --image sample.jpg
"""


# %% Imports
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 - creates the CUDA context

# %% Config
LETTERBOX_FILL = 114
NORM_SCALE     = 1.0 / 255.0


# %%
def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio onto a square padded canvas.

    Args:
        image: HxWx3 BGR uint8 image.
        size: Target square side length.

    Returns:
        size x size x 3 BGR uint8 canvas.
    """
    height, width = image.shape[:2]
    scale = min(size / float(height), size / float(width))
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), LETTERBOX_FILL, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


# %%
def preprocess(image: np.ndarray, size: int, dtype: np.dtype) -> np.ndarray:
    """Letterbox, BGR->RGB, HWC->CHW, scale to [0,1], add batch dim.

    Args:
        image: HxWx3 BGR uint8 image.
        size: Square network input size.
        dtype: Target numpy dtype matching the engine's input binding.

    Returns:
        Contiguous 1x3xsize x size array in ``dtype``.
    """
    boxed = letterbox(image, size)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) * NORM_SCALE
    return np.ascontiguousarray(chw[None], dtype=dtype)


# %%
def load_engine(path: Path, logger: trt.Logger) -> trt.ICudaEngine:
    """Deserialise a TensorRT engine from disk.

    Args:
        path: Path to the ``.engine`` file.
        logger: TensorRT logger.

    Returns:
        The deserialised engine.

    Raises:
        RuntimeError: If deserialisation fails.
    """
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {path}")
    return engine


# %%
def main(config: dict) -> int:
    """Benchmark one engine on one image.

    Args:
        config: Runtime configuration.

    Returns:
        Process exit code.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(Path(config["engine"]), logger)
    context = engine.create_execution_context()

    # Discover bindings (TRT 8.2: num_bindings / get_binding_*).
    inputs, outputs = [], []
    print("=== bindings ===")
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = tuple(engine.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(i))
        is_in = engine.binding_is_input(i)
        print(f"  [{i}] {'IN ' if is_in else 'OUT'} {name} {shape} {np.dtype(dtype).name}")
        dims = [d if d > 0 else (1 if j == 0 else (3 if j == 1 else config["imgsz"]))
                for j, d in enumerate(shape)]
        nbytes = int(np.prod(dims)) * np.dtype(dtype).itemsize
        entry = {
            "index": i, "name": name, "dims": dims, "dtype": dtype,
            "dev": cuda.mem_alloc(nbytes),
            "host": cuda.pagelocked_empty(int(np.prod(dims)), dtype),
        }
        (inputs if is_in else outputs).append(entry)

    # bindings[] must be ordered by binding index.
    bindings = [0] * engine.num_bindings
    for entry in inputs + outputs:
        bindings[entry["index"]] = int(entry["dev"])

    in_entry = inputs[0]
    imgsz = in_entry["dims"][-1]

    image = cv2.imread(config["image"])
    if image is None:
        print(f"[error] could not read image {config['image']}")
        return 1
    tensor = preprocess(image, imgsz, in_entry["dtype"])
    print(f"=== input {image.shape[1]}x{image.shape[0]} -> {tensor.shape} {tensor.dtype} ===")

    stream = cuda.Stream()

    def infer_only() -> None:
        """One H2D -> execute -> D2H cycle on the preprocessed tensor."""
        in_entry["host"][:] = tensor.ravel()
        cuda.memcpy_htod_async(in_entry["dev"], in_entry["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["dev"], stream)
        stream.synchronize()

    for _ in range(config["warmup"]):
        infer_only()

    start = time.perf_counter()
    for _ in range(config["runs"]):
        infer_only()
    cycle_s = (time.perf_counter() - start) / config["runs"]

    start = time.perf_counter()
    for _ in range(config["runs"]):
        t = preprocess(image, imgsz, in_entry["dtype"])
        in_entry["host"][:] = t.ravel()
        cuda.memcpy_htod_async(in_entry["dev"], in_entry["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["dev"], stream)
        stream.synchronize()
    e2e_s = (time.perf_counter() - start) / config["runs"]

    print("=== results ===")
    print(f"  infer cycle (H2D+GPU+D2H) : {cycle_s*1000:7.2f} ms  ->  {1.0/cycle_s:6.2f} FPS")
    print(f"  end-to-end (preproc+infer): {e2e_s*1000:7.2f} ms  ->  {1.0/e2e_s:6.2f} FPS")
    print(f"  runs={config['runs']} warmup={config['warmup']} imgsz={imgsz}")
    for out in outputs:
        arr = np.asarray(out["host"]).reshape(out["dims"])
        print(f"  out {out['name']} {tuple(out['dims'])} range [{arr.min():.3f}, {arr.max():.3f}]")
    return 0


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a TensorRT engine on the Jetson.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    raise SystemExit(main(vars(args)))
