"""cam_infer.py - live IMX219 camera crack segmentation on the Jetson Nano.

Captures a short burst from the CSI camera via GStreamer/argus (the Nano's
pip OpenCV has no GStreamer support, so capture goes through gst-launch to disk
rather than cv2.VideoCapture), runs the TensorRT engine on the auto-exposure-
settled frames, overlays the crack masks, and writes an annotated still plus an
animated GIF for the portfolio.

3.6-safe. Reuses the engine/decode helpers from infer_seg.py (same directory).

Usage:
    OPENBLAS_CORETYPE=ARMV8 python3 cam_infer.py \
        --engine models/yolo11n-seg_416_fp16.engine --frames 45 --warmup 15
"""

# %% Imports
import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 - creates the CUDA context

from infer_seg import load_engine, allocate, preprocess, decode, overlay

# %% Config
CAP_WIDTH   = 1280
CAP_HEIGHT  = 720
CAP_DIR     = "/tmp/cam_infer"
GIF_MS      = 120          # per-frame duration in the output GIF


# %%
def capture_burst(n_frames, out_dir):
    """Capture a burst of JPEG frames from the IMX219 via GStreamer.

    Args:
        n_frames: Number of frames to capture.
        out_dir: Directory the frames are written into.

    Returns:
        Sorted list of captured frame paths.

    Raises:
        RuntimeError: If capture produced no frames.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("cap_*.jpg"):
        old.unlink()
    pipeline = (
        "gst-launch-1.0 -e nvarguscamerasrc num-buffers={n} ! "
        "'video/x-raw(memory:NVMM),width={w},height={h},framerate=30/1' ! "
        "nvvidconv ! 'video/x-raw,format=I420' ! jpegenc ! "
        "multifilesink location={d}/cap_%03d.jpg"
    ).format(n=n_frames, w=CAP_WIDTH, h=CAP_HEIGHT, d=out_dir)
    subprocess.run(
        ["bash", "-c", pipeline],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    frames = sorted(out.glob("cap_*.jpg"))
    if not frames:
        raise RuntimeError("camera capture produced no frames")
    return frames


# %%
def build_runner(engine_path):
    """Load the engine and allocate buffers once for reuse across frames.

    Args:
        engine_path: Path to the ``.engine`` file.

    Returns:
        ``(context, inputs, outputs, bindings, stream, imgsz)``.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(engine_path, logger)
    context = engine.create_execution_context()
    inputs, outputs, bindings = allocate(engine, 640)
    stream = cuda.Stream()
    imgsz = inputs[0]["dims"][-1]
    return context, inputs, outputs, bindings, stream, imgsz


# %%
def infer_frame(image, runner):
    """Run one frame through the engine and return the annotated overlay.

    Args:
        image: HxWx3 BGR uint8 frame.
        runner: Tuple from :func:`build_runner`.

    Returns:
        ``(annotated_bgr, n_detections, crack_fraction)``.
    """
    context, inputs, outputs, bindings, stream, imgsz = runner
    in_entry = inputs[0]
    tensor, scale, pad_x, pad_y = preprocess(image, imgsz, in_entry["dtype"])
    in_entry["host"][:] = tensor.ravel()
    cuda.memcpy_htod_async(in_entry["dev"], in_entry["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["dev"], stream)
    stream.synchronize()
    mask, ndet, boxes = decode(
        outputs, scale, pad_x, pad_y, image.shape[1], image.shape[0], imgsz,
    )
    return overlay(image, mask, boxes), ndet, float((mask > 0).mean())


# %%
def save_gif(frames_bgr, path, duration_ms):
    """Write a list of BGR frames as an animated GIF via PIL.

    Args:
        frames_bgr: List of HxWx3 BGR uint8 frames.
        path: Output GIF path.
        duration_ms: Per-frame duration in milliseconds.
    """
    from PIL import Image
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    if not imgs:
        return
    imgs[0].save(
        path, save_all=True, append_images=imgs[1:], loop=0, duration=duration_ms,
    )


# %%
def main(config):
    """Capture from the camera, run inference, and write the demo outputs.

    Args:
        config: Runtime configuration.

    Returns:
        Process exit code.
    """
    print("[cam] capturing {} frames from IMX219...".format(config["frames"]))
    frames = capture_burst(config["frames"], CAP_DIR)
    kept = frames[config["warmup"]:]     # discard AE warmup frames
    print("[cam] captured {}, using {} after warmup".format(len(frames), len(kept)))
    if not kept:
        print("[cam] no frames left after warmup; lower --warmup")
        return 1

    runner = build_runner(config["engine"])
    annotated_frames = []
    best = None
    t0 = time.perf_counter()
    for fp in kept:
        image = cv2.imread(str(fp))
        if image is None:
            continue
        ann, ndet, frac = infer_frame(image, runner)
        annotated_frames.append(ann)
        if best is None or frac > best[1]:
            best = (ann, frac, ndet)
    elapsed = time.perf_counter() - t0
    n = max(1, len(annotated_frames))
    print("[cam] processed {} frames in {:.1f}s ({:.1f} ms/frame incl. decode)".format(
        n, elapsed, 1000.0 * elapsed / n))

    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)
    still = str(Path(config["out_dir"]) / "cam_annotated.jpg")
    gif = str(Path(config["out_dir"]) / "cam_demo.gif")
    if best is not None:
        cv2.imwrite(still, best[0])
        print("[cam] best frame: {} detections, crack {:.2f}% -> {}".format(
            best[2], 100.0 * best[1], still))
    save_gif(annotated_frames, gif, GIF_MS)
    print("[cam] wrote GIF ({} frames) -> {}".format(len(annotated_frames), gif))
    return 0


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live IMX219 crack segmentation on the Jetson Nano.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--out-dir", dest="out_dir", default=".")
    args = parser.parse_args()
    raise SystemExit(main(vars(args)))
