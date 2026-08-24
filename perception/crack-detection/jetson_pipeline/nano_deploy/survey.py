"""survey.py - geotagged crack survey on the Jetson Nano.

Runs the crack-segmentation engine over a source (the CSI camera, or a folder of
images), scores each detection's severity from the mask, reads the current GPS
fix from gpsd, and writes one record per crack-bearing frame to a JSONL log plus
an annotated thumbnail. The log feeds render_map.py, which turns it into an
interactive defect map.

Coordinates: the U-Blox fix is read via gpsd. Until the stereo-camera
metric-geolocation stage lands, records without a live fix carry
``gps_fix: false`` and null coordinates, and render_map.py places them on a
placeholder track. When a real fix is present the same code records true
coordinates - no changes needed.

3.6-safe. Reuses the engine/decode helpers from infer_seg.py (same directory).

Usage:
    OPENBLAS_CORETYPE=ARMV8 python3 survey.py \
        --engine models/yolo11n-seg_416_fp16.engine --source montage --out-dir survey_out
    OPENBLAS_CORETYPE=ARMV8 python3 survey.py \
        --engine models/yolo11n-seg_416_fp16.engine --source camera --frames 60
"""

# %% Imports
import argparse
import json
import subprocess
import time
from datetime import datetime
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
CAP_DIR     = "/tmp/survey_cap"
IMAGE_EXTS  = (".png", ".jpg", ".jpeg", ".bmp")

# Severity thresholds on crack-pixel fraction (% of frame). Crude but defensible
# and tunable; feeds the map pin colour. Documented so a reviewer can retune.
SEV_MINOR_MAX    = 1.0     # < 1%   -> minor
SEV_MODERATE_MAX = 5.0     # 1-5%   -> moderate, > 5% -> severe
DETECT_MIN_FRAC  = 0.05    # below this crack fraction, treat frame as no-crack
THUMB_WIDTH      = 400


# %%
def capture_burst(n_frames, out_dir):
    """Capture a JPEG burst from the IMX219 via GStreamer.

    Args:
        n_frames: Number of frames to capture.
        out_dir: Directory the frames are written into.

    Returns:
        Sorted list of captured frame paths (empty on failure).
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
    subprocess.run(["bash", "-c", pipeline], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(out.glob("cap_*.jpg"))


# %%
def list_source(source, frames, warmup):
    """Resolve the input source to a list of image paths.

    Args:
        source: ``"camera"`` or a path to a file or directory of images.
        frames: Frames to capture when ``source`` is the camera.
        warmup: Leading camera frames to discard (auto-exposure).

    Returns:
        List of image paths to process.
    """
    if source == "camera":
        return capture_burst(frames, CAP_DIR)[warmup:]
    path = Path(source)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [path] if path.is_file() else []


# %% GPS
def read_gps(timeout_s):
    """Read the current fix from gpsd via gpspipe.

    Args:
        timeout_s: Seconds to wait for a position report.

    Returns:
        A dict with ``fix`` (bool), ``lat``, ``lon`` (float or None) and
        ``mode`` (0/1 = no fix, 2 = 2D, 3 = 3D).
    """
    result = {"fix": False, "lat": None, "lon": None, "mode": 0}
    try:
        proc = subprocess.run(
            ["gpspipe", "-w", "-n", "15"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    for line in proc.stdout.splitlines():
        line = line.strip()
        if '"class":"TPV"' not in line:
            continue
        try:
            tpv = json.loads(line)
        except ValueError:
            continue
        mode = int(tpv.get("mode", 0))
        if mode >= 2 and "lat" in tpv and "lon" in tpv:
            result.update(fix=True, lat=float(tpv["lat"]),
                          lon=float(tpv["lon"]), mode=mode)
            return result
        result["mode"] = max(result["mode"], mode)
    return result


# %% Severity
def score_severity(mask):
    """Derive severity metrics from a binary crack mask.

    Args:
        mask: HxW uint8 mask, 0/255.

    Returns:
        A dict with ``area_pct``, ``length_px``, ``width_px`` and ``severity``.
    """
    area_pct = 100.0 * float((mask > 0).mean())
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    area_px = float((mask > 0).sum())
    # Thin-structure approximations: length ~ half the perimeter, mean width ~
    # area / length. Rough, but adequate to rank severity.
    length_px = perimeter / 2.0 if perimeter > 0 else 0.0
    width_px = (area_px / length_px) if length_px > 0 else 0.0
    if area_pct < SEV_MINOR_MAX:
        severity = "minor"
    elif area_pct < SEV_MODERATE_MAX:
        severity = "moderate"
    else:
        severity = "severe"
    return {
        "area_pct": round(area_pct, 3),
        "length_px": round(length_px, 1),
        "width_px": round(width_px, 2),
        "severity": severity,
    }


# %%
def build_runner(engine_path):
    """Load the engine and allocate reusable buffers.

    Args:
        engine_path: Path to the ``.engine`` file.

    Returns:
        ``(context, inputs, outputs, bindings, stream, imgsz)``.
    """
    engine = load_engine(engine_path, trt.Logger(trt.Logger.WARNING))
    context = engine.create_execution_context()
    inputs, outputs, bindings = allocate(engine, 640)
    return context, inputs, outputs, bindings, cuda.Stream(), inputs[0]["dims"][-1]


# %%
def infer(image, runner):
    """Run one frame and return the overlay, mask and detection count.

    Args:
        image: HxWx3 BGR uint8 frame.
        runner: Tuple from :func:`build_runner`.

    Returns:
        ``(annotated_bgr, mask, n_detections)``.
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
    return overlay(image, mask, boxes), mask, ndet


# %%
def save_thumb(image, path, width):
    """Write a width-limited JPEG thumbnail.

    Args:
        image: HxWx3 BGR uint8 image.
        path: Output path.
        width: Target width in pixels (aspect preserved).
    """
    h, w = image.shape[:2]
    scale = width / float(w) if w > width else 1.0
    thumb = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
    cv2.imwrite(str(path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])


# %% Main
def main(config):
    """Run the survey and write the detection log + thumbnails.

    Args:
        config: Runtime configuration.

    Returns:
        Process exit code.
    """
    out_dir = Path(config["out_dir"])
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "detections.jsonl"

    sources = list_source(config["source"], config["frames"], config["warmup"])
    if not sources:
        print("[survey] no input frames from source '{}'".format(config["source"]))
        return 1
    print("[survey] {} frame(s) to process".format(len(sources)))

    runner = build_runner(config["engine"])
    gps = read_gps(config["gps_timeout"])
    if gps["fix"]:
        print("[survey] GPS: fix mode {} @ {:.6f},{:.6f}".format(
            gps["mode"], gps["lat"], gps["lon"]))
    else:
        print("[survey] GPS: no live fix (mode {}) - coords null; placeholder "
              "track used pending stereo-camera geolocation".format(gps["mode"]))

    records = []
    with log_path.open("w", encoding="utf-8") as log:
        for i, src in enumerate(sources):
            image = cv2.imread(str(src))
            if image is None:
                continue
            annotated, mask, ndet = infer(image, runner)
            sev = score_severity(mask)
            if sev["area_pct"] < DETECT_MIN_FRAC:
                continue                                  # no crack in this frame
            # Re-read GPS per detection when a live fix exists (moving survey).
            fix = read_gps(config["gps_timeout"]) if gps["fix"] else gps
            thumb_name = "det_{:04d}.jpg".format(i)
            save_thumb(annotated, thumbs / thumb_name, THUMB_WIDTH)
            rec = {
                "id": i,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source": src.name,
                "n_detections": ndet,
                "gps_fix": fix["fix"],
                "lat": fix["lat"],
                "lon": fix["lon"],
                "thumb": "thumbs/" + thumb_name,
            }
            rec.update(sev)
            log.write(json.dumps(rec) + "\n")
            records.append(rec)

    counts = {"minor": 0, "moderate": 0, "severe": 0}
    for r in records:
        counts[r["severity"]] += 1
    print("[survey] {} detection(s): {} minor, {} moderate, {} severe".format(
        len(records), counts["minor"], counts["moderate"], counts["severe"]))
    print("[survey] log -> {}".format(log_path))
    print("[survey] thumbs -> {}".format(thumbs))
    return 0


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geotagged crack survey on the Jetson Nano.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--source", default="camera",
                        help="'camera' or a path to an image file/directory")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--gps-timeout", dest="gps_timeout", type=float, default=3.0)
    parser.add_argument("--out-dir", dest="out_dir", default="survey_out")
    args = parser.parse_args()
    raise SystemExit(main(vars(args)))
