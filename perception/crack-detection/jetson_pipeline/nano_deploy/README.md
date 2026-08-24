# nano_deploy — Jetson Nano (Tegra X1) concrete-crack-segmentation runtime

On-device TensorRT inference for **concrete crack detection** on the **original
Jetson Nano Dev Kit** — JetPack 4.6.x / L4T R32.7.6, **Python 3.6**, CUDA 10.2,
**TensorRT 8.2**, Maxwell (SM 5.3).

> **Project status.** This is the current state of a larger system, not the
> finished product. Detection and severity ranking work on-device today; planned
> next steps include **metric crack sizing** (width/length in millimetres) via a
> **stereo-camera setup** for true 3D geolocation, replacing the current
> single-camera pixel-space metrics and placeholder coordinates.

This is separate from the rest of `jetson_pipeline/`, which targets the **Orin**
(JetPack 6.x, TensorRT 10, Python 3.10+). The files here are deliberately
**Python 3.6-safe** — no `from __future__ import annotations`, no `X | Y` unions —
because that is the newest Python the Nano's final JetPack ships.

The Nano runs inference through the **raw TensorRT + pycuda** runtime with
hand-written YOLO-seg post-processing. There is no PyTorch, ONNX Runtime or
Ultralytics on the device: models are exported to ONNX on the laptop and only
the engine is built and run here.

## Files

| File | Purpose |
|------|---------|
| `infer_seg.py` | Load a YOLO-seg `.engine`, run one image, decode (NMS + mask assembly), save an annotated overlay + binary mask, report a per-stage timing breakdown. |
| `bench_engine.py` | Time an engine (pure infer cycle vs full preprocess+infer) for FPS profiling. |
| `cam_infer.py` | Live IMX219 camera capture → segmentation → annotated still + GIF. |
| `survey.py` | Crack survey over camera or an image folder: per-detection severity (from mask area) + GPS fix → `detections.jsonl` + thumbnails. |
| `render_map.py` | Host-side: turn a survey log into an interactive defect map, a CSV report and a preview image. |

## One-time device setup

The Nano has no internet in USB device-mode, so install offline (download on the
laptop, `scp` over). `pycuda` is the only extra dependency; TensorRT is already
present. `pycuda==2021.1` is the last version supporting Python 3.6 and must be
built from source with CUDA on `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_ROOT=/usr/local/cuda
export OPENBLAS_CORETYPE=ARMV8            # else numpy import SIGILLs on the A57
python3 -m pip install --user --no-index --no-build-isolation \
    --find-links ~/wheels pycuda==2021.1 pytools appdirs
```

## Build an engine (on the Nano, from a transferred ONNX)

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolo11n-seg_crack_416.onnx \
  --fp16 --workspace=1024 --saveEngine=models/yolo11n-seg_416_fp16.engine
```

The ONNX input size must be a multiple of 32. The dataset is 400x400 native, so
**416** (13x32) is the correct size — it is the smallest stride-valid size that
discards no pixels (384 would downscale and lose thin cracks).

## Run

```bash
export OPENBLAS_CORETYPE=ARMV8
python3 infer_seg.py \
  --engine models/yolo11n-seg_416_fp16.engine \
  --image sample.png --out annotated.png --mask-out mask.png
```

`infer_seg.py` auto-detects the input size and the mask-prototype resolution
(imgsz/4) from the engine bindings, so the same script works for 640/512/416
engines without edits.

## Performance notes (measured, honest)

- **The power supply dominates.** A weak micro-USB cable starves the board: MAXN
  with a bad cable was *slower* than 5 W because the rail sagged. A good cable
  plus `sudo nvpmodel -m 0` (MAXN, 4 cores) unlocked real gains. Do not trust
  perf numbers taken on a suspect cable or an unmeasured supply.
- The workload is **memory-bandwidth bound**; `jetson_clocks` (which pins EMC on
  this box, and errors on the missing `l4t_dfs.conf`) did not help.
- FP16 only. **INT8 is unavailable** — Maxwell has no DP4A instruction.
- Reporting FPS on micro-USB without a power-rail measurement is not defensible;
  quote FPS only from a barrel-jack 5 V/4 A supply (with the J48 jumper) or with
  logged clock-stability under sustained load.

See the memory note `jetson-nano-goliath` for the full board profile and the
connection details (`ssh goliath@192.168.55.1`).
