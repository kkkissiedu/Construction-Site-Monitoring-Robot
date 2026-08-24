# Construction Site Monitoring Robot

A mobile robot for autonomous construction-site monitoring — bringing together
the **computer-vision** (perception) and **robotics** (navigation, control)
sides of the system in one place.

> **Project status.** Active, early-stage. This monorepo is the current state of
> a larger system, not a finished product. What runs today is the perception
> stack — concrete crack detection with on-device inference on a Jetson. The
> robotics side and several perception upgrades (below) are planned.

## What works today

- **Concrete crack segmentation** (YOLO11n-seg) trained and benchmarked on the
  laptop, exported to ONNX and compiled to a **TensorRT FP16 engine that runs
  on a Jetson Nano** through a lightweight custom runtime (no PyTorch or
  Ultralytics on the device).
- **On-device survey tooling**: capture → segment → per-detection **severity
  ranking** (from the crack mask) → GPS-tagged log → interactive defect map.

## Repository layout

```
perception/
  crack-detection/
    benchmark/        # training, validation, export, benchmark app (laptop)
    jetson_pipeline/  # ONNX/TensorRT export pipeline + docs
      nano_deploy/    # Jetson Nano runtime: engine inference, camera, survey, map
robotics/             # (planned) navigation, control, platform integration
docs/                 # (planned) design notes, results, ADRs
```

Model weights, datasets, compiled engines and run outputs are intentionally
**not** tracked here (see `.gitignore`) — the repo stays code-only.

## Roadmap

- **Metric crack sizing** — width and length in millimetres via a **stereo-camera
  setup** for true 3D geolocation, replacing the current single-camera
  pixel-space metrics and placeholder coordinates.
- **Robotics integration** — autonomous navigation and coverage planning so the
  robot surveys a site on its own.
- **Multiclass distress** — extend beyond binary crack/background to distinct
  distress types.
- **Real-time on-device mapping** — stream geotagged detections live.

## Hardware (current bench setup)

- NVIDIA Jetson Nano (Tegra X1, 4 GB) — edge inference
- CSI camera (IMX219) — perception input
- U-Blox GNSS receiver — geolocation

See `perception/crack-detection/jetson_pipeline/nano_deploy/README.md` for the
device setup, runtime, and honest performance notes.
