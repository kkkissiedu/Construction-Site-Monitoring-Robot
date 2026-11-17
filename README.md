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
- **Rover drive controller**: an STM32F103C6 drives the four wheel motors from
  `V <left> <right>` commands over a UART, with a failsafe stop and a watchdog.
  Bench-tested with the wheels lifted; not yet connected to the Jetson. See
  `robotics/rover-firmware/README.md`.

## Repository layout

```
perception/
  crack-detection/
    benchmark/        # training, validation, export, benchmark app (laptop)
    jetson_pipeline/  # ONNX/TensorRT export pipeline + docs
      nano_deploy/    # Jetson Nano runtime: engine inference, camera, survey, map
robotics/
  rover-firmware/     # STM32F103C6 drive controller: UART commands, BTS7960 PWM
                      # (planned) navigation, control, platform integration
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
- Four DC gearmotors on BTS7960 drivers, run by an STM32F103C6 (Blue Pill), on a
  3S Li-ion pack

See `perception/crack-detection/jetson_pipeline/nano_deploy/README.md` for the
device setup, runtime, and honest performance notes.
