# jetson_pipeline — Training, Export and Jetson Deployment

Training, export and Jetson Orin Nano Super deployment for the GhanaCrack YOLO
segmentation models. Shares the dataset and trained weights with
`crack_detection_benchmark/` but is otherwise a separate pipeline with its own
scripts and its own config.

Nothing here hardcodes a model list. The model set comes from `model_matrix` in
[`jetson_config.yaml`](jetson_config.yaml) crossed against a filesystem scan of
`crack_detection_benchmark/models/`, using the shared naming convention:

```
{version}{size}-seg_crack_finetuned.{pt,onnx,engine}   # FP32 / FP16
{version}{size}-seg_crack_int8.engine                  # INT8 (laptop build)
{version}{size}-seg_crack_finetuned_jetson.engine      # FP16 (device build)
{version}{size}-seg_crack_int8_jetson.engine           # INT8 (device build)
models/calibration_cache/{version}{size}-seg_int8.cache
```

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `train_missing.py` | Laptop | Trains only the models missing from the matrix |
| `export_all.py` | Laptop | ONNX FP32 → TRT FP16 → TRT INT8, in that order |
| `calibrate_int8.py` | Both | Builds INT8 calibration caches from the CrackSeg9k test split |
| `export_on_device.py` | Jetson | Recompiles transferred ONNX graphs for the device GPU |
| `verify_exports.py` | Both | Loads every artefact and runs one forward pass |
| `jetson_config.yaml` | — | Single source of truth for paths and hyperparameters |
| `requirements_jetson.txt` | Jetson | On-device deps only — no training packages |

---

## Workflow 1 — Laptop: train the missing models

```bash
python jetson_pipeline/train_missing.py
```

Prints a discovery report, then asks before training anything:

```
               Model Matrix
┌─────────┬───────┬───────┬───────┬───────┐
│ Version │   N   │   S   │   M   │   L   │
├─────────┼───────┼───────┼───────┼───────┤
│ YOLO11  │  OK   │  OK   │  OK   │  OK   │
│ YOLO12  │ TRAIN │ TRAIN │ TRAIN │ TRAIN │
│ YOLO26  │  OK   │  OK   │ TRAIN │ TRAIN │
└─────────┴───────┴───────┴───────┴───────┘
Found 6 missing models. Proceed? [y/n]
```

What it does per model, in matrix order (YOLO11 → YOLO12 → YOLO26, n→s→m→l):

1. Downloads the pretrained `{version}{size}-seg.pt` through Ultralytics.
2. **Phase 1** — backbone frozen for `freeze_epochs`, flat LR schedule.
3. **Phase 2** — fully unfrozen for the remaining epochs, cosine decay to `lrf`.
4. Validates on the val split and appends a row to `training_log.csv`.
5. Copies the best checkpoint to
   `crack_detection_benchmark/models/{version}{size}-seg_crack_finetuned.pt`.

Batch sizes are chosen at startup from `vram_thresholds` by reading
`torch.cuda.get_device_properties(0).total_memory` — under 5.5 GiB uses
`safe_4gb`, under 7.5 GiB uses `safe_6gb`, otherwise `safe_8gb`.

**Resuming.** A crashed run leaves `runs/train/{model}_phase{n}/weights/last.pt`
behind and the next invocation resumes from it — a run with a checkpoint is
never restarted. A `RuntimeError` (e.g. a CUDA fault) is retried up to three
times, clearing the CUDA cache between attempts. If you change a phase's epoch
budget or freeze depth the old checkpoint no longer matches and that phase
restarts from scratch, with a warning saying so.

Pass `--yes` to skip the confirmation for an unattended run.

---

## Workflow 2 — Laptop: export every format

```bash
python jetson_pipeline/export_all.py
```

Three phases, strictly ordered. **Each phase finishes for every model before
the next begins**, and INT8 always runs last:

| Phase | Output | Notes |
|-------|--------|-------|
| 1 — ONNX FP32 | `{name}_crack_finetuned.onnx` | opset 17, simplified; verified > 1 MB |
| 2 — TRT FP16 | `{name}_crack_finetuned.engine` | Skipped for models whose ONNX failed |
| 3 — TRT INT8 | `{name}_crack_int8.engine` | Calibrated on 200 test-split images |

A failure in one model is logged and the phase continues to the next model — it
never aborts the run. INT8 failures also capture a full traceback in
`int8_failures.log`. The whole INT8 phase is wrapped so a native crash inside
the TensorRT builder still reaches the summary table.

Run a subset with `--phases 1 2` (ONNX + FP16 only).

### Pre-building calibration caches

`export_all.py` builds any missing cache itself, but you can do it up front:

```bash
python jetson_pipeline/calibrate_int8.py
```

```bash
python jetson_pipeline/calibrate_int8.py --force
```

This scans for `*_crack_finetuned.onnx` and, for each, streams 200 images from
`data/processed/images/test/` — sampled with a fixed seed of 42, so the set is
reproducible — through `trt.IInt8EntropyCalibrator2`. Preprocessing matches YOLO
inference exactly: letterbox to 640×640 on a constant-114 canvas, BGR→RGB,
HWC→CHW, scaled to [0,1], float32, contiguous. Caches land in
`models/calibration_cache/` — one per model, since calibration is
architecture-specific — and are reused on later builds rather than recomputed.

Expect roughly 5–10 minutes per model: the calibration pass itself is quick, but
TensorRT's tactic search over a quantised segmentation graph is not.

### Verify

```bash
python jetson_pipeline/verify_exports.py
```

Loads every artefact and runs one forward pass. ONNX graphs get a random
`(1,3,640,640)` float32 tensor and a strict output-shape check; engines are
loaded through the Ultralytics wrapper and must return a result. Exits 0 if all
pass and 1 if any fail, so it can gate a script or CI job.

> **Note on expected shapes.** YOLO11 and YOLO12 share the anchor-grid head,
> `(1, 37, 8400)` + `(1, 32, 160, 160)`. YOLO26 is NMS-free end to end and emits
> `(1, 300, 38)` + `(1, 32, 160, 160)` instead — 300 already-selected detections
> of 4 box + 1 conf + 1 class + 32 mask coefficients. Both contracts live in
> `verify.expected_shapes` / `verify.expected_shapes_by_version`; checking stays
> strict for both.

---

## Workflow 3 — Transfer to the Jetson

TensorRT engines are tied to the GPU architecture, driver and TensorRT build
that produced them, so **do not copy the laptop's `.engine` files** — copy the
ONNX graphs and rebuild on the device.

Set your host once:

```bash
export JETSON=orin@192.168.1.50 DEST=/home/orin/ghanacrack
```

**1. Create the layout on the device:**

```bash
ssh $JETSON "mkdir -p $DEST/jetson_pipeline $DEST/crack_detection_benchmark/models/calibration_cache"
```

**2. Scripts and config** — four files, because `export_on_device.py` imports
`calibrate_int8.py` and both read `jetson_config.yaml`:

```bash
scp jetson_pipeline/export_on_device.py jetson_pipeline/calibrate_int8.py jetson_pipeline/verify_exports.py jetson_pipeline/jetson_config.yaml jetson_pipeline/requirements_jetson.txt $JETSON:$DEST/jetson_pipeline/
```

**3. ONNX graphs:**

```bash
rsync -avh --progress crack_detection_benchmark/models/*_crack_finetuned.onnx $JETSON:$DEST/crack_detection_benchmark/models/
```

**4. Calibration caches** — a few KB each; skip only if you want the Jetson to
recalibrate, which also requires step 5:

```bash
rsync -avh crack_detection_benchmark/models/calibration_cache/ $JETSON:$DEST/crack_detection_benchmark/models/calibration_cache/
```

**5. Test split** — only needed when a calibration cache is missing and has to
be rebuilt on-device:

```bash
rsync -avh crack_detection_benchmark/data/processed/images/test/ $JETSON:$DEST/crack_detection_benchmark/data/processed/images/test/
```

The relative layout must be preserved: every path in `jetson_config.yaml`
resolves from the script's own directory, so `jetson_pipeline/` and
`crack_detection_benchmark/` must remain siblings on the device.

**6. Install the on-device dependencies:**

```bash
ssh $JETSON "cd $DEST/jetson_pipeline && python3 -m pip install -r requirements_jetson.txt"
```

TensorRT and CUDA come from JetPack via apt — do **not** pip install them.

---

## Workflow 4 — Jetson: rebuild the engines for the device GPU

```bash
cd ~/ghanacrack/jetson_pipeline && python3 export_on_device.py
```

It first detects and reports the environment:

```
[jetson-env] JetPack: 6.x (L4T R36.4.0)
[jetson-env] TensorRT: 10.3.0
[jetson-env] CUDA: 12.6
[jetson-env] GPU: Orin (SM 8.7), 8192 MiB VRAM
[jetson-env] Recommended batch size: 1 (fixed for inference)
[jetson-env] Builder API path: TensorRT 10.x (implicit explicit-batch, memory-pool workspace)
```

then prints the transfer checklist, so a missing file is named before any build
starts:

```
[transfer-check] Required files on Jetson:
  ✔ yolo11n-seg_crack_finetuned.onnx - found
  ✗ yolo12n-seg_crack_finetuned.onnx - MISSING
  ✔ calibration_cache/yolo11n-seg_int8.cache - found
```

Use `--check-only` to stop after the report.

**What it produces**, into `crack_detection_benchmark/models/`:

* `{name}_crack_finetuned_jetson.engine` — FP16
* `{name}_crack_int8_jetson.engine` — INT8, reusing the transferred cache, or
  calibrating on-device when the cache is absent

**TensorRT version adaptation** is automatic. On 8.x the builder uses the
`EXPLICIT_BATCH` network flag and `max_workspace_size`; on 10.x it uses the
implicit explicit-batch network and `set_memory_pool_limit`. Precision flags are
resolved by name (`FP16` or `kFP16`) rather than hardcoded, and the missing
`trt.nptype` helper is restored on TRT 10+.

The script uses the raw TensorRT Python API only — no Ultralytics, no PyTorch.
`rich` is used when installed and falls back to plain-text tables when not.

---

## Configuration

Everything lives in [`jetson_config.yaml`](jetson_config.yaml). Paths are
written relative to `jetson_pipeline/` and resolved with `pathlib` at runtime —
no absolute path appears in any script.

| Key | Meaning |
|-----|---------|
| `model_matrix` | Versions × sizes the pipeline expects to exist |
| `training.sizes.{n,s,m,l}` | Per-size epochs, LR, freeze depth, patience |
| `vram_thresholds` | Batch-size tables selected by detected VRAM |
| `export.int8_calib_images` | Calibration set size (default 200) |
| `export.int8_calib_split` | Split to calibrate from (default `test`) |
| `naming.*` | Filename suffixes shared with the benchmark app |
| `export.workspace_mib` | TensorRT builder workspace on the laptop |
| `jetson.workspace_mib` | TensorRT builder workspace on the device |
| `verify.expected_shapes*` | ONNX output contracts, with per-version overrides |

---

## Troubleshooting

### TensorRT version mismatch

```
[TRT] [E] The engine plan file is not compatible with this version of TensorRT
[TRT] [E] Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed
```

You copied a `.engine` built elsewhere. Engines are not portable across GPU
architectures, TensorRT versions or driver versions. Copy the `.onnx` instead
and run `export_on_device.py`. Confirm both sides with:

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### `module 'tensorrt' has no attribute 'nptype'`

TensorRT 10 removed `trt.nptype`, which Ultralytics still calls. Every script
here installs a replacement at import time via
`calibrate_int8.patch_trt_nptype()`, and `crack_detection_benchmark/app.py`
carries the same shim. If you hit this in your own script, call that function
before loading any engine — importing `calibrate_int8` is enough.

### INT8 calibration cache mismatch

```
[TRT] [E] Calibration failure: tensor scales for <layer> not found in cache
[TRT] [W] Missing scale and zero-point for tensor <name>, expect fall back to non-int8
```

The cache was built from a different graph. Caches are architecture-specific —
one per model, never shared across sizes or versions — and any re-export that
changes the graph invalidates them. Delete the stale file and rebuild:

```bash
python jetson_pipeline/calibrate_int8.py --force
```

The cache's first line encodes the TensorRT version that wrote it, e.g.
`TRT-100700-EntropyCalibration2` for TensorRT 10.7. A cache written by a
different major version should be rebuilt rather than reused.

### `Could not find any implementation for node …/proto/…` during an INT8 build

```
[TRT] [E] Error Code 10: Internal Error (Could not find any implementation for
node /model.23/proto/cv3/conv/Conv + PWN(...))
```

The builder found no usable tactic for the YOLO-seg mask-prototype branch while
INT8 was enabled. This is a property of the graph and the TensorRT build, not of
your machine's memory.

**Measured on this project — TensorRT 10.7, RTX 4070 Laptop (SM 8.9),
yolo11n-seg. Every one of these was actually run:**

| Attempt | Result |
|---------|--------|
| Raw builder, INT8 only, 2048 MiB workspace | fails on `/model.23/proto/cv3/conv/Conv` |
| Raw builder, INT8 + FP16, 2048 MiB | fails identically |
| Raw builder, INT8 + FP16, 6144 MiB | fails identically |
| Ultralytics `export(format="engine", int8=True, data=…)` | fails identically |
| Raw builder + `PREFER_PRECISION_CONSTRAINTS`, proto layers pinned to FP16 | fails identically |

Conclusions, each backed by the run above rather than assumed:

* **Workspace size is not the cause.** Tested at 2048 and 6144 MiB — identical
  failure. Do not chase this.
* **It is not specific to this repo's builder.** Ultralytics' own exporter fails
  on the same node with the same error.
* **FP16 fallback does not rescue it.** With the proto layers explicitly pinned,
  TensorRT reports `No valid obedient candidate choices … that meet the
  preferred precision` and then finds no implementation at all — so there is no
  conforming INT8 *or* FP16 tactic for that fused `Conv + PWN(Sigmoid, Mul)`
  once INT8 is enabled. (A pure FP16 build of the same graph succeeds, which is
  why the FP16 engines exist and verify.)

This is a TensorRT-build/architecture limitation for the YOLO-seg mask-prototype
branch, not a defect in this pipeline. **It may not reproduce on the Jetson** —
JetPack 6.x ships TensorRT 10.3 on Orin's SM 8.7, a different kernel library
from the desktop's 10.7 on SM 8.9, and INT8 tactic coverage for fused kernels
genuinely differs between them. `export_on_device.py`'s INT8 stage is therefore
worth running on the target; it just could not be validated from the laptop.

If it does fail on the device too, nothing else is affected: the model is logged
to `int8_failures.log`, the phase continues, FP32/FP16 artefacts are unchanged,
and the benchmark app simply omits the INT8 option for that model.

One practical detail worth knowing: **the calibration cache is still produced**
even when the engine build fails. TensorRT writes the calibration table before
tactic selection, so `calibrate_int8.py` completes successfully and leaves a
valid, reusable cache behind — verified at 17 KB with a
`TRT-100700-EntropyCalibration2` header. Do not delete it after a failed build;
the retry will not have to recalibrate.

### VRAM out-of-memory during export

```
[TRT] [E] 2: [virtualMemoryBuffer.cpp] Requested amount of GPU memory could not be allocated
torch.OutOfMemoryError: CUDA out of memory
```

Close the benchmark app and any other CUDA process first — `export_all.py` calls
`torch.cuda.empty_cache()` between every model, but it cannot reclaim another
process's memory. Then lower the workspace setting, or export in stages:

```bash
python jetson_pipeline/export_all.py --phases 1 2
```

On the Jetson, give the builder the full power budget first:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

and add swap for the larger models:

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```

### Training will not resume

If a phase restarts instead of resuming, the run printed the reason: the saved
`epochs` or `freeze` in `last.pt` no longer matches the config. Either restore
the previous values in `jetson_config.yaml` or accept the restart. To force a
clean start, delete `jetson_pipeline/runs/train/{model}_phase*`.
