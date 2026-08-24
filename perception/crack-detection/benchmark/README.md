# Crack Detection Benchmark

Real-time crack detection benchmark for a concrete inspection robot. It times
three models — **YOLO11n** (detection), **YOLO11n-seg** (segmentation) and
**FastSAM-s** (segmentation) — on the images and videos you provide, then
produces per-frame metrics, summary statistics and charts.

Built for a Windows laptop with an RTX 4070 (8 GB).

## Project layout

```
crack_detection_benchmark/
├── models/             # model weights live here (pretrained + fine-tuned)
├── inputs/             # drop your images and videos here
├── outputs/            # annotated results are written here
├── results/            # CSV logs and charts
├── data/               # training data and prepared dataset
│   ├── raw/            # place the CrackSeg9k download here
│   └── processed/      # YOLO-format dataset written by prepare_data.py
├── config.yaml         # benchmark parameters
├── train_config.yaml   # training hyperparameters
├── benchmark.py        # benchmark entry point
├── prepare_data.py     # CrackSeg9k -> YOLO dataset conversion
├── train.py            # two-phase fine-tuning of all three models
├── validate.py         # post-training evaluation
├── detector.py         # model wrapper and inference logic
├── metrics.py          # FPS, latency and per-frame timing
├── visualise.py        # annotated frame rendering
├── report.py           # summary table and chart generation
└── requirements.txt
```

## Setup

1. Install Python 3.10 or newer.
2. (Recommended) create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   For CUDA acceleration on the RTX 4070, install a CUDA-enabled PyTorch build
   matching your driver — see https://pytorch.org/get-started/locally/.

### Model weights

* **YOLO11n** (`yolo11n.pt`) — downloaded automatically by Ultralytics on first
  run into the `models/` directory.
* **YOLO11n-seg** (`yolo11n-seg.pt`) — downloaded automatically by Ultralytics
  on first run into the `models/` directory.
* **FastSAM-s** (`FastSAM-s.pt`) — **must be downloaded manually.** Get it from
  the Ultralytics assets release and place it in `models/`:

  ```
  https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt
  ```

  If `models/FastSAM-s.pt` is missing, the benchmark stops with a clear error
  message containing this URL.

## Adding input files

Drop any `.jpg`, `.jpeg`, `.png`, `.mp4`, `.avi` or `.mov` files into the
`inputs/` directory. Subdirectories are not scanned — files must sit directly
in `inputs/`. Frames are processed at **native resolution** (no resizing).

## Running

```powershell
python benchmark.py
```

The run shows a live `rich` progress table (current model, current file,
current FPS, running mean FPS). Each model is warmed up before timing begins,
then every input file is processed frame by frame. Videos are timed per frame;
static images are inferred 20 times and averaged for stable timing.

All parameters — confidence/IOU thresholds, device, warmup iterations, whether
to save annotated frames or show a live preview — are set in `config.yaml`.

## Outputs

| Location | Contents |
| --- | --- |
| `outputs/{model}/{video}/frame_00000.jpg` | annotated video frames |
| `outputs/{model}/{image}.jpg` | annotated images |
| `results/frame_metrics.csv` | master per-frame timing log |
| `results/summary.csv` | per-model summary statistics |
| `results/fps_comparison.png` | grouped mean-FPS bar chart (p95 error bars) |
| `results/latency_distribution.png` | overlapping latency KDE curves |
| `results/latency_timeline.png` | per-frame latency timeline (first video) |

A summary table is also printed to the terminal: model, mean FPS, p50/p95/p99
latency, cold-start latency and parameter count.

The first real inference frame after warmup is flagged `cold_start=True` in
`frame_metrics.csv` so it can be excluded from steady-state analysis.

## Training

The benchmark ships with the generic COCO-pretrained weights. To make the
models actually good at finding cracks, fine-tune them on the **CrackSeg9k**
dataset. The training pipeline converts CrackSeg9k to YOLO format, fine-tunes
all three models and feeds the new weights straight back into the benchmark.

All training hyperparameters live in `train_config.yaml` — epochs, image size,
batch size, learning rate schedule, backbone-freeze length, early-stopping
patience, the train/val/test split ratios and per-augmentation toggles.

### Getting CrackSeg9k

Download the CrackSeg9k dataset (images + binary crack masks) from its public
release:

```
https://github.com/Dhananjay42/crackseg9k
```

Extract it into `data/raw/` so the structure looks like this:

```
data/raw/<subfolder>/images/<name>.jpg     # RGB images
data/raw/<subfolder>/masks/<name>.png      # binary masks (or annotations/)
```

`<subfolder>` is the source category (e.g. `surfaces`, `bridges`, `walls`);
the split is stratified across these so every category is represented in
train/val/test. Masks are matched to images by shared filename — a sibling
`masks/` **or** `annotations/` folder is accepted.

### Step 1 — `python prepare_data.py`

Converts every binary mask into YOLO segmentation polygons (one `.txt` per
image, class `0` = `crack`), copies images into the train/val/test split and
writes `data/processed/dataset.yaml`. Check the printed `rich` summary table:
the train/val/test image and polygon counts should be non-zero, and the
"skipped" rows tell you how many masks were blank, unmatched or pure noise.

### Step 2 — `python train.py`

Fine-tunes YOLO11n, YOLO11n-seg and FastSAM-s sequentially. Each model is
trained in two phases — backbone frozen first, then unfrozen for end-to-end
fine-tuning. A live `rich` table shows per-epoch train/val loss, mAP50 and
learning rate; a final comparison table reports best mAP per model.

On an **RTX 4070 (8 GB)** at the default settings (100 epochs, 640 px,
batch 16) expect roughly **45–90 minutes per model** — about **2.5–4.5 hours**
for all three — and **6–7.5 GB** of GPU memory. If you hit a CUDA
out-of-memory error, lower `batch` in `train_config.yaml` to 8 or 4.

Outputs: `models/{model}_crack_finetuned.pt` (best) and
`models/{model}_crack_last.pt` (last) per model, plus
`results/training_summary.csv`.

### Step 3 — `python validate.py`

Evaluates each fine-tuned model on the **test** split and produces, per model,
`results/val_{model}_confusion_matrix.png`, `results/val_{model}_pr_curve.png`
and `results/val_{model}_f1_curve.png`, plus a grouped mAP comparison chart
`results/val_map_comparison.png`. It then **repoints `config.yaml`** at the
fine-tuned checkpoints and prints a summary table (mAP50, mAP50-95, precision,
recall, F1).

### Step 4 — `python benchmark.py`

Re-run the benchmark. Because `validate.py` already updated `config.yaml`, the
benchmark loads the fine-tuned weights automatically — no manual config edit
is needed.

> **Note:** fine-tuning does not change the benchmark FPS numbers. Inference
> speed depends on the network architecture, which is unchanged — only the
> learned weights differ. Fine-tuning improves *detection quality*, not
> *throughput*.
