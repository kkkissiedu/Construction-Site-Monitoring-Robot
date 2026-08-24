# %%
"""train_missing.py - fine-tune only the models missing from the model matrix.

The script never carries a hardcoded model list. It derives the full expected
set from ``model_matrix`` in ``jetson_config.yaml``, scans ``models_dir`` for
``*_crack_finetuned.pt`` files, and trains the difference.

For every missing model it:

1. Downloads the pretrained ``{version}{size}-seg.pt`` via Ultralytics.
2. Runs **phase 1** with the backbone frozen for ``freeze_epochs``.
3. Runs **phase 2** with the whole network unfrozen for the remaining epochs,
   under a cosine learning-rate decay down to ``lrf``.
4. Resumes from ``runs/train/{model}_phase{n}/weights/last.pt`` when a crashed
   run left one behind - a run with a checkpoint is never restarted.
5. Retries up to three times on a ``RuntimeError``, clearing the CUDA cache
   between attempts.
6. Validates on the val split and appends a row to ``training_log.csv``.
7. Copies the best checkpoint to
   ``crack_detection_benchmark/models/{version}{size}-seg_crack_finetuned.pt``.

Usage:
    conda activate cuda_pt
    python jetson_pipeline/train_missing.py
    python jetson_pipeline/train_missing.py --yes    # skip the confirmation
"""

from __future__ import annotations

import os

# CUDA_LAUNCH_BLOCKING=1 makes kernel launches synchronous so a CUDA fault is
# reported at its true call site. It must be set before torch is imported.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Silences the Ultralytics tqdm batch bars so the rich progress bar owns the
# terminal. Must be set before ultralytics is imported.
os.environ["YOLO_VERBOSE"] = "False"

import argparse
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

# %% Config
_HERE: Path = Path(__file__).parent
_PROJECT_ROOT: Path = _HERE.parent
_CONFIG_PATH: Path = _HERE / "jetson_config.yaml"
_RUNS_DIR: Path = _HERE / "runs" / "train"

# Canonical size ordering used for every report and training sequence.
_SIZE_ORDER: tuple[str, ...] = ("n", "s", "m", "l", "x")

# VRAM tier boundaries in GiB. A GPU is assigned the first tier it fits under.
_VRAM_TIERS: tuple[tuple[float, str], ...] = (
    (5.5, "safe_4gb"),
    (7.5, "safe_6gb"),
    (float("inf"), "safe_8gb"),
)

# Bytes per GiB, used to turn ``total_memory`` into a human tier decision.
_BYTES_PER_GIB: float = 1024.0 ** 3


# %% Config loading
def load_config(config_path: Path) -> dict[str, Any]:
    """Load ``jetson_config.yaml``.

    Args:
        config_path: Path to the pipeline configuration file.

    Returns:
        The parsed configuration as a nested dict.

    Raises:
        FileNotFoundError: If the configuration file is absent.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing pipeline config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# %%
def resolve_path(config_value: str) -> Path:
    """Resolve a config path written relative to ``jetson_pipeline/``.

    Args:
        config_value: A relative path string from ``jetson_config.yaml``.

    Returns:
        The absolute, normalised path.
    """
    return (_HERE / config_value).resolve()


# %% Model discovery
def parse_model_stem(stem: str, finetuned_suffix: str) -> tuple[str, str] | None:
    """Parse a checkpoint stem into ``(version, size_letter)``.

    Mirrors ``_parse_model_filename`` in ``crack_detection_benchmark/app.py`` so
    both halves of the project agree on the naming convention.

    Args:
        stem: Filename without extension, e.g. ``yolo11n-seg_crack_finetuned``.
        finetuned_suffix: Stem suffix marking a fine-tuned checkpoint.

    Returns:
        ``(version, size_letter)``, or ``None`` when the stem does not match
        the convention or carries an unknown size letter.
    """
    if not stem.endswith(finetuned_suffix):
        return None
    head = stem[: -len(finetuned_suffix)]
    if not head.endswith("-seg"):
        return None
    head = head[: -len("-seg")]
    if not head:
        return None
    size_letter = head[-1]
    if size_letter not in _SIZE_ORDER:
        return None
    version = head[:-1]
    if not version:
        return None
    return version, size_letter


# %%
def scan_existing_models(models_dir: Path, finetuned_suffix: str) -> dict[str, set[str]]:
    """Scan ``models_dir`` for fine-tuned checkpoints already on disk.

    The filename convention is ``{version}{size}-seg{finetuned_suffix}.pt``,
    e.g. ``yolo11n-seg_crack_finetuned.pt`` -> version ``yolo11``, size ``n``.

    Args:
        models_dir: Directory holding the trained weights.
        finetuned_suffix: Stem suffix marking a fine-tuned checkpoint.

    Returns:
        A ``{version: {size_letter, ...}}`` mapping. Empty when the directory
        is absent - never raises.
    """
    found: dict[str, set[str]] = {}
    if not models_dir.is_dir():
        return found
    for path in sorted(models_dir.glob(f"*{finetuned_suffix}.pt")):
        parsed = parse_model_stem(path.stem, finetuned_suffix)
        if parsed is None:
            continue
        version, size_letter = parsed
        found.setdefault(version, set()).add(size_letter)
    return found


# %%
def compute_missing(
    matrix: dict[str, list[str]],
    existing: dict[str, set[str]],
) -> list[tuple[str, str]]:
    """Diff the expected model matrix against what is already on disk.

    Args:
        matrix: The ``{version: [size, ...]}`` block from the config.
        existing: The ``{version: {size, ...}}`` mapping from the disk scan.

    Returns:
        Ordered ``(version, size_letter)`` pairs still to be trained. Versions
        follow their declaration order in the config; sizes follow
        :data:`_SIZE_ORDER`.
    """
    missing: list[tuple[str, str]] = []
    for version, sizes in matrix.items():
        have = existing.get(version, set())
        wanted = set(sizes)
        for size_letter in _SIZE_ORDER:
            if size_letter in wanted and size_letter not in have:
                missing.append((version, size_letter))
    return missing


# %%
def model_key(version: str, size_letter: str) -> str:
    """Build the composite model key used across the whole project.

    Args:
        version: Version prefix, e.g. ``yolo12``.
        size_letter: Single size letter, e.g. ``n``.

    Returns:
        The key, e.g. ``yolo12n-seg``.
    """
    return f"{version}{size_letter}-seg"


# %%
def print_discovery_report(
    matrix: dict[str, list[str]],
    existing: dict[str, set[str]],
    missing: list[tuple[str, str]],
    models_dir: Path,
    console: Console,
) -> None:
    """Print the full startup discovery report.

    Args:
        matrix: The expected model matrix.
        existing: Versions/sizes found on disk.
        missing: Pairs that still need training.
        models_dir: The directory that was scanned.
        console: ``rich`` console for output.
    """
    console.print("[bold]Jetson pipeline - model discovery[/bold]")
    console.print(f"[dim]Scanned: {models_dir}[/dim]")

    active_letters = [
        letter
        for letter in _SIZE_ORDER
        if any(letter in set(sizes) for sizes in matrix.values())
    ]
    table = Table(title="Model Matrix", title_style="bold white")
    table.add_column("Version", style="bold")
    for letter in active_letters:
        table.add_column(letter.upper(), justify="center")

    missing_set = set(missing)
    for version, sizes in matrix.items():
        row = [version.upper()]
        wanted = set(sizes)
        have = existing.get(version, set())
        for letter in active_letters:
            if letter not in wanted:
                row.append("[dim]-[/dim]")
            elif letter in have:
                row.append("[green]OK[/green]")
            elif (version, letter) in missing_set:
                row.append("[yellow]TRAIN[/yellow]")
            else:
                row.append("[red]?[/red]")
        table.add_row(*row)
    console.print(table)

    have_count = sum(
        1
        for version, sizes in matrix.items()
        for letter in set(sizes)
        if letter in existing.get(version, set())
    )
    console.print(f"[green]Present:[/green] {have_count} model(s)")
    if missing:
        names = ", ".join(model_key(version, letter) for version, letter in missing)
        console.print(f"[yellow]Missing:[/yellow] {len(missing)} model(s) - {names}")
        console.print(f"[bold]Will train:[/bold] {names}")
    else:
        console.print("[green]Missing:[/green] none - the matrix is complete.")

    # Surface finetuned weights the matrix does not ask for so an unexpected
    # file is visible rather than silently ignored.
    extras = [
        model_key(version, letter)
        for version, letters in sorted(existing.items())
        for letter in sorted(letters, key=_SIZE_ORDER.index)
        if letter not in set(matrix.get(version, []))
    ]
    if extras:
        console.print(f"[dim]Outside the matrix (ignored): {', '.join(extras)}[/dim]")


# %%
def confirm_proceed(count: int, assume_yes: bool, console: Console) -> bool:
    """Ask the user to confirm before any training starts.

    Args:
        count: Number of models that would be trained.
        assume_yes: When ``True`` skip the prompt entirely.
        console: ``rich`` console for output.

    Returns:
        ``True`` when training should proceed.
    """
    if assume_yes:
        console.print("[dim]--yes supplied; skipping confirmation.[/dim]")
        return True
    if sys.stdin is None or not sys.stdin.isatty():
        console.print(
            "[yellow]No interactive stdin; re-run with --yes to train "
            "non-interactively.[/yellow]"
        )
        return False
    answer = input(f"Found {count} missing models. Proceed? [y/n] ").strip().lower()
    return answer in {"y", "yes"}


# %% VRAM handling
def select_batch_sizes(
    vram_thresholds: dict[str, dict[str, int]],
    console: Console,
) -> dict[str, int]:
    """Pick the per-size batch table that fits the detected GPU.

    Args:
        vram_thresholds: The ``vram_thresholds`` block from the config.
        console: ``rich`` console for output.

    Returns:
        A ``{size_letter: batch}`` mapping. Falls back to the smallest tier
        when CUDA is unavailable so a CPU dry-run still works.
    """
    if not torch.cuda.is_available():
        tier = "safe_4gb"
        console.print(
            f"[yellow][vram] CUDA unavailable - using the '{tier}' batch table.[/yellow]"
        )
        return dict(vram_thresholds[tier])

    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / _BYTES_PER_GIB
    selected = "safe_8gb"
    for limit, tier in _VRAM_TIERS:
        if vram_gib < limit:
            selected = tier
            break

    batches = dict(vram_thresholds[selected])
    console.print(
        f"[dim][vram] {properties.name}: {vram_gib:.1f} GiB -> tier '{selected}'.[/dim]"
    )
    listing = ", ".join(
        f"{letter}={batches[letter]}" for letter in _SIZE_ORDER if letter in batches
    )
    console.print(f"[dim][vram] batch sizes: {listing}[/dim]")
    return batches


# %%
def resolve_device(requested: str) -> str:
    """Resolve the compute device, falling back to CPU when CUDA is absent.

    Args:
        requested: The device string from the config (e.g. ``"cuda"``).

    Returns:
        ``"cuda"`` when available and requested, otherwise ``"cpu"``.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA requested but not available - falling back to CPU.")
        return "cpu"
    return requested


# %% Training kwargs
def filter_supported_kwargs(
    kwargs: dict[str, Any],
    console: Console,
) -> dict[str, Any]:
    """Drop train kwargs the installed Ultralytics version does not accept.

    Ultralytics raises on any unknown argument, which would abort the whole
    pipeline. ``clip_grad`` and ``label_smoothing`` are accepted by some
    releases and rejected by others (YOLO26 in particular drops
    ``label_smoothing``), so unsupported keys are dropped with a warning
    instead of crashing.

    Args:
        kwargs: Candidate keyword arguments for ``model.train()``.
        console: ``rich`` console used to report dropped arguments.

    Returns:
        A new dict containing only Ultralytics-supported keys.
    """
    supported: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in kwargs.items():
        if key in DEFAULT_CFG_DICT:
            supported[key] = value
        else:
            dropped.append(key)
    if dropped:
        console.print(
            f"[yellow][train] Ignoring args unsupported by this Ultralytics "
            f"version: {', '.join(sorted(dropped))}.[/yellow]"
        )
    return supported


# %%
def build_train_kwargs(
    config: dict[str, Any],
    size_spec: dict[str, Any],
    batch: int,
    dataset_yaml: Path,
    device: str,
    epochs: int,
    freeze: int,
    cos_lr: bool,
    run_name: str,
    console: Console,
) -> dict[str, Any]:
    """Assemble the keyword arguments for one ``model.train()`` call.

    Args:
        config: The full pipeline configuration.
        size_spec: The per-size hyperparameter block.
        batch: Batch size chosen by the VRAM tier check.
        dataset_yaml: Path to the Ultralytics dataset descriptor.
        device: Resolved compute device string.
        epochs: Epochs for this phase.
        freeze: Number of leading layers to freeze (0 disables freezing).
        cos_lr: Whether to use cosine LR decay (phase 2 only).
        run_name: Run subdirectory name under ``runs/train``.
        console: ``rich`` console used by the kwarg compatibility filter.

    Returns:
        Keyword arguments ready to splat into ``model.train()``.
    """
    training = config["training"]
    augmentation = training["augmentation"]
    recovery = config["recovery"]
    kwargs: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "imgsz": training["imgsz"],
        "batch": batch,
        "lr0": size_spec["lr0"],
        "lrf": size_spec["lrf"],
        "warmup_epochs": size_spec["warmup"],
        "patience": size_spec["patience"],
        "save_period": training["save_period"],
        "device": device,
        "workers": training["workers"],
        "freeze": freeze,
        "cos_lr": cos_lr,
        "close_mosaic": size_spec["close_mosaic"],
        "copy_paste": augmentation["copy_paste"],
        "degrees": augmentation["degrees"],
        "overlap_mask": augmentation["overlap_mask"],
        "hsv_h": augmentation["hsv_h"],
        "hsv_s": augmentation["hsv_s"],
        "hsv_v": augmentation["hsv_v"],
        "mosaic": augmentation["mosaic"],
        "label_smoothing": recovery["label_smoothing"],
        "clip_grad": recovery["clip_grad"],
        "project": str(_RUNS_DIR),
        "name": run_name,
        "exist_ok": True,
        "plots": True,
        "verbose": False,
    }
    return filter_supported_kwargs(kwargs, console)


# %%
def training_phases(
    epochs: int,
    freeze_epochs: int,
    freeze_layers: int,
) -> list[tuple[str, int, int, bool]]:
    """Split the epoch budget into the frozen and unfrozen phases.

    Args:
        epochs: Total epochs to train for.
        freeze_epochs: Epochs to spend in phase 1 with the backbone frozen.
        freeze_layers: Number of leading layers that make up the backbone.

    Returns:
        ``(phase_label, phase_epochs, freeze_layers, cos_lr)`` tuples. Phase 1
        trains flat; phase 2 uses cosine LR decay.
    """
    phase1_epochs = max(0, min(freeze_epochs, epochs))
    phase2_epochs = epochs - phase1_epochs

    phases: list[tuple[str, int, int, bool]] = []
    if phase1_epochs > 0:
        phases.append(("Phase 1 frozen", phase1_epochs, freeze_layers, False))
    if phase2_epochs > 0:
        phases.append(("Phase 2 full", phase2_epochs, 0, True))
    if not phases:
        phases.append(("Full fine-tune", max(1, epochs), 0, True))
    return phases


# %% Progress reporting
class EpochTracker:
    """Accumulates per-epoch state so the progress bar can render live stats.

    Attributes:
        model_name: The model currently being trained.
        phase_label: Human-readable label of the active phase.
        completed: Number of epochs finished across all phases.
        last_stats: Latest formatted metric string.
    """

    def __init__(self, model_name: str) -> None:
        """Initialise an empty tracker.

        Args:
            model_name: The model this tracker follows.
        """
        self.model_name: str = model_name
        self.phase_label: str = ""
        self.completed: int = 0
        self.last_stats: str = "-"

    def begin_phase(self, phase_label: str) -> None:
        """Record that a new training phase has started.

        Args:
            phase_label: Human-readable phase name.
        """
        self.phase_label = phase_label

    def record_epoch(self, metrics: dict[str, Any]) -> None:
        """Fold one finished epoch's metrics into the tracker state.

        Args:
            metrics: The Ultralytics metrics dict for the finished epoch.
        """
        self.completed += 1
        map50 = first_metric(metrics, ("metrics/mAP50(M)", "metrics/mAP50(B)"))
        map50_95 = first_metric(metrics, ("metrics/mAP50-95(M)", "metrics/mAP50-95(B)"))
        self.last_stats = f"mAP50={map50:.3f} mAP50-95={map50_95:.3f}"


# %%
def first_metric(metrics: dict[str, Any], names: tuple[str, ...]) -> float:
    """Return the first metric present under ``names``.

    Args:
        metrics: The Ultralytics metrics dict.
        names: Candidate metric keys, most-preferred first.

    Returns:
        The metric value, or ``0.0`` when none of the keys are present.
    """
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return float(value)
    return 0.0


# %%
def build_training_progress(console: Console) -> Progress:
    """Build the single-line rich progress bar used during training.

    Args:
        console: The ``rich`` console the bar renders to.

    Returns:
        A configured, unstarted :class:`~rich.progress.Progress`.
    """
    return Progress(
        TextColumn("[bold cyan]{task.fields[model]}"),
        TextColumn("[dim][{task.fields[phase]}][/dim]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("{task.fields[stats]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


# %%
def make_epoch_callback(tracker: EpochTracker, progress: Progress, task_id: int) -> Any:
    """Create the ``on_fit_epoch_end`` callback bound to the progress bar.

    A named nested function is returned rather than a lambda so the pipeline
    stays free of lambdas in any callback context.

    Args:
        tracker: The tracker accumulating epoch state.
        progress: The live progress instance.
        task_id: The progress task to advance.

    Returns:
        The Ultralytics callback function.
    """

    def on_fit_epoch_end(trainer: Any) -> None:
        """Advance the progress bar when an epoch finishes.

        Args:
            trainer: The Ultralytics trainer emitting the event.
        """
        tracker.record_epoch(dict(getattr(trainer, "metrics", {}) or {}))
        progress.update(
            task_id,
            completed=tracker.completed,
            phase=tracker.phase_label,
            stats=tracker.last_stats,
        )

    return on_fit_epoch_end


# %%
def clear_cuda_cache(trainer: Any) -> None:
    """Release cached CUDA blocks at the end of each epoch.

    Args:
        trainer: The Ultralytics trainer emitting the event (unused).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# %% Resume / retry
def is_resumable(
    checkpoint: Path,
    epochs: int,
    freeze: int,
    console: Console,
) -> bool:
    """Decide whether ``checkpoint`` can be resumed for this phase.

    Ultralytics restores the epoch budget and freeze setting from the
    checkpoint's saved ``train_args``. Resuming a checkpoint produced under a
    different budget silently trains the wrong schedule, so a mismatch starts
    the phase from scratch instead.

    Args:
        checkpoint: Path to the candidate ``last.pt``.
        epochs: Epoch budget configured for this phase.
        freeze: Number of leading layers to freeze for this phase.
        console: ``rich`` console for the mismatch warning.

    Returns:
        ``True`` when the checkpoint exists and matches this phase's settings.
    """
    if not checkpoint.is_file():
        return False
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint must not abort
        console.print(f"[yellow]Cannot read {checkpoint}: {exc}. Starting fresh.[/yellow]")
        return False

    train_args = payload.get("train_args") or {}
    saved_epochs = train_args.get("epochs")
    saved_freeze = train_args.get("freeze")
    epochs_match = saved_epochs is None or int(saved_epochs) == int(epochs)
    freeze_match = saved_freeze is None or int(saved_freeze) == int(freeze)

    if not (epochs_match and freeze_match):
        console.print(
            f"[yellow]Not resuming {checkpoint.parent.parent.name}: the schedule "
            f"changed (saved epochs={saved_epochs} freeze={saved_freeze}; "
            f"new epochs={epochs} freeze={freeze}). Starting from scratch.[/yellow]"
        )
    return epochs_match and freeze_match


# %%
def train_phase_with_retry(
    name: str,
    phase_index: int,
    weights_in: str,
    epochs: int,
    freeze: int,
    cos_lr: bool,
    config: dict[str, Any],
    size_spec: dict[str, Any],
    batch: int,
    dataset_yaml: Path,
    device: str,
    callbacks: dict[str, Any],
    console: Console,
) -> Path:
    """Train one phase, auto-resuming and retrying on a ``RuntimeError``.

    A crashed attempt leaves ``runs/train/{name}_phase{n}/weights/last.pt``
    behind; the next attempt detects it and resumes rather than restarting.

    Args:
        name: The model key being trained.
        phase_index: 1-based phase number.
        weights_in: Weights to start a fresh (non-resumed) run from.
        epochs: Epochs for this phase.
        freeze: Number of leading layers to freeze.
        cos_lr: Whether to use cosine LR decay.
        config: The full pipeline configuration.
        size_spec: The per-size hyperparameter block.
        batch: Batch size chosen by the VRAM tier check.
        dataset_yaml: Path to the dataset descriptor.
        device: Resolved compute device string.
        callbacks: Mapping of Ultralytics event name to callback function.
        console: ``rich`` console for status output.

    Returns:
        The Ultralytics run directory of the completed phase.

    Raises:
        RuntimeError: When every retry attempt fails.
    """
    max_attempts = int(config["recovery"]["max_train_attempts"])
    run_name = f"{name}_phase{phase_index}"
    checkpoint = _RUNS_DIR / run_name / "weights" / "last.pt"
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        resume = is_resumable(checkpoint, epochs, freeze, console)
        start_weights = str(checkpoint) if resume else weights_in
        try:
            model = YOLO(start_weights)
            for event, callback in callbacks.items():
                model.add_callback(event, callback)

            kwargs = build_train_kwargs(
                config, size_spec, batch, dataset_yaml, device,
                epochs, freeze, cos_lr, run_name, console,
            )
            if resume:
                kwargs["resume"] = True
                kwargs["model"] = str(checkpoint)
                console.print(f"[yellow]Resuming {run_name} from {checkpoint}[/yellow]")

            model.train(**kwargs)
            return Path(model.trainer.save_dir)
        except RuntimeError as exc:
            last_error = exc
            console.print(
                f"[red]RuntimeError during {run_name} "
                f"(attempt {attempt}/{max_attempts}): {exc}[/red]"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(f"{run_name} failed after {max_attempts} attempts.") from last_error


# %% Weights handling
def ensure_pretrained_weights(
    version: str,
    size_letter: str,
    models_dir: Path,
    console: Console,
) -> str:
    """Return a path to the pretrained weights, downloading them if needed.

    ``YOLO("{version}{size}-seg.pt")`` triggers the Ultralytics auto-download
    into the working directory; the file is relocated into ``models_dir`` so
    repeated runs reuse it.

    Args:
        version: Version prefix, e.g. ``yolo12``.
        size_letter: Single size letter, e.g. ``n``.
        models_dir: Directory the pretrained weights are cached in.
        console: ``rich`` console for status output.

    Returns:
        A string path to the local pretrained weights.
    """
    weights_name = f"{version}{size_letter}-seg.pt"
    local_path = models_dir / weights_name
    if local_path.is_file():
        return str(local_path)

    console.print(f"[cyan]Downloading pretrained weights {weights_name}...[/cyan]")
    models_dir.mkdir(parents=True, exist_ok=True)
    YOLO(weights_name)

    downloaded = Path(weights_name)
    if downloaded.is_file():
        shutil.move(str(downloaded), str(local_path))
        return str(local_path)

    # Ultralytics resolved the name from its own cache; hand the bare name back
    # so the trainer resolves it the same way.
    return weights_name


# %%
def extract_val_scores(metrics_obj: Any) -> tuple[float, float]:
    """Pull mAP50 and mAP50-95 out of an Ultralytics validation result.

    Segmentation models are scored on their mask metrics, falling back to box
    metrics when a mask head is absent.

    Args:
        metrics_obj: The object returned by ``model.val()``.

    Returns:
        ``(mAP50, mAP50-95)``.
    """
    for attribute in ("seg", "box"):
        sub = getattr(metrics_obj, attribute, None)
        if sub is not None and hasattr(sub, "map50"):
            return float(sub.map50), float(sub.map)
    return 0.0, 0.0


# %%
def append_training_log(log_path: Path, row: dict[str, Any]) -> None:
    """Append one model's result to ``training_log.csv``.

    The CSV is rewritten from the accumulated set on every call so a partial
    run still leaves a valid file behind. An existing row for the same model is
    replaced rather than duplicated.

    Args:
        log_path: Path to the CSV log.
        row: The per-model row to record.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.is_file():
        existing = pd.read_csv(log_path).to_dict("records")
    else:
        existing = []
    existing = [item for item in existing if item.get("model") != row["model"]]
    existing.append(row)
    pd.DataFrame(existing).to_csv(log_path, index=False)


# %% Per-model training
def train_one_model(
    version: str,
    size_letter: str,
    config: dict[str, Any],
    batches: dict[str, int],
    models_dir: Path,
    dataset_yaml: Path,
    device: str,
    console: Console,
) -> dict[str, Any]:
    """Run two-phase fine-tuning, validation and promotion for one model.

    Args:
        version: Version prefix, e.g. ``yolo12``.
        size_letter: Single size letter, e.g. ``n``.
        config: The full pipeline configuration.
        batches: The VRAM-selected ``{size_letter: batch}`` mapping.
        models_dir: Directory the fine-tuned checkpoint is promoted into.
        dataset_yaml: Path to the dataset descriptor.
        device: Resolved compute device string.
        console: ``rich`` console for status output.

    Returns:
        A summary row for ``training_log.csv``.
    """
    name = model_key(version, size_letter)
    size_spec = config["training"]["sizes"][size_letter]
    batch = int(batches.get(size_letter, size_spec["batch"]))
    freeze_layers = int(config["recovery"]["backbone_freeze_layers"])
    finetuned_suffix = config["naming"]["finetuned_suffix"]

    weights_in = ensure_pretrained_weights(version, size_letter, models_dir, console)
    phases = training_phases(
        int(size_spec["epochs"]), int(size_spec["freeze_epochs"]), freeze_layers,
    )
    total_epochs = sum(phase_epochs for _, phase_epochs, _, _ in phases)

    console.print(
        f"[bold cyan]==> Fine-tuning {name}[/bold cyan] "
        f"(batch={batch}, epochs={total_epochs})"
    )
    tracker = EpochTracker(name)
    start_time = time.perf_counter()
    last_save_dir: Path = _RUNS_DIR / f"{name}_phase1"

    with build_training_progress(console) as progress:
        task_id = progress.add_task(
            "", total=total_epochs, model=name, phase=phases[0][0], stats="-",
        )
        # Named callbacks only - no lambdas anywhere in the callback chain.
        callbacks: dict[str, Any] = {
            "on_fit_epoch_end": make_epoch_callback(tracker, progress, task_id),
            "on_train_epoch_end": clear_cuda_cache,
        }
        for phase_index, (phase_label, phase_epochs, freeze, cos_lr) in enumerate(phases, start=1):
            tracker.begin_phase(phase_label)
            progress.update(task_id, phase=phase_label)
            last_save_dir = train_phase_with_retry(
                name, phase_index, weights_in, phase_epochs, freeze, cos_lr,
                config, size_spec, batch, dataset_yaml, device, callbacks, console,
            )
            # Phase 2 continues from the weights phase 1 produced.
            weights_in = str(last_save_dir / "weights" / "last.pt")

    elapsed_minutes = (time.perf_counter() - start_time) / 60.0

    # Promote the final-phase best checkpoint to the shared models directory.
    best_source = last_save_dir / "weights" / "best.pt"
    finetuned_path = models_dir / f"{name}{finetuned_suffix}.pt"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_source, finetuned_path)
    console.print(f"[green]Saved[/green] {finetuned_path.name}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_model = YOLO(str(finetuned_path))
    val_metrics = val_model.val(
        data=str(dataset_yaml),
        split="val",
        device=device,
        workers=config["training"]["workers"],
        project=str(_RUNS_DIR),
        name=f"{name}_val",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    map50, map50_95 = extract_val_scores(val_metrics)

    console.print(
        f"[green]{name} done[/green] - mAP50={map50:.4f} "
        f"mAP50-95={map50_95:.4f} in {elapsed_minutes:.1f} min"
    )
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": name,
        "version": version,
        "size": size_letter,
        "mAP50": round(map50, 4),
        "mAP50-95": round(map50_95, 4),
        "epochs": tracker.completed,
        "train_time_min": round(elapsed_minutes, 2),
        "status": "ok",
    }


# %%
def print_summary_table(rows: list[dict[str, Any]], console: Console) -> None:
    """Print the final cross-model training summary.

    Args:
        rows: Per-model summary rows.
        console: ``rich`` console to print to.
    """
    table = Table(title="Training Summary", title_style="bold white")
    table.add_column("Model", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("mAP50", justify="right")
    table.add_column("mAP50-95", justify="right")
    table.add_column("Epochs", justify="right")
    table.add_column("Minutes", justify="right")

    for row in rows:
        ok = row["status"] == "ok"
        table.add_row(
            str(row["model"]),
            "[green]OK[/green]" if ok else f"[red]{row['status']}[/red]",
            f"{row['mAP50']:.4f}",
            f"{row['mAP50-95']:.4f}",
            str(row["epochs"]),
            f"{row['train_time_min']:.2f}",
        )
    console.print(table)


# %% Main function
def main(assume_yes: bool) -> int:
    """Discover missing models, confirm, and train them in matrix order.

    Args:
        assume_yes: Skip the interactive confirmation prompt.

    Returns:
        Process exit code - ``0`` on success, ``1`` when any model failed.
    """
    console = Console()
    config = load_config(_CONFIG_PATH)

    models_dir = resolve_path(config["models_dir"])
    dataset_yaml = resolve_path(config["dataset_yaml"])
    finetuned_suffix = config["naming"]["finetuned_suffix"]
    log_path = _HERE / config["recovery"]["training_log_csv"]

    existing = scan_existing_models(models_dir, finetuned_suffix)
    missing = compute_missing(config["model_matrix"], existing)
    print_discovery_report(config["model_matrix"], existing, missing, models_dir, console)

    if not missing:
        console.print("[bold green]Nothing to train.[/bold green]")
        return 0
    if not dataset_yaml.is_file():
        console.print(f"[red]Dataset descriptor not found: {dataset_yaml}[/red]")
        return 1
    if not confirm_proceed(len(missing), assume_yes, console):
        console.print("[yellow]Aborted by user.[/yellow]")
        return 0

    batches = select_batch_sizes(config["vram_thresholds"], console)
    device = resolve_device(config["training"]["device"])
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures = 0
    for version, size_letter in missing:
        name = model_key(version, size_letter)
        try:
            row = train_one_model(
                version, size_letter, config, batches,
                models_dir, dataset_yaml, device, console,
            )
        except Exception as exc:  # noqa: BLE001 - one bad model must not stop the rest
            failures += 1
            console.print(f"[red]{name} failed: {exc}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": name,
                "version": version,
                "size": size_letter,
                "mAP50": 0.0,
                "mAP50-95": 0.0,
                "epochs": 0,
                "train_time_min": 0.0,
                "status": f"failed: {type(exc).__name__}",
            }
        append_training_log(log_path, row)
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary_table(rows, console)
    console.print(f"[dim]Log written to {log_path}[/dim]")
    if failures:
        console.print(f"[red]{failures} model(s) failed.[/red]")
        return 1
    console.print("[bold green]All missing models trained.[/bold green]")
    return 0


# %% Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train every model missing from the jetson_config.yaml matrix.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()
    sys.exit(main(assume_yes=bool(args.yes)))
