# %%
"""Two-phase fine-tuning of YOLO11n-seg, YOLO11s-seg and YOLO11m-seg on cracks.

For every model declared in ``train_config.yaml`` the pipeline:

1. Downloads the pretrained weights via Ultralytics if they are not already in
   ``models/`` (instantiating ``YOLO(weights)`` triggers the download).
2. Runs **phase 1** - the backbone layers are frozen while only the detection
   and mask heads adapt to the crack class.
3. Runs **phase 2** - the whole network is unfrozen and fine-tuned end to end
   with a cosine learning-rate decay down to ``lrf``.
4. Resumes automatically from a crashed ``runs/train/{model}_phase2/...`` run
   and retries up to three times on a ``RuntimeError`` (e.g. a CUDA fault).
5. Copies the best checkpoint into ``models/{model}_crack_finetuned.pt``.
6. Validates the fine-tuned model on the val split and appends a timestamped
   row to ``results/training_summary.csv``.

A VRAM safety check shrinks the per-model batch size on small GPUs before any
training starts. The script is Windows-safe: ``workers=0``, an
``if __name__ == "__main__"`` guard and no lambda functions anywhere.

Run with::

    conda activate cuda_pt
    python train.py
"""

from __future__ import annotations

import os

# CUDA_LAUNCH_BLOCKING=1 makes CUDA kernel launches synchronous so a crash is
# reported at its true call site instead of an unrelated later line. It must be
# set before torch is imported - a stability fix carried over from a prior
# crash that produced misleading stack traces.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# YOLO_VERBOSE=False silences Ultralytics' tqdm batch bars and lowers its
# LOGGER to WARNING, leaving the terminal free for our single-line rich
# progress bar. It must be set before importing ultralytics so the module-level
# `VERBOSE` flag and `TQDM` class pick it up.
os.environ["YOLO_VERBOSE"] = "False"

import shutil
import time
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

# Directory / file locations, all relative to this file.
_BASE_DIR: Path = Path(__file__).parent
_TRAIN_CONFIG_PATH: Path = _BASE_DIR / "train_config.yaml"
_MODELS_DIR: Path = _BASE_DIR / "models"
_RESULTS_DIR: Path = _BASE_DIR / "results"
_DATASET_YAML_PATH: Path = _BASE_DIR / "data" / "processed" / "dataset.yaml"
_RUNS_DIR: Path = _BASE_DIR / "runs" / "train"
_TRAINING_SUMMARY_CSV: Path = _RESULTS_DIR / "training_summary.csv"

# Number of leading layers that make up the YOLO11 backbone. Passing this as
# Ultralytics' ``freeze`` argument freezes the backbone during phase 1.
_BACKBONE_FREEZE_LAYERS: int = 10

# Crash-recovery controls.
_MAX_TRAIN_ATTEMPTS: int = 3      # attempts per phase before giving up
_CACHE_CLEAR_INTERVAL: int = 10   # clear the CUDA cache every N epochs

# Gradient-clipping norm. Ultralytics already clips the gradient L2 norm to this
# value internally on every optimizer step, so it is documented here rather than
# passed as a train() kwarg (the installed Ultralytics rejects an unknown
# ``clip_grad`` argument). The constant keeps the intended value discoverable.
_CLIP_GRAD_NORM: float = 10.0

# VRAM thresholds (in GB) for the automatic batch-size safety net.
_VRAM_HALVE_THRESHOLD_GB: float = 8.0   # below this, halve any batch > 8
_VRAM_FLOOR_THRESHOLD_GB: float = 6.0   # below this, force every batch to 4
_VRAM_FLOOR_BATCH: int = 4
_VRAM_HALVE_BATCH_LIMIT: int = 8


# %%
def load_train_config(config_path: Path) -> dict[str, Any]:
    """Load and parse the training configuration YAML file.

    Args:
        config_path: Path to ``train_config.yaml``.

    Returns:
        The parsed configuration as a nested dict.
    """
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# %%
def resolve_device(requested: str) -> str:
    """Resolve the compute device, falling back to CPU when CUDA is absent.

    Args:
        requested: The device string requested in the config (e.g. ``"cuda"``).

    Returns:
        ``"cuda"`` if available and requested, otherwise ``"cpu"``.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA requested but not available - falling back to CPU.")
        return "cpu"
    return requested


# %%
def select_models_to_train(
    config: dict[str, Any],
    console: Console,
) -> list[str]:
    """Resolve which models to fine-tune from the ``models_to_train`` key.

    The optional ``models_to_train`` list filters the models defined under
    ``models:``. When the key is absent every defined model is trained. Any
    requested name that is not defined under ``models:`` is reported and
    ignored. The chosen and skipped models are printed at startup.

    Args:
        config: The parsed training configuration.
        console: ``rich`` console for status output.

    Returns:
        The ordered list of model names to fine-tune.
    """
    defined: list[str] = list(config["models"].keys())
    requested = config.get("models_to_train")

    if requested is None:
        selected = list(defined)
    else:
        unknown = [name for name in requested if name not in defined]
        if unknown:
            console.print(
                f"[yellow]Ignoring unknown models_to_train entries (not under "
                f"'models:'): {', '.join(unknown)}.[/yellow]"
            )
        # Preserve the order in which models are defined under `models:`.
        selected = [name for name in defined if name in set(requested)]

    skipped = [name for name in defined if name not in selected]
    console.print(f"[bold]Training:[/bold] {', '.join(selected) or '(none)'}")
    console.print(f"[dim]Skipping:[/dim] {', '.join(skipped) or '(none)'}")
    return selected


# %%
def apply_vram_safety(models: dict[str, dict[str, Any]], console: Console) -> None:
    """Shrink per-model batch sizes to fit the detected GPU memory.

    The configured batch sizes target an 8 GB RTX 4070. On a smaller GPU they
    would trigger out-of-memory crashes, so the batch values are lowered in
    place before any training begins:

    * VRAM below 6 GB: every model's batch is forced to 4.
    * VRAM below 8 GB: any model whose batch exceeds 8 is halved.

    Args:
        models: The per-model configuration mapping, modified in place.
        console: ``rich`` console for the emitted warnings.
    """
    if not torch.cuda.is_available():
        return

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    console.print(f"[dim][train] Detected {vram_gb:.1f} GB of GPU memory.[/dim]")

    if vram_gb < _VRAM_FLOOR_THRESHOLD_GB:
        for name, spec in models.items():
            if int(spec["batch"]) != _VRAM_FLOOR_BATCH:
                console.print(
                    f"[yellow]VRAM < {_VRAM_FLOOR_THRESHOLD_GB:.0f} GB: forcing "
                    f"{name} batch {spec['batch']} -> {_VRAM_FLOOR_BATCH}.[/yellow]"
                )
                spec["batch"] = _VRAM_FLOOR_BATCH
        return

    if vram_gb < _VRAM_HALVE_THRESHOLD_GB:
        for name, spec in models.items():
            batch = int(spec["batch"])
            if batch > _VRAM_HALVE_BATCH_LIMIT:
                halved = max(1, batch // 2)
                console.print(
                    f"[yellow]VRAM < {_VRAM_HALVE_THRESHOLD_GB:.0f} GB: halving "
                    f"{name} batch {batch} -> {halved}.[/yellow]"
                )
                spec["batch"] = halved


# %%
def ensure_pretrained_weights(weights_name: str, console: Console) -> str:
    """Return a path to the pretrained weights, downloading them if necessary.

    If ``models/{weights_name}`` already exists it is used directly. Otherwise
    ``YOLO(weights_name)`` is instantiated, which makes Ultralytics download the
    checkpoint into the current directory; the file is then relocated into
    ``models/`` so repeated runs reuse it.

    Args:
        weights_name: The pretrained weights filename (e.g. ``yolo11s-seg.pt``).
        console: ``rich`` console for status output.

    Returns:
        A string path to the local pretrained weights.
    """
    local_path = _MODELS_DIR / weights_name
    if local_path.is_file():
        return str(local_path)

    console.print(f"[cyan]Downloading pretrained weights {weights_name}...[/cyan]")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Instantiating YOLO with a bare name triggers the Ultralytics auto-download.
    YOLO(weights_name)

    downloaded = Path(weights_name)
    if downloaded.is_file():
        shutil.move(str(downloaded), str(local_path))
        return str(local_path)

    # Ultralytics resolved the name from its cache; fall back to the bare name so
    # the trainer can resolve it the same way.
    return weights_name


# %%
def filter_supported_kwargs(
    kwargs: dict[str, Any],
    console: Console,
) -> dict[str, Any]:
    """Drop train kwargs the installed Ultralytics version does not accept.

    Ultralytics raises a ``SyntaxError`` on any unknown argument, which would
    crash the whole pipeline. ``label_smoothing`` (and a hypothetical
    ``clip_grad``) are accepted by some releases and rejected by others; this
    filter keeps the pipeline portable by passing through only the arguments
    present in the active default config and warning about the rest.

    Args:
        kwargs: The candidate keyword arguments for ``model.train()``.
        console: ``rich`` console used to warn about dropped arguments.

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
class EpochTracker:
    """Accumulates per-epoch training metrics for the live ``rich`` table."""

    def __init__(self, display_name: str) -> None:
        """Initialise an empty tracker for one model.

        Args:
            display_name: Human-readable model name shown in the table title.
        """
        self.display_name: str = display_name
        self.rows: list[dict[str, float]] = []
        self.phase_label: str = ""
        self.epoch_offset: int = 0

    # %%
    def begin_phase(self, phase_label: str) -> None:
        """Record that a new training phase has started.

        Args:
            phase_label: Human-readable phase name (e.g. ``"Phase 1"``).
        """
        self.phase_label = phase_label
        self.epoch_offset = len(self.rows)

    # %%
    def record(self, trainer: Any) -> None:
        """Capture one epoch's metrics from an Ultralytics trainer.

        Args:
            trainer: The Ultralytics trainer object passed to the callback.
        """
        metrics: dict[str, Any] = dict(getattr(trainer, "metrics", {}) or {})

        loss_items = trainer.label_loss_items(getattr(trainer, "tloss", None), prefix="train")
        train_loss = float(sum(float(value) for value in loss_items.values()))

        val_losses = [float(v) for k, v in metrics.items() if str(k).startswith("val/")]
        val_loss = float(sum(val_losses)) if val_losses else float("nan")

        map50 = 0.0
        for key, value in metrics.items():
            text = str(key)
            if "mAP50" in text and "95" not in text:
                map50 = float(value)
                break

        learning_rate = 0.0
        lr_map = getattr(trainer, "lr", None)
        if isinstance(lr_map, dict) and lr_map:
            learning_rate = float(next(iter(lr_map.values())))

        self.rows.append(
            {
                "phase": self.phase_label,
                "epoch": float(self.epoch_offset + int(getattr(trainer, "epoch", 0)) + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "map50": map50,
                "lr": learning_rate,
            }
        )

    # %%
    def best_map50(self) -> float:
        """Return the highest mAP50 recorded across every epoch.

        Returns:
            The best per-epoch mAP50, or ``0.0`` if no rows were recorded.
        """
        if not self.rows:
            return 0.0
        return max(float(row["map50"]) for row in self.rows)

    # %%
    def last_stats_text(self) -> str:
        """Render the most recent epoch's stats as a single-line string.

        Returns:
            A compact ``loss=... mAP50=... LR=...`` summary, or ``"-"`` if no
            epoch has been recorded yet.
        """
        if not self.rows:
            return "-"
        row = self.rows[-1]
        return (
            f"loss={row['train_loss']:.4f} "
            f"mAP50={row['map50']:.4f} "
            f"LR={row['lr']:.5f}"
        )


# %%
def build_training_progress(console: Console) -> Progress:
    """Construct the single-line ``rich`` progress bar used during training.

    The bar shows the model name, current phase, completed/total epochs, a
    percentage, ETA, elapsed time and the latest epoch's loss / mAP50 / LR.
    Ultralytics' own batch-level output is silenced via ``YOLO_VERBOSE=False``
    so this single line is the only thing rendered during training.

    Args:
        console: The ``rich`` console the progress bar writes to.

    Returns:
        A configured ``rich`` :class:`~rich.progress.Progress` instance ready to
        be used as a context manager.
    """
    return Progress(
        TextColumn("[bold cyan]{task.fields[model]}[/]"),
        TextColumn("[dim][{task.fields[phase]}][/dim]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        TextColumn("elapsed"),
        TimeElapsedColumn(),
        TextColumn("{task.fields[stats]}"),
        console=console,
        refresh_per_second=4,
        transient=False,
    )


# %%
def build_train_kwargs(
    model_spec: dict[str, Any],
    top_config: dict[str, Any],
    device: str,
    epochs: int,
    freeze: int,
    cos_lr: bool,
    name: str,
    console: Console,
) -> dict[str, Any]:
    """Assemble the keyword arguments for a single ``model.train()`` call.

    Args:
        model_spec: The per-model hyperparameter block from the config.
        top_config: The top-level config holding ``workers`` and ``save_period``.
        device: The resolved compute device string.
        epochs: Number of epochs for this training phase.
        freeze: Number of leading layers to freeze (0 disables freezing).
        cos_lr: Whether to use cosine learning-rate decay (phase 2 only).
        name: Run name (a subdirectory created under ``runs/train``).
        console: ``rich`` console used by the kwarg compatibility filter.

    Returns:
        A dict of keyword arguments ready to splat into ``model.train()``.
    """
    kwargs: dict[str, Any] = {
        "data": str(_DATASET_YAML_PATH),
        "epochs": epochs,
        "imgsz": model_spec["imgsz"],
        "batch": model_spec["batch"],
        "lr0": model_spec["lr0"],
        "lrf": model_spec["lrf"],
        "warmup_epochs": model_spec["warmup_epochs"],
        "patience": model_spec["patience"],
        "save_period": top_config["save_period"],
        "device": device,
        "workers": top_config["workers"],
        "freeze": freeze,
        "cos_lr": cos_lr,
        "close_mosaic": model_spec["close_mosaic"],
        "copy_paste": model_spec["copy_paste"],
        "degrees": model_spec["degrees"],
        "label_smoothing": model_spec["label_smoothing"],
        "overlap_mask": model_spec["overlap_mask"],
        "hsv_h": model_spec["hsv_h"],
        "hsv_s": model_spec["hsv_s"],
        "hsv_v": model_spec["hsv_v"],
        "mosaic": model_spec["mosaic"],
        "clip_grad": _CLIP_GRAD_NORM,
        "project": str(_RUNS_DIR),
        "name": name,
        "exist_ok": True,
        "plots": True,
        "verbose": False,
    }
    return filter_supported_kwargs(kwargs, console)


# %%
def training_phases(epochs: int, freeze_epochs: int) -> list[tuple[str, int, int, bool]]:
    """Split the total epoch budget into the frozen and unfrozen phases.

    Args:
        epochs: Total number of epochs to train for.
        freeze_epochs: Epochs to spend in phase 1 with the backbone frozen.

    Returns:
        A list of ``(phase_label, phase_epochs, freeze_layers, cos_lr)`` tuples.
        Phase 1 trains with a flat schedule; phase 2 uses cosine LR decay.
    """
    phase1_epochs = max(0, min(freeze_epochs, epochs))
    phase2_epochs = epochs - phase1_epochs

    phases: list[tuple[str, int, int, bool]] = []
    if phase1_epochs > 0:
        phases.append(("Phase 1 frozen", phase1_epochs, _BACKBONE_FREEZE_LAYERS, False))
    if phase2_epochs > 0:
        phases.append(("Phase 2 full", phase2_epochs, 0, True))
    if not phases:
        phases.append(("Full fine-tune", max(1, epochs), 0, True))
    return phases


# %%
def extract_val_scores(metrics_obj: Any) -> tuple[float, float]:
    """Pull mAP50 and mAP50-95 out of an Ultralytics validation result.

    Segmentation models are scored on their mask metrics, falling back to box
    metrics when a mask head is absent.

    Args:
        metrics_obj: The object returned by ``model.val()``.

    Returns:
        A tuple of ``(mAP50, mAP50-95)``.
    """
    for attribute in ("seg", "box"):
        sub = getattr(metrics_obj, attribute, None)
        if sub is not None and hasattr(sub, "map50"):
            return float(sub.map50), float(sub.map)
    return 0.0, 0.0


# %%
def make_epoch_callback(
    tracker: EpochTracker,
    progress: Progress,
    task_id: int,
) -> Any:
    """Create the ``on_fit_epoch_end`` callback bound to the live progress bar.

    A named nested function is returned rather than a lambda so the pipeline
    contains no lambdas in any callback or multiprocessing context.

    Args:
        tracker: The :class:`EpochTracker` accumulating epoch rows.
        progress: The ``rich`` :class:`~rich.progress.Progress` instance.
        task_id: The task id within ``progress`` to advance on every epoch.

    Returns:
        A callable accepting the Ultralytics trainer object.
    """

    def on_fit_epoch_end(trainer: Any) -> None:
        """Record the finished epoch and advance the progress bar by one."""
        tracker.record(trainer)
        progress.update(
            task_id,
            advance=1,
            phase=tracker.rows[-1]["phase"],
            stats=tracker.last_stats_text(),
        )

    return on_fit_epoch_end


# %%
def make_batch_heartbeat_callbacks(
    tracker: EpochTracker,
    progress: Progress,
    task_id: int,
) -> tuple[Any, Any]:
    """Create ``on_train_epoch_start`` and ``on_train_batch_end`` callbacks.

    With ``YOLO_VERBOSE=False`` Ultralytics emits no batch-level output, so a
    long epoch (e.g. ~2 min for YOLO11m at imgsz=640) looks frozen between
    epoch-boundary updates. These callbacks push a per-batch ``batch X/Y
    loss=...`` heartbeat into the progress bar's ``stats`` field so the line
    visibly moves throughout each epoch. Rich's ``refresh_per_second`` throttles
    the actual terminal redraws to avoid flicker.

    A mutable closure state dict tracks the current batch index because the
    Ultralytics trainer does not expose it as an attribute.

    Args:
        tracker: The :class:`EpochTracker` whose last-epoch stats are appended.
        progress: The ``rich`` :class:`~rich.progress.Progress` instance.
        task_id: The task id within ``progress`` to update on every batch.

    Returns:
        A ``(on_train_epoch_start, on_train_batch_end)`` pair of callbacks.
    """
    state: dict[str, int] = {"batch": 0, "total": 0}

    def on_train_epoch_start(trainer: Any) -> None:
        """Reset the per-epoch batch counter and capture the total batch count."""
        state["batch"] = 0
        loader = getattr(trainer, "train_loader", None)
        try:
            state["total"] = int(len(loader)) if loader is not None else 0
        except TypeError:
            state["total"] = 0

    def on_train_batch_end(trainer: Any) -> None:
        """Push a ``batch X/Y loss=...`` heartbeat into the progress bar."""
        state["batch"] += 1
        total = state["total"]
        live = f"batch {state['batch']}/{total}" if total else f"batch {state['batch']}"

        tloss = getattr(trainer, "tloss", None)
        if tloss is not None:
            try:
                cur_loss = float(tloss.sum()) if hasattr(tloss, "sum") else float(sum(tloss))
                live = f"{live} loss={cur_loss:.4f}"
            except Exception:  # noqa: BLE001 - heartbeat must never crash training
                pass

        last_completed = tracker.last_stats_text()
        if last_completed != "-":
            stats = f"{live} | last: {last_completed}"
        else:
            stats = live
        progress.update(task_id, stats=stats)

    return on_train_epoch_start, on_train_batch_end


# %%
def clear_cuda_cache(trainer: Any) -> None:
    """Periodically release cached CUDA memory to curb fragmentation.

    Registered on the ``on_train_epoch_end`` event; the cache is emptied once
    every ``_CACHE_CLEAR_INTERVAL`` epochs. Long runs on an 8 GB GPU can
    otherwise fragment VRAM and trigger spurious out-of-memory errors.

    Args:
        trainer: The Ultralytics trainer object passed to the callback.
    """
    if not torch.cuda.is_available():
        return
    epoch = int(getattr(trainer, "epoch", 0))
    if (epoch + 1) % _CACHE_CLEAR_INTERVAL == 0:
        torch.cuda.empty_cache()


# %%
def make_recovery_callback(model_name: str) -> Any:
    """Create an ``on_model_save`` callback that mirrors ``last.pt`` to models/.

    Every time Ultralytics writes a checkpoint, the freshest ``last.pt`` is
    copied to ``models/{model_name}_crack_recovery.pt`` so a crash never costs
    more than one save interval of progress. A named nested function is
    returned rather than a lambda.

    Args:
        model_name: Name of the model currently training.

    Returns:
        A callable accepting the Ultralytics trainer object.
    """

    def on_model_save(trainer: Any) -> None:
        """Copy the freshest checkpoint into the model's recovery slot."""
        last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
        if last_pt.is_file():
            shutil.copy2(last_pt, _MODELS_DIR / f"{model_name}_crack_recovery.pt")

    return on_model_save


# %%
def is_resumable(
    checkpoint: Path,
    epochs: int,
    freeze: int,
    console: Console | None = None,
) -> bool:
    """Return whether a checkpoint is from a crashed run that still matches the config.

    Ultralytics records ``epoch == -1`` once a run finishes cleanly; a
    non-negative epoch below the budget means the run stopped early (e.g. a
    crash) and ``model.train(resume=True)`` can continue it.

    The checkpoint must also match the current config. When ``resume=True``
    Ultralytics' :meth:`check_resume` replaces ``self.args`` with the
    checkpoint's saved ``train_args`` and only ``imgsz``, ``batch``, ``device``
    and ``close_mosaic`` from the new overrides survive - so a stale checkpoint
    silently overrides a freshly reduced ``epochs`` value. To prevent that,
    this helper refuses to resume when the saved ``epochs`` or ``freeze`` no
    longer matches the value about to be used, forcing a fresh start instead.

    Args:
        checkpoint: Path to a candidate ``last.pt`` checkpoint.
        epochs: Total epoch budget configured for that phase.
        freeze: Number of leading layers to freeze for that phase.
        console: Optional ``rich`` console for a "config changed" warning.

    Returns:
        ``True`` if the checkpoint exists, represents an unfinished run, and was
        produced under the same epoch budget and freeze configuration.
    """
    if not checkpoint.is_file():
        return False
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - a corrupt checkpoint is simply not resumable
        return False
    if not isinstance(ckpt, dict):
        return False

    epoch = int(ckpt.get("epoch", -1))
    if not (0 <= epoch < epochs - 1):
        return False

    train_args = ckpt.get("train_args") or {}
    saved_epochs = train_args.get("epochs")
    saved_freeze = train_args.get("freeze")

    epochs_match = saved_epochs is None or int(saved_epochs) == int(epochs)
    freeze_match = saved_freeze is None or int(saved_freeze) == int(freeze)

    if not (epochs_match and freeze_match) and console is not None:
        console.print(
            f"[yellow]Refusing to resume {checkpoint.parent.parent.name}: config "
            f"changed (saved epochs={saved_epochs} freeze={saved_freeze}; "
            f"new epochs={epochs} freeze={freeze}). Starting phase from scratch.[/yellow]"
        )
    return epochs_match and freeze_match


# %%
def train_phase_with_retry(
    model_name: str,
    phase_index: int,
    weights_in: str,
    epochs: int,
    freeze: int,
    cos_lr: bool,
    model_spec: dict[str, Any],
    top_config: dict[str, Any],
    device: str,
    callbacks: dict[str, Any],
    console: Console,
) -> Path:
    """Train a single phase, auto-resuming and retrying on a ``RuntimeError``.

    Ultralytics writes all run artifacts under ``runs/train/{name}``. If a
    crashed ``runs/train/{model_name}_phase{n}/weights/last.pt`` is found,
    training resumes from it. On a ``RuntimeError`` (e.g. a transient CUDA
    fault) the CUDA cache is cleared and the phase is retried up to
    ``_MAX_TRAIN_ATTEMPTS`` times - each retry resuming from the checkpoint the
    crashed attempt left behind.

    Args:
        model_name: Name of the model being trained.
        phase_index: 1-based training phase number.
        weights_in: Weights to start a fresh (non-resumed) run from.
        epochs: Number of epochs for this phase.
        freeze: Number of leading layers to freeze (0 disables freezing).
        cos_lr: Whether to use cosine LR decay for this phase.
        model_spec: The per-model hyperparameter block.
        top_config: The top-level config block.
        device: The resolved compute device string.
        callbacks: Mapping of Ultralytics event name to callback function.
        console: ``rich`` console for status output.

    Returns:
        The Ultralytics run directory of the completed phase.

    Raises:
        RuntimeError: If every retry attempt fails.
    """
    run_name = f"{model_name}_phase{phase_index}"
    checkpoint = _RUNS_DIR / run_name / "weights" / "last.pt"
    last_error: BaseException | None = None

    for attempt in range(1, _MAX_TRAIN_ATTEMPTS + 1):
        resume = is_resumable(checkpoint, epochs, freeze, console)
        start_weights = str(checkpoint) if resume else weights_in
        try:
            model = YOLO(start_weights)
            for event, callback in callbacks.items():
                model.add_callback(event, callback)

            kwargs = build_train_kwargs(
                model_spec, top_config, device, epochs, freeze, cos_lr, run_name, console
            )
            if resume:
                # Resume the crashed run; Ultralytics restores the remaining
                # state (optimizer, epoch, EMA) from the checkpoint's args.
                kwargs["resume"] = True
                kwargs["model"] = str(checkpoint)
                console.print(
                    f"[yellow]Resuming {run_name} from {checkpoint}[/yellow]"
                )

            model.train(**kwargs)
            return Path(model.trainer.save_dir)
        except RuntimeError as exc:
            last_error = exc
            console.print(
                f"[red]RuntimeError during {run_name} "
                f"(attempt {attempt}/{_MAX_TRAIN_ATTEMPTS}): {exc}[/red]"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # The failed attempt leaves last.pt behind; the next loop iteration
            # detects it via is_resumable() and resumes automatically.

    raise RuntimeError(
        f"{run_name} failed after {_MAX_TRAIN_ATTEMPTS} attempts."
    ) from last_error


# %%
def force_restart_run_dirs(model_name: str, console: Console) -> None:
    """Delete every per-phase run directory for ``model_name`` and its recovery slot.

    Called when the model's ``force_restart`` flag is true. Wiping the
    ``runs/train/{model_name}_phase*`` directories guarantees Ultralytics has no
    ``last.pt`` to resume from, so the next ``train_phase_with_retry`` call
    starts cleanly from the pretrained weights. The mirrored recovery checkpoint
    is also removed so it cannot mask a real fresh start.

    Args:
        model_name: The config key (e.g. ``yolo11s-seg``).
        console: ``rich`` console for status output.
    """
    targets = sorted(_RUNS_DIR.glob(f"{model_name}_phase*"))
    recovery = _MODELS_DIR / f"{model_name}_crack_recovery.pt"

    if not targets and not recovery.is_file():
        console.print(
            f"[dim]force_restart=true for {model_name}: nothing stale to remove.[/dim]"
        )
        return

    for run_dir in targets:
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
            console.print(f"[yellow]force_restart: removed {run_dir}[/yellow]")
    if recovery.is_file():
        recovery.unlink()
        console.print(f"[yellow]force_restart: removed {recovery}[/yellow]")


# %%
def train_one_model(
    model_name: str,
    config: dict[str, Any],
    device: str,
    console: Console,
) -> dict[str, Any]:
    """Run two-phase fine-tuning and validation for a single model.

    Args:
        model_name: The config key (e.g. ``yolo11s-seg``).
        config: The parsed training configuration.
        device: The resolved compute device string.
        console: ``rich`` console for status output.

    Returns:
        A summary dict with the model name, best mAP50, best mAP50-95, epochs
        actually trained, training time in minutes and an ISO timestamp.
    """
    model_spec: dict[str, Any] = config["models"][model_name]
    if bool(model_spec.get("force_restart", False)):
        force_restart_run_dirs(model_name, console)

    weights_in = ensure_pretrained_weights(model_spec["weights"], console)
    phases = training_phases(model_spec["epochs"], model_spec["freeze_epochs"])

    console.print(f"[bold cyan]==> Fine-tuning {model_name}[/bold cyan]")
    tracker = EpochTracker(model_name)
    start_time = time.perf_counter()
    last_save_dir: Path = _RUNS_DIR / f"{model_name}_phase1"
    total_epochs = sum(phase_epochs for _, phase_epochs, _, _ in phases)

    with build_training_progress(console) as progress:
        task_id = progress.add_task(
            "",
            total=total_epochs,
            model=model_name,
            phase=phases[0][0],
            stats="-",
        )
        # Callbacks: single-line progress bar (epoch + intra-epoch heartbeat),
        # periodic CUDA cache clearing and recovery-checkpoint mirroring. All
        # are named functions - no lambdas.
        on_train_epoch_start, on_train_batch_end = make_batch_heartbeat_callbacks(
            tracker, progress, task_id
        )
        callbacks: dict[str, Any] = {
            "on_fit_epoch_end": make_epoch_callback(tracker, progress, task_id),
            "on_train_epoch_start": on_train_epoch_start,
            "on_train_batch_end": on_train_batch_end,
            "on_train_epoch_end": clear_cuda_cache,
            "on_model_save": make_recovery_callback(model_name),
        }
        for phase_index, (phase_label, phase_epochs, freeze, cos_lr) in enumerate(phases, start=1):
            tracker.begin_phase(phase_label)
            progress.update(task_id, phase=phase_label)
            last_save_dir = train_phase_with_retry(
                model_name,
                phase_index,
                weights_in,
                phase_epochs,
                freeze,
                cos_lr,
                model_spec,
                config,
                device,
                callbacks,
                console,
            )
            # Phase 2 resumes from the weights produced by phase 1.
            weights_in = str(last_save_dir / "weights" / "last.pt")

    elapsed_minutes = (time.perf_counter() - start_time) / 60.0

    # Promote the final-phase checkpoints to stable names inside models/.
    best_source = last_save_dir / "weights" / "best.pt"
    last_source = last_save_dir / "weights" / "last.pt"
    finetuned_path = _MODELS_DIR / f"{model_name}_crack_finetuned.pt"
    last_path = _MODELS_DIR / f"{model_name}_crack_last.pt"
    shutil.copy2(best_source, finetuned_path)
    shutil.copy2(last_source, last_path)
    console.print(f"[green]Saved[/green] {finetuned_path.name} and {last_path.name}")

    # Validate the fine-tuned model on the val split immediately.
    val_model = YOLO(str(finetuned_path))
    val_metrics = val_model.val(
        data=str(_DATASET_YAML_PATH),
        split="val",
        device=device,
        workers=config["workers"],
        project=str(_RUNS_DIR),
        name=f"{model_name}_val",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    map50, map50_95 = extract_val_scores(val_metrics)

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "mAP50": round(map50, 4),
        "mAP50-95": round(map50_95, 4),
        "epochs": len(tracker.rows),
        "train_time_min": round(elapsed_minutes, 2),
    }
    append_training_summary(summary)
    console.print(
        f"[green]{model_name} done[/green] - mAP50={map50:.4f} "
        f"mAP50-95={map50_95:.4f} in {elapsed_minutes:.1f} min"
    )
    return summary


# %%
def append_training_summary(summary: dict[str, Any]) -> None:
    """Append one model's results to ``results/training_summary.csv``.

    The CSV is rewritten from the full accumulated set on every call so a
    partial run still leaves a valid file behind.

    Args:
        summary: The per-model summary dict to append.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if _TRAINING_SUMMARY_CSV.is_file():
        existing = pd.read_csv(_TRAINING_SUMMARY_CSV).to_dict("records")
    else:
        existing = []
    existing = [row for row in existing if row.get("model") != summary["model"]]
    existing.append(summary)
    pd.DataFrame(existing).to_csv(_TRAINING_SUMMARY_CSV, index=False)


# %%
def print_comparison_table(summaries: list[dict[str, Any]], console: Console) -> None:
    """Print the final cross-model comparison table using ``rich``.

    Args:
        summaries: Per-model summary dicts produced by :func:`train_one_model`.
        console: The ``rich`` console to print to.
    """
    table = Table(title="Fine-tuning Comparison", title_style="bold white")
    table.add_column("Model", style="bold")
    table.add_column("Best mAP50", justify="right")
    table.add_column("Best mAP50-95", justify="right")
    table.add_column("Epochs trained", justify="right")
    table.add_column("Training time (min)", justify="right")

    for summary in summaries:
        table.add_row(
            str(summary["model"]),
            f"{summary['mAP50']:.4f}",
            f"{summary['mAP50-95']:.4f}",
            str(summary["epochs"]),
            f"{summary['train_time_min']:.2f}",
        )
    console.print(table)


# %%
def run_training() -> None:
    """Load the configuration and fine-tune every model declared in the config."""
    console = Console()
    console.print("[bold]Crack Detection - Model Fine-tuning[/bold]")

    if not _DATASET_YAML_PATH.is_file():
        console.print(
            f"[red]Dataset not found at {_DATASET_YAML_PATH}.[/red] "
            "Run 'python prepare_data.py' first."
        )
        return

    config = load_train_config(_TRAIN_CONFIG_PATH)
    device = resolve_device(config["device"])
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_models_to_train(config, console)
    if not selected:
        console.print("[red]No models selected to train - nothing to do.[/red]")
        return

    # Only shrink batch sizes for the models that will actually train.
    selected_specs = {name: config["models"][name] for name in selected}
    apply_vram_safety(selected_specs, console)

    summaries: list[dict[str, Any]] = []
    for model_name in selected:
        summary = train_one_model(model_name, config, device, console)
        summaries.append(summary)

    print_comparison_table(summaries, console)
    console.print(f"[green]Training summary written to[/green] {_TRAINING_SUMMARY_CSV}")
    console.print("[bold green]All models fine-tuned.[/bold green]")


# %%
def main() -> None:
    """Entry point for the fine-tuning pipeline."""
    run_training()


# %%
if __name__ == "__main__":
    main()
