# %%
"""export_all.py - export every fine-tuned model to ONNX, TRT FP16 and TRT INT8.

Three phases run in strict order and each phase completes for **all** models
before the next one begins:

* **Phase 1** - ONNX FP32 for every ``*_crack_finetuned.pt``.
* **Phase 2** - TensorRT FP16 for every model that produced an ONNX graph.
* **Phase 3** - TensorRT INT8, last, using the calibration caches built by
  ``calibrate_int8.py`` from the CrackSeg9k test split.

A failure in one model is logged and the phase moves on; it never aborts the
run. INT8 failures additionally capture a full traceback in
``int8_failures.log``.

Usage:
    conda activate cuda_pt
    python jetson_pipeline/export_all.py
    python jetson_pipeline/export_all.py --phases 1 2   # ONNX + FP16 only
"""

from __future__ import annotations

import os

# CUDA_LAUNCH_BLOCKING=1 makes kernel launches synchronous so a CUDA fault is
# reported at its true call site. It must be set before torch is imported.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# %% Imports
import argparse
import faulthandler
import shutil
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from ultralytics import YOLO

from calibrate_int8 import (
    cache_path_for,
    collect_calibration_images,
    ensure_calibration_cache,
    patch_trt_nptype,
)

# %% Config
_HERE: Path = Path(__file__).parent
_PROJECT_ROOT: Path = _HERE.parent
_CONFIG_PATH: Path = _HERE / "jetson_config.yaml"

# Canonical size ordering, shared with the rest of the pipeline.
_SIZE_ORDER: tuple[str, ...] = ("n", "s", "m", "l", "x")

# Phase identifiers, in the only order they may run.
_PHASE_ONNX: int = 1
_PHASE_TRT_FP16: int = 2
_PHASE_TRT_INT8: int = 3
_ALL_PHASES: tuple[int, ...] = (_PHASE_ONNX, _PHASE_TRT_FP16, _PHASE_TRT_INT8)

# Phase number paired with the results key it writes.
_PHASE_KEYS: tuple[tuple[int, str], ...] = (
    (_PHASE_ONNX, "onnx"),
    (_PHASE_TRT_FP16, "trt_fp16"),
    (_PHASE_TRT_INT8, "trt_int8"),
)

# Signals indicating a catastrophic native failure inside the TensorRT builder.
# Only these are settable on both Windows and Linux.
_FATAL_SIGNALS: tuple[str, ...] = ("SIGSEGV", "SIGABRT", "SIGFPE", "SIGILL")

# TensorRT 10 removed trt.nptype while Ultralytics still calls it. The patch
# must be installed before any engine export or YOLO(.engine) load.
patch_trt_nptype()


# %% Signal handling
class FatalSignalError(RuntimeError):
    """Raised when a fatal native signal interrupts an export.

    Converting the signal into a Python exception lets the per-model
    try/except log the failure and continue with the next model instead of the
    process dying inside the TensorRT builder.
    """


# %%
def install_fatal_signal_handlers(console: Console) -> None:
    """Convert fatal native signals into :class:`FatalSignalError`.

    ``faulthandler`` is enabled alongside so the native stack is dumped before
    the exception is raised. Signals the platform refuses to override are
    skipped - that is a platform limit, not an error.

    Args:
        console: ``rich`` console for the status line.
    """
    faulthandler.enable()
    installed: list[str] = []

    def _raise_fatal(signal_number: int, frame: Any) -> None:
        """Translate a fatal signal into a Python exception.

        Args:
            signal_number: The signal that fired.
            frame: The interrupted stack frame (unused).

        Raises:
            FatalSignalError: Always.
        """
        raise FatalSignalError(f"fatal signal {signal_number} during export")

    for name in _FATAL_SIGNALS:
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            signal.signal(number, _raise_fatal)
            installed.append(name)
        except (OSError, ValueError, RuntimeError):
            continue
    console.print(
        f"[dim][export] fatal-signal handlers: {', '.join(installed) or 'none'}[/dim]"
    )


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


# %% Discovery
def parse_model_stem(stem: str, finetuned_suffix: str) -> tuple[str, str] | None:
    """Parse a checkpoint stem into ``(version, size_letter)``.

    Args:
        stem: Filename without extension.
        finetuned_suffix: Stem suffix marking a fine-tuned checkpoint.

    Returns:
        ``(version, size_letter)``, or ``None`` when the stem does not match.
    """
    if not stem.endswith(finetuned_suffix):
        return None
    head = stem[: -len(finetuned_suffix)]
    if not head.endswith("-seg"):
        return None
    head = head[: -len("-seg")]
    if not head or head[-1] not in _SIZE_ORDER:
        return None
    version = head[:-1]
    return (version, head[-1]) if version else None


# %%
def model_sort_key(model: dict[str, Any]) -> tuple[str, int]:
    """Sort key placing models in version then size order.

    Args:
        model: A model dict from :func:`discover_exportable_models`.

    Returns:
        ``(version, size_index)``.
    """
    return str(model["version"]), _SIZE_ORDER.index(str(model["size"]))


# %%
def discover_exportable_models(
    models_dir: Path,
    finetuned_suffix: str,
) -> list[dict[str, Any]]:
    """Find every fine-tuned checkpoint that can be exported.

    Args:
        models_dir: Directory holding the trained weights.
        finetuned_suffix: Stem suffix marking a fine-tuned checkpoint.

    Returns:
        One dict per model with ``name``, ``version``, ``size`` and ``pt``
        keys, sorted by version then by :data:`_SIZE_ORDER`.
    """
    models: list[dict[str, Any]] = []
    if not models_dir.is_dir():
        return models
    for pt_path in sorted(models_dir.glob(f"*{finetuned_suffix}.pt")):
        parsed = parse_model_stem(pt_path.stem, finetuned_suffix)
        if parsed is None:
            continue
        version, size_letter = parsed
        models.append({
            "name": f"{version}{size_letter}-seg",
            "version": version,
            "size": size_letter,
            "pt": pt_path,
        })
    models.sort(key=model_sort_key)
    return models


# %%
def artefact_paths(
    model: dict[str, Any],
    models_dir: Path,
    naming: dict[str, str],
) -> dict[str, Path]:
    """Return every export artefact path for one model.

    Args:
        model: A model dict from :func:`discover_exportable_models`.
        models_dir: Directory the artefacts live in.
        naming: The ``naming`` block from the config.

    Returns:
        A mapping with ``onnx``, ``engine_fp16`` and ``engine_int8`` paths.
    """
    name = str(model["name"])
    finetuned = naming["finetuned_suffix"]
    int8 = naming["int8_suffix"]
    return {
        "onnx": models_dir / f"{name}{finetuned}.onnx",
        "engine_fp16": models_dir / f"{name}{finetuned}.engine",
        "engine_int8": models_dir / f"{name}{int8}.engine",
    }


# %% Utilities
def clear_cuda() -> None:
    """Release cached CUDA blocks between model exports."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# %%
def file_ok(path: Path, min_bytes: int) -> bool:
    """Check that an exported file exists and is plausibly complete.

    Args:
        path: The artefact to check.
        min_bytes: Minimum acceptable size in bytes.

    Returns:
        ``True`` when the file exists and is at least ``min_bytes`` long.
    """
    return path.is_file() and path.stat().st_size >= min_bytes


# %%
def build_phase_progress(console: Console) -> Progress:
    """Build the per-phase rich progress bar.

    Args:
        console: The ``rich`` console the bar renders to.

    Returns:
        A configured, unstarted :class:`~rich.progress.Progress`.
    """
    return Progress(
        TextColumn("[bold cyan]{task.fields[phase]}"),
        TextColumn("[dim]{task.fields[model]}[/dim]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# %% Phase 1 - ONNX FP32
def export_onnx(
    model: dict[str, Any],
    paths: dict[str, Path],
    export_cfg: dict[str, Any],
    min_bytes: int,
    console: Console,
) -> bool:
    """Export one model to ONNX FP32.

    Args:
        model: The model dict being exported.
        paths: Its artefact paths.
        export_cfg: The ``export`` block from the config.
        min_bytes: Minimum acceptable ONNX size in bytes.
        console: ``rich`` console for status output.

    Returns:
        ``True`` when a plausible ONNX file exists afterwards.
    """
    name = str(model["name"])
    try:
        yolo = YOLO(str(model["pt"]))
        output = yolo.export(
            format="onnx",
            imgsz=int(export_cfg["imgsz"]),
            opset=int(export_cfg["onnx_opset"]),
            simplify=True,
        )
        produced = Path(str(output))
        if produced.is_file() and produced.resolve() != paths["onnx"].resolve():
            shutil.move(str(produced), str(paths["onnx"]))
    except Exception as exc:  # noqa: BLE001 - one bad export must not stop the phase
        console.print(f"[red]  ONNX  {name}: {exc}[/red]")
        return False
    finally:
        clear_cuda()

    if not file_ok(paths["onnx"], min_bytes):
        console.print(
            f"[red]  ONNX  {name}: output missing or under "
            f"{min_bytes / 1e6:.0f} MB[/red]"
        )
        return False
    size_mb = paths["onnx"].stat().st_size / 1e6
    console.print(
        f"[green]  ONNX  {name}[/green] -> {paths['onnx'].name} ({size_mb:.1f} MB)"
    )
    return True


# %% Phase 2 - TensorRT FP16
def export_trt_fp16(
    model: dict[str, Any],
    paths: dict[str, Path],
    export_cfg: dict[str, Any],
    console: Console,
) -> bool:
    """Export one model to a TensorRT FP16 engine.

    Only attempted when the model's ONNX graph exists, so phase 2 mirrors the
    outcome of phase 1 rather than silently re-deriving the graph for a model
    whose ONNX export failed.

    Args:
        model: The model dict being exported.
        paths: Its artefact paths.
        export_cfg: The ``export`` block from the config.
        console: ``rich`` console for status output.

    Returns:
        ``True`` when the engine file exists afterwards.
    """
    name = str(model["name"])
    if not paths["onnx"].is_file():
        console.print(f"[yellow]  FP16  {name}: no ONNX from phase 1 - skipped[/yellow]")
        return False
    try:
        yolo = YOLO(str(model["pt"]))
        output = yolo.export(
            format="engine",
            half=True,
            imgsz=int(export_cfg["imgsz"]),
            device=0,
        )
        produced = Path(str(output))
        if produced.is_file() and produced.resolve() != paths["engine_fp16"].resolve():
            shutil.move(str(produced), str(paths["engine_fp16"]))
    except Exception as exc:  # noqa: BLE001 - one bad export must not stop the phase
        console.print(f"[red]  FP16  {name}: {exc}[/red]")
        return False
    finally:
        clear_cuda()

    if not paths["engine_fp16"].is_file():
        console.print(f"[red]  FP16  {name}: engine not produced[/red]")
        return False
    size_mb = paths["engine_fp16"].stat().st_size / 1e6
    console.print(
        f"[green]  FP16  {name}[/green] -> {paths['engine_fp16'].name} ({size_mb:.1f} MB)"
    )
    return True


# %% Phase 3 - TensorRT INT8
def log_int8_failure(log_path: Path, model_name: str, error: BaseException) -> None:
    """Append one INT8 failure with a full traceback.

    Args:
        log_path: Path to ``int8_failures.log``.
        model_name: The model that failed.
        error: The exception raised.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"=== {stamp} | {model_name} ===\n{trace}\n")


# %%
def seed_ultralytics_cache(cache_file: Path, int8_pt: Path, console: Console) -> None:
    """Place our calibration cache where the Ultralytics exporter looks for it.

    Ultralytics writes (and reuses) its INT8 cache next to the file being
    exported, as ``{stem}.cache``. Copying the cache built by
    ``calibrate_int8.py`` there makes the export reuse our CrackSeg9k
    calibration instead of running a second pass of its own.

    Args:
        cache_file: The cache produced by ``calibrate_int8.py``.
        int8_pt: The temporary checkpoint the INT8 export runs on.
        console: ``rich`` console for status output.
    """
    if not cache_file.is_file():
        return
    target = int8_pt.with_suffix(".cache")
    shutil.copy2(cache_file, target)
    console.print(f"[dim]  seeded exporter cache -> {target.name}[/dim]")


# %%
def export_trt_int8(
    model: dict[str, Any],
    paths: dict[str, Path],
    config: dict[str, Any],
    models_dir: Path,
    dataset_yaml: Path,
    calib_images: list[Path],
    failure_log: Path,
    console: Console,
) -> bool:
    """Build the calibration cache and export one INT8 engine.

    The export runs against a temporary ``{name}_crack_int8.pt`` copy of the
    checkpoint so Ultralytics writes ``{name}_crack_int8.engine`` directly and
    cannot overwrite the FP16 engine produced in phase 2.

    Args:
        model: The model dict being exported.
        paths: Its artefact paths.
        config: The full pipeline configuration.
        models_dir: Directory the artefacts live in.
        dataset_yaml: Dataset descriptor passed to the exporter.
        calib_images: Sampled calibration images.
        failure_log: Path to ``int8_failures.log``.
        console: ``rich`` console for status output.

    Returns:
        ``True`` when the INT8 engine exists afterwards.
    """
    name = str(model["name"])
    export_cfg = config["export"]
    naming = config["naming"]
    cache_dir = models_dir / export_cfg["int8_calib_cache"]
    cache_file = cache_path_for(name, cache_dir, naming["cache_suffix"])
    int8_pt = models_dir / f"{name}{naming['int8_suffix']}.pt"
    int8_onnx = int8_pt.with_suffix(".onnx")
    seeded_cache = int8_pt.with_suffix(".cache")

    if not paths["onnx"].is_file():
        console.print(f"[yellow]  INT8  {name}: no ONNX from phase 1 - skipped[/yellow]")
        return False

    def _log(message: str) -> None:
        """Forward a calibration log line to the console.

        Args:
            message: The line to print.
        """
        console.print(f"[dim]  {message}[/dim]")

    try:
        ensure_calibration_cache(
            model_name=name,
            onnx_path=paths["onnx"],
            cache_file=cache_file,
            image_paths=calib_images,
            imgsz=int(export_cfg["imgsz"]),
            workspace_mib=int(export_cfg["workspace_mib"]),
            log=_log,
            force=False,
        )

        shutil.copy2(model["pt"], int8_pt)
        seed_ultralytics_cache(cache_file, int8_pt, console)

        yolo = YOLO(str(int8_pt))
        output = yolo.export(
            format="engine",
            int8=True,
            imgsz=int(export_cfg["imgsz"]),
            device=0,
            data=str(dataset_yaml),
        )
        produced = Path(str(output))
        if produced.is_file() and produced.resolve() != paths["engine_int8"].resolve():
            shutil.move(str(produced), str(paths["engine_int8"]))

        # Ultralytics may have rewritten the cache during the build; keep the
        # authoritative copy under calibration_cache/ in sync for the Jetson.
        if seeded_cache.is_file():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seeded_cache, cache_file)
    except Exception as exc:  # noqa: BLE001 - one bad export must not stop the phase
        console.print(f"[red]  INT8  {name}: {exc}[/red]")
        log_int8_failure(failure_log, name, exc)
        return False
    finally:
        for temporary in (int8_pt, int8_onnx, seeded_cache):
            if temporary.is_file():
                temporary.unlink()
        clear_cuda()

    if not paths["engine_int8"].is_file():
        console.print(f"[red]  INT8  {name}: engine not produced[/red]")
        return False
    size_mb = paths["engine_int8"].stat().st_size / 1e6
    console.print(
        f"[green]  INT8  {name}[/green] -> {paths['engine_int8'].name} ({size_mb:.1f} MB)"
    )
    return True


# %% Phase drivers
def run_phase_onnx(
    models: list[dict[str, Any]],
    all_paths: dict[str, dict[str, Path]],
    config: dict[str, Any],
    results: dict[str, dict[str, bool]],
    console: Console,
) -> None:
    """Run phase 1 for every model.

    Args:
        models: Discovered models.
        all_paths: Per-model artefact paths.
        config: The full pipeline configuration.
        results: Per-model outcome map, updated in place.
        console: ``rich`` console for status output.
    """
    console.print("[bold]Phase 1/3 - ONNX FP32[/bold]")
    min_bytes = int(config["recovery"]["min_onnx_bytes"])
    with build_phase_progress(console) as progress:
        task_id = progress.add_task("", total=len(models), phase="ONNX", model="")
        for model in models:
            name = str(model["name"])
            progress.update(task_id, model=name)
            results[name]["onnx"] = export_onnx(
                model, all_paths[name], config["export"], min_bytes, console,
            )
            progress.advance(task_id)


# %%
def run_phase_trt_fp16(
    models: list[dict[str, Any]],
    all_paths: dict[str, dict[str, Path]],
    config: dict[str, Any],
    results: dict[str, dict[str, bool]],
    console: Console,
) -> None:
    """Run phase 2 for every model.

    Args:
        models: Discovered models.
        all_paths: Per-model artefact paths.
        config: The full pipeline configuration.
        results: Per-model outcome map, updated in place.
        console: ``rich`` console for status output.
    """
    console.print("[bold]Phase 2/3 - TensorRT FP16[/bold]")
    if not config["export"]["trt_fp16"]:
        console.print("[yellow]  trt_fp16 disabled in config - phase skipped[/yellow]")
        return
    with build_phase_progress(console) as progress:
        task_id = progress.add_task("", total=len(models), phase="TRT FP16", model="")
        for model in models:
            name = str(model["name"])
            progress.update(task_id, model=name)
            results[name]["trt_fp16"] = export_trt_fp16(
                model, all_paths[name], config["export"], console,
            )
            progress.advance(task_id)


# %%
def run_phase_trt_int8(
    models: list[dict[str, Any]],
    all_paths: dict[str, dict[str, Path]],
    config: dict[str, Any],
    models_dir: Path,
    data_dir: Path,
    dataset_yaml: Path,
    results: dict[str, dict[str, bool]],
    console: Console,
) -> None:
    """Run phase 3 for every model, after phases 1 and 2 have finished.

    The whole phase is wrapped so a catastrophic native failure (OOM, a signal
    raised inside the TensorRT builder) is recorded rather than killing the
    process before the summary table is printed.

    Args:
        models: Discovered models.
        all_paths: Per-model artefact paths.
        config: The full pipeline configuration.
        models_dir: Directory the artefacts live in.
        data_dir: Root of the processed dataset.
        dataset_yaml: Dataset descriptor passed to the exporter.
        results: Per-model outcome map, updated in place.
        console: ``rich`` console for status output.
    """
    console.print("[bold]Phase 3/3 - TensorRT INT8[/bold]")
    export_cfg = config["export"]
    failure_log = _HERE / config["recovery"]["int8_failure_log"]

    if not export_cfg["trt_int8"]:
        console.print("[yellow]  trt_int8 disabled in config - phase skipped[/yellow]")
        return

    try:
        split_dir = data_dir / "images" / export_cfg["int8_calib_split"]
        calib_images = collect_calibration_images(
            split_dir, int(export_cfg["int8_calib_images"]),
        )
        if not calib_images:
            console.print(
                f"[red]  no calibration images under {split_dir} - phase skipped[/red]"
            )
            return
        console.print(
            f"[dim]  calibration set: {len(calib_images)} image(s) from "
            f"{export_cfg['int8_calib_split']}[/dim]"
        )

        with build_phase_progress(console) as progress:
            task_id = progress.add_task("", total=len(models), phase="TRT INT8", model="")
            for model in models:
                name = str(model["name"])
                progress.update(task_id, model=name)
                results[name]["trt_int8"] = export_trt_int8(
                    model, all_paths[name], config, models_dir,
                    dataset_yaml, calib_images, failure_log, console,
                )
                progress.advance(task_id)
    except BaseException as exc:  # noqa: BLE001 - a fatal phase must still report
        console.print(f"[red]  INT8 phase aborted: {exc}[/red]")
        log_int8_failure(failure_log, "<phase>", exc)


# %% Reporting
def print_discovery(
    models: list[dict[str, Any]],
    models_dir: Path,
    console: Console,
) -> None:
    """Print what the filesystem scan found before exporting anything.

    Args:
        models: Discovered models.
        models_dir: The directory that was scanned.
        console: ``rich`` console for output.
    """
    console.print("[bold]Jetson pipeline - export[/bold]")
    console.print(f"[dim]Scanned: {models_dir}[/dim]")
    if models:
        names = ", ".join(str(model["name"]) for model in models)
        console.print(f"[green]Found {len(models)} finetuned model(s):[/green] {names}")
    else:
        console.print(
            "[red]No *_crack_finetuned.pt found - run train_missing.py first.[/red]"
        )


# %%
def print_summary(
    models: list[dict[str, Any]],
    results: dict[str, dict[str, bool]],
    phases: tuple[int, ...],
    console: Console,
) -> None:
    """Print the final per-model export summary table.

    Args:
        models: Discovered models.
        results: Per-model outcome map.
        phases: The phases that actually ran.
        console: ``rich`` console for output.
    """
    table = Table(title="Export Summary", title_style="bold white")
    table.add_column("Model", style="bold")
    table.add_column("ONNX", justify="center")
    table.add_column("TRT FP16", justify="center")
    table.add_column("TRT INT8", justify="center")
    table.add_column("Status", justify="left")

    for model in models:
        name = str(model["name"])
        outcome = results[name]
        cells: list[str] = []
        for phase, key in _PHASE_KEYS:
            if phase not in phases:
                cells.append("[dim]skip[/dim]")
            elif outcome[key]:
                cells.append("[green]OK[/green]")
            else:
                cells.append("[red]FAIL[/red]")

        ran = [key for phase, key in _PHASE_KEYS if phase in phases]
        failed = [key for key in ran if not outcome[key]]
        if not failed:
            status = "[green]complete[/green]"
        elif len(failed) == len(ran):
            status = "[red]all formats failed[/red]"
        else:
            status = f"[yellow]failed: {', '.join(failed)}[/yellow]"
        table.add_row(name, *cells, status)

    console.print(table)


# %% Main function
def main(phases: tuple[int, ...]) -> int:
    """Export every discovered model through the requested phases.

    Args:
        phases: The phase numbers to run, always executed in ascending order.

    Returns:
        Process exit code - ``0`` when every attempted export succeeded.
    """
    console = Console()
    install_fatal_signal_handlers(console)
    config = load_config(_CONFIG_PATH)

    models_dir = resolve_path(config["models_dir"])
    data_dir = resolve_path(config["data_dir"])
    dataset_yaml = resolve_path(config["dataset_yaml"])
    naming = config["naming"]

    models = discover_exportable_models(models_dir, naming["finetuned_suffix"])
    print_discovery(models, models_dir, console)
    if not models:
        return 1

    all_paths = {
        str(model["name"]): artefact_paths(model, models_dir, naming)
        for model in models
    }
    results: dict[str, dict[str, bool]] = {
        str(model["name"]): {"onnx": False, "trt_fp16": False, "trt_int8": False}
        for model in models
    }

    ordered = tuple(phase for phase in _ALL_PHASES if phase in phases)
    if _PHASE_ONNX in ordered:
        run_phase_onnx(models, all_paths, config, results, console)
    if _PHASE_TRT_FP16 in ordered:
        run_phase_trt_fp16(models, all_paths, config, results, console)
    # INT8 always runs last, after every ONNX and FP16 export has finished.
    if _PHASE_TRT_INT8 in ordered:
        run_phase_trt_int8(
            models, all_paths, config, models_dir, data_dir,
            dataset_yaml, results, console,
        )

    print_summary(models, results, ordered, console)
    failures = sum(
        1
        for model in models
        for phase, key in _PHASE_KEYS
        if phase in ordered and not results[str(model["name"])][key]
    )
    if failures:
        console.print(f"[red]{failures} export(s) failed - see the table above.[/red]")
        return 1
    console.print("[bold green]All exports complete.[/bold green]")
    return 0


# %% Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export every finetuned model to ONNX FP32, TRT FP16 and TRT INT8.",
    )
    parser.add_argument(
        "--phases", type=int, nargs="+", choices=list(_ALL_PHASES),
        default=list(_ALL_PHASES),
        help="Phases to run (1=ONNX, 2=TRT FP16, 3=TRT INT8). INT8 always runs last.",
    )
    args = parser.parse_args()
    sys.exit(main(phases=tuple(args.phases)))
