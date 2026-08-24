# %%
"""Post-training evaluation of the fine-tuned crack-detection models.

For each fine-tuned model this script:

1. Loads ``models/{model}_crack_finetuned.pt``.
2. Runs ``model.val()`` on the **test** split of the prepared dataset.
3. Copies the confusion matrix, PR curve and F1 curve into ``results/``.
4. Builds a grouped mAP comparison chart in the project's dark palette.
5. Repoints the model weight paths in ``config.yaml`` at the fine-tuned
   checkpoints so a plain ``python benchmark.py`` picks them up automatically.

Once every model is validated the script automatically runs
``export_models.py`` as a subprocess so the ONNX and TensorRT engines are
regenerated without manual intervention.

Run with::

    conda activate cuda_pt
    python validate.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering - no display required
import matplotlib.pyplot as plt
import torch
import yaml
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO

# Directory / file locations, all relative to this file.
_BASE_DIR: Path = Path(__file__).parent
_TRAIN_CONFIG_PATH: Path = _BASE_DIR / "train_config.yaml"
_CONFIG_PATH: Path = _BASE_DIR / "config.yaml"
_MODELS_DIR: Path = _BASE_DIR / "models"
_RESULTS_DIR: Path = _BASE_DIR / "results"
_DATASET_YAML_PATH: Path = _BASE_DIR / "data" / "processed" / "dataset.yaml"
_RUNS_DIR: Path = _BASE_DIR / "runs" / "val"
_EXPORT_SCRIPT_PATH: Path = _BASE_DIR / "export_models.py"

# Model registry: maps a config key to its display name. The keys match the
# model block keys in config.yaml so the weight paths can be repointed.
_MODEL_REGISTRY: dict[str, str] = {
    "yolo11n-seg": "YOLO11n-seg",
    "yolo11s-seg": "YOLO11s-seg",
    "yolo11m-seg": "YOLO11m-seg",
}

# Dark plotting palette - identical to report.py.
_BACKGROUND: str = "#0D0D0D"
_DPI: int = 150
_PALETTE: tuple[str, ...] = ("#E87722", "#00B4A6", "#3498DB")

# Candidate filenames for the diagnostic plots Ultralytics writes, in
# preference order (mask plots first for segmentation models).
_PR_CURVE_CANDIDATES: tuple[str, ...] = ("MaskPR_curve.png", "BoxPR_curve.png", "PR_curve.png")
_F1_CURVE_CANDIDATES: tuple[str, ...] = ("MaskF1_curve.png", "BoxF1_curve.png", "F1_curve.png")
_CONFUSION_MATRIX_NAME: str = "confusion_matrix.png"


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
        print("[validate] CUDA requested but not available - falling back to CPU.")
        return "cpu"
    return requested


# %%
def extract_metrics(metrics_obj: Any) -> dict[str, float]:
    """Pull the headline scores out of an Ultralytics validation result.

    Segmentation models are scored on their mask metrics; detection models on
    their box metrics. The F1 score is derived from precision and recall.

    Args:
        metrics_obj: The object returned by ``model.val()``.

    Returns:
        A dict with ``mAP50``, ``mAP50-95``, ``precision``, ``recall`` and
        ``f1``.
    """
    sub: Any = None
    for attribute in ("seg", "box"):
        candidate = getattr(metrics_obj, attribute, None)
        if candidate is not None and hasattr(candidate, "map50"):
            sub = candidate
            break

    if sub is None:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = float(sub.mp)
    recall = float(sub.mr)
    denominator = precision + recall
    f1 = (2.0 * precision * recall / denominator) if denominator > 0 else 0.0

    return {
        "mAP50": float(sub.map50),
        "mAP50-95": float(sub.map),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# %%
def copy_diagnostic_plot(
    save_dir: Path,
    candidates: tuple[str, ...],
    destination: Path,
) -> bool:
    """Copy the first matching diagnostic plot from a val run into ``results/``.

    Args:
        save_dir: The Ultralytics validation run directory.
        candidates: Candidate source filenames, tried in preference order.
        destination: Destination path for the copied plot.

    Returns:
        ``True`` if a plot was found and copied, ``False`` otherwise.
    """
    for filename in candidates:
        source = save_dir / filename
        if source.is_file():
            shutil.copy2(source, destination)
            return True
    return False


# %%
def validate_one_model(
    model_key: str,
    device: str,
    workers: int,
    console: Console,
) -> dict[str, Any] | None:
    """Validate one fine-tuned model on the test split and save its plots.

    Args:
        model_key: Registry key (``yolo11n``, ``yolo11n_seg`` or ``fastsam``).
        device: The resolved compute device string.
        workers: Number of DataLoader workers (0 on Windows).
        console: ``rich`` console for status output.

    Returns:
        A metrics dict augmented with ``model_key`` and ``display`` keys, or
        ``None`` if the fine-tuned weights are missing.
    """
    display_name = _MODEL_REGISTRY[model_key]
    finetuned_path = _MODELS_DIR / f"{model_key}_crack_finetuned.pt"
    if not finetuned_path.is_file():
        console.print(
            f"[yellow]Skipping {display_name}:[/yellow] no fine-tuned weights at "
            f"{finetuned_path.name}. Run 'python train.py' first."
        )
        return None

    console.print(f"[cyan]Validating {display_name} on the test split...[/cyan]")
    model = YOLO(str(finetuned_path))
    metrics_obj = model.val(
        data=str(_DATASET_YAML_PATH),
        split="test",
        device=device,
        workers=workers,
        project=str(_RUNS_DIR),
        name=model_key,
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    save_dir = Path(metrics_obj.save_dir)
    copy_diagnostic_plot(
        save_dir,
        (_CONFUSION_MATRIX_NAME,),
        _RESULTS_DIR / f"val_{model_key}_confusion_matrix.png",
    )
    copy_diagnostic_plot(
        save_dir,
        _PR_CURVE_CANDIDATES,
        _RESULTS_DIR / f"val_{model_key}_pr_curve.png",
    )
    copy_diagnostic_plot(
        save_dir,
        _F1_CURVE_CANDIDATES,
        _RESULTS_DIR / f"val_{model_key}_f1_curve.png",
    )

    scores = extract_metrics(metrics_obj)
    scores["model_key"] = model_key
    scores["display"] = display_name
    return scores


# %%
def _style_axes(ax: plt.Axes) -> None:
    """Apply the shared dark-background style to a Matplotlib axes.

    Args:
        ax: The axes to style, modified in place.
    """
    ax.set_facecolor(_BACKGROUND)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True, color="#333333", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.tick_params(colors="#CCCCCC")
    ax.xaxis.label.set_color("#EEEEEE")
    ax.yaxis.label.set_color("#EEEEEE")
    ax.title.set_color("#FFFFFF")


# %%
def plot_map_comparison(results: list[dict[str, Any]], destination: Path) -> None:
    """Render the grouped mAP50 / mAP50-95 comparison bar chart.

    Each model gets a pair of bars: mAP50 (orange) and mAP50-95 (teal), drawn
    on the project's dark background.

    Args:
        results: Per-model metrics dicts from :func:`validate_one_model`.
        destination: Output PNG path.
    """
    names = [item["display"] for item in results]
    map50 = [item["mAP50"] for item in results]
    map50_95 = [item["mAP50-95"] for item in results]
    positions = range(len(names))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BACKGROUND)
    ax.bar(
        [p - bar_width / 2 for p in positions],
        map50,
        width=bar_width,
        color=_PALETTE[0],
        label="mAP50",
    )
    ax.bar(
        [p + bar_width / 2 for p in positions],
        map50_95,
        width=bar_width,
        color=_PALETTE[1],
        label="mAP50-95",
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Fine-tuned Model Accuracy - Test Split")
    _style_axes(ax)
    legend = ax.legend(facecolor=_BACKGROUND, edgecolor="#444444")
    for text in legend.get_texts():
        text.set_color("#EEEEEE")
    fig.tight_layout()
    fig.savefig(destination, dpi=_DPI, facecolor=_BACKGROUND)
    plt.close(fig)


# %%
def update_config_weights(
    config_path: Path,
    updated_keys: list[str],
    console: Console,
) -> None:
    """Repoint the model weight paths in ``config.yaml`` at fine-tuned files.

    The file is rewritten line by line so the existing comments and layout are
    preserved; only the ``weights:`` line under each updated model block is
    changed.

    Args:
        config_path: Path to the benchmark ``config.yaml``.
        updated_keys: Model keys whose weights should be repointed.
        console: ``rich`` console for status output.
    """
    lines = config_path.read_text(encoding="utf-8").splitlines()
    current_key: str | None = None
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # A model block header sits two spaces in and ends with a colon.
        candidate_key = stripped[:-1] if stripped.endswith(":") else ""
        if candidate_key in _MODEL_REGISTRY and line.startswith("  ") and not line.startswith("   "):
            current_key = candidate_key
            new_lines.append(line)
            continue

        if (
            current_key in updated_keys
            and stripped.startswith("weights:")
        ):
            indent = line[: len(line) - len(line.lstrip())]
            finetuned = f"models/{current_key}_crack_finetuned.pt"
            new_lines.append(f"{indent}weights: {finetuned}   # fine-tuned by validate.py")
            continue

        new_lines.append(line)

    config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    console.print(
        f"[green]config.yaml updated[/green] - weights now point to fine-tuned "
        f"checkpoints for: {', '.join(updated_keys)}"
    )


# %%
def print_summary_table(results: list[dict[str, Any]], console: Console) -> None:
    """Print the final validation summary table using ``rich``.

    Args:
        results: Per-model metrics dicts from :func:`validate_one_model`.
        console: The ``rich`` console to print to.
    """
    table = Table(title="Fine-tuned Model Validation - Test Split", title_style="bold white")
    table.add_column("Model", style="bold")
    table.add_column("mAP50", justify="right")
    table.add_column("mAP50-95", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    for item in results:
        table.add_row(
            str(item["display"]),
            f"{item['mAP50']:.4f}",
            f"{item['mAP50-95']:.4f}",
            f"{item['precision']:.4f}",
            f"{item['recall']:.4f}",
            f"{item['f1']:.4f}",
        )
    console.print(table)


# %%
def run_export_subprocess(console: Console) -> None:
    """Run ``export_models.py`` as a subprocess after validation completes.

    The export step is launched with the same interpreter that is running this
    script so it stays inside the active conda environment. A non-zero exit (or
    a missing script) is logged but never propagated, so an export failure
    cannot abort the overall pipeline.

    Args:
        console: ``rich`` console for status output.
    """
    if not _EXPORT_SCRIPT_PATH.is_file():
        console.print(
            f"[yellow]Skipping export:[/yellow] {_EXPORT_SCRIPT_PATH.name} not found."
        )
        return

    console.print("[bold]Running export_models.py...[/bold]")
    try:
        subprocess.run([sys.executable, str(_EXPORT_SCRIPT_PATH)], check=True)
    except subprocess.CalledProcessError as exc:
        console.print(
            f"[red]Export step failed (exit {exc.returncode}); continuing.[/red]"
        )


# %%
def run_validation() -> None:
    """Validate every fine-tuned model and produce all diagnostic outputs."""
    console = Console()
    console.print("[bold]Crack Detection - Post-training Validation[/bold]")

    if not _DATASET_YAML_PATH.is_file():
        console.print(
            f"[red]Dataset not found at {_DATASET_YAML_PATH}.[/red] "
            "Run 'python prepare_data.py' first."
        )
        return

    config = load_train_config(_TRAIN_CONFIG_PATH)
    device = resolve_device(config["device"])
    workers = int(config["workers"])
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for model_key in _MODEL_REGISTRY:
        scores = validate_one_model(model_key, device, workers, console)
        if scores is not None:
            results.append(scores)

    if not results:
        console.print("[red]No fine-tuned models found - nothing to validate.[/red]")
        return

    comparison_path = _RESULTS_DIR / "val_map_comparison.png"
    plot_map_comparison(results, comparison_path)
    console.print(f"[green]Comparison chart written to[/green] {comparison_path}")

    update_config_weights(_CONFIG_PATH, [item["model_key"] for item in results], console)
    print_summary_table(results, console)
    console.print("[bold green]Validation complete.[/bold green]")

    # Automatically regenerate ONNX / TensorRT artifacts for the fresh weights.
    run_export_subprocess(console)


# %%
def main() -> None:
    """Entry point for the post-training validation pipeline."""
    run_validation()


# %%
if __name__ == "__main__":
    main()
