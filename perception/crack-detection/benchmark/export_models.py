# %%
"""Export the fine-tuned crack-detection models to ONNX and TensorRT FP16.

The model list is read from ``train_config.yaml`` so the exporter always tracks
whatever was trained. For every model it loads
``models/{model}_crack_finetuned.pt`` and writes:

* an ONNX graph (``models/{model}_crack_finetuned.onnx``), and
* a TensorRT FP16 engine (``models/{model}_crack_finetuned.engine``).

Each export is independent: a failure for one model or one format is logged and
the exporter moves on, so a missing TensorRT toolchain never aborts the run.

Run with::

    conda activate cuda_pt
    python export_models.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO

# Directory / file locations, all relative to this file.
_BASE_DIR: Path = Path(__file__).parent
_TRAIN_CONFIG_PATH: Path = _BASE_DIR / "train_config.yaml"
_MODELS_DIR: Path = _BASE_DIR / "models"

# Default image size used when a model block omits ``imgsz``.
_DEFAULT_IMGSZ: int = 640


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
        print("[export] CUDA requested but not available - falling back to CPU.")
        return "cpu"
    return requested


# %%
def export_one_format(
    weights_path: Path,
    export_format: str,
    imgsz: int,
    device: str,
    half: bool,
    console: Console,
) -> bool:
    """Export a single model to one format, logging and swallowing failures.

    Args:
        weights_path: Path to the fine-tuned ``.pt`` checkpoint.
        export_format: Ultralytics export format (``"onnx"`` or ``"engine"``).
        imgsz: Inference image size baked into the exported graph.
        device: The resolved compute device string.
        half: Whether to export in FP16 (TensorRT only).
        console: ``rich`` console for status output.

    Returns:
        ``True`` if the export succeeded, ``False`` otherwise.
    """
    label = "TensorRT FP16" if export_format == "engine" else export_format.upper()
    try:
        model = YOLO(str(weights_path))
        output = model.export(
            format=export_format,
            imgsz=imgsz,
            device=device,
            half=half,
        )
        console.print(f"[green]  {label} ->[/green] {output}")
        return True
    except Exception as exc:  # noqa: BLE001 - one bad export must not stop the rest
        console.print(f"[red]  {label} export failed: {exc}[/red]")
        return False


# %%
def export_one_model(
    model_name: str,
    model_spec: dict[str, Any],
    device: str,
    console: Console,
) -> dict[str, Any]:
    """Export one fine-tuned model to ONNX and TensorRT FP16.

    Args:
        model_name: The config key (e.g. ``yolo11s-seg``).
        model_spec: The per-model hyperparameter block from the config.
        device: The resolved compute device string.
        console: ``rich`` console for status output.

    Returns:
        A result dict with the model name and the ONNX / TensorRT outcomes.
    """
    finetuned_path = _MODELS_DIR / f"{model_name}_crack_finetuned.pt"
    imgsz = int(model_spec.get("imgsz", _DEFAULT_IMGSZ))
    result: dict[str, Any] = {"model": model_name, "onnx": False, "tensorrt": False}

    if not finetuned_path.is_file():
        console.print(
            f"[yellow]Skipping {model_name}:[/yellow] no fine-tuned weights at "
            f"{finetuned_path.name}. Run 'python train.py' first."
        )
        return result

    console.print(f"[cyan]Exporting {model_name}...[/cyan]")
    result["onnx"] = export_one_format(
        finetuned_path, "onnx", imgsz, device, half=False, console=console
    )
    # TensorRT engines are device-specific; FP16 halves memory and boosts speed.
    result["tensorrt"] = export_one_format(
        finetuned_path, "engine", imgsz, device, half=True, console=console
    )
    return result


# %%
def print_export_table(results: list[dict[str, Any]], console: Console) -> None:
    """Print the export outcome summary table using ``rich``.

    Args:
        results: Per-model export result dicts from :func:`export_one_model`.
        console: The ``rich`` console to print to.
    """
    table = Table(title="Model Export Summary", title_style="bold white")
    table.add_column("Model", style="bold")
    table.add_column("ONNX", justify="center")
    table.add_column("TensorRT FP16", justify="center")

    for result in results:
        onnx_mark = "[green]OK[/green]" if result["onnx"] else "[red]FAIL[/red]"
        trt_mark = "[green]OK[/green]" if result["tensorrt"] else "[red]FAIL[/red]"
        table.add_row(str(result["model"]), onnx_mark, trt_mark)
    console.print(table)


# %%
def run_export() -> None:
    """Export every model declared in ``train_config.yaml`` to ONNX and TensorRT."""
    console = Console()
    console.print("[bold]Crack Detection - Model Export[/bold]")

    config = load_train_config(_TRAIN_CONFIG_PATH)
    device = resolve_device(config["device"])
    models: dict[str, dict[str, Any]] = config["models"]

    results: list[dict[str, Any]] = []
    for model_name, model_spec in models.items():
        results.append(export_one_model(model_name, model_spec, device, console))

    print_export_table(results, console)
    console.print("[bold green]Export complete.[/bold green]")


# %%
def main() -> None:
    """Entry point for the export pipeline."""
    run_export()


# %%
if __name__ == "__main__":
    main()
