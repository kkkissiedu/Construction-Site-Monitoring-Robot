# %%
"""verify_exports.py - load every exported artefact and run one forward pass.

An export that finishes without raising can still be corrupt or truncated. This
script proves each artefact actually runs:

* ``.onnx`` - loaded with ``onnxruntime.InferenceSession`` and fed a random
  ``(1, 3, 640, 640)`` float32 tensor. Output shapes are compared against
  ``verify.expected_shapes`` in the config: ``(1, 37, 8400)`` for the detection
  head (4 box + 1 class + 32 mask coefficients) and ``(1, 32, 160, 160)`` for
  the mask prototypes.
* ``.engine`` - loaded through the Ultralytics ``YOLO`` wrapper and run once on
  a random frame; the result must not be ``None``.

Exits ``0`` when every artefact passes and ``1`` when any fails, so it can gate
a CI job or a deployment script.

Usage:
    conda activate cuda_pt
    python jetson_pipeline/verify_exports.py
"""

from __future__ import annotations

# %% Imports
import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

# %% Config
_HERE: Path = Path(__file__).parent
_CONFIG_PATH: Path = _HERE / "jetson_config.yaml"

# Canonical size ordering, shared with the rest of the pipeline.
_SIZE_ORDER: tuple[str, ...] = ("n", "s", "m", "l", "x")

# Engines built for the Jetson only load on the Jetson, so they are skipped
# unless this machine is one.
_TEGRA_RELEASE_PATH: Path = Path("/etc/nv_tegra_release")

# Per-artefact status strings.
_STATUS_PASS: str = "PASS"
_STATUS_FAIL: str = "FAIL"
_STATUS_SKIP: str = "SKIP"


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
def suffix_length_key(item: tuple[str, str]) -> int:
    """Sort key ordering ``(flavour, suffix)`` pairs by suffix length.

    Args:
        item: A ``(flavour, suffix)`` pair.

    Returns:
        The suffix length.
    """
    return len(item[1])


# %%
def parse_artefact_stem(stem: str, suffixes: dict[str, str]) -> tuple[str, str] | None:
    """Parse an exported artefact stem into ``(model_name, flavour)``.

    Args:
        stem: Filename without extension.
        suffixes: ``{flavour: stem_suffix}`` mapping derived from the config.

    Returns:
        ``(model_name, flavour)`` such as ``("yolo11n-seg", "TRT FP16")``, or
        ``None`` when the stem matches no known suffix.
    """
    # Longest suffix first so `_crack_int8_jetson` is not shadowed by
    # `_crack_int8`.
    for flavour, suffix in sorted(suffixes.items(), key=suffix_length_key, reverse=True):
        if not stem.endswith(suffix):
            continue
        head = stem[: -len(suffix)]
        if not head.endswith("-seg"):
            continue
        head = head[: -len("-seg")]
        if not head or head[-1] not in _SIZE_ORDER:
            continue
        version = head[:-1]
        if not version:
            continue
        return f"{version}{head[-1]}-seg", flavour
    return None


# %%
def split_model_name(model_name: str) -> tuple[str, str]:
    """Split a model key into ``(version, size_letter)``.

    Args:
        model_name: Model key, e.g. ``yolo26l-seg``.

    Returns:
        ``(version, size_letter)``, defaulting the size to ``"n"`` when the
        key does not end in a recognised size letter.
    """
    head = model_name[: -len("-seg")] if model_name.endswith("-seg") else model_name
    if head and head[-1] in _SIZE_ORDER:
        return head[:-1], head[-1]
    return head, "n"


# %%
def artefact_sort_key(artefact: dict[str, Any]) -> tuple[str, int, str]:
    """Sort key grouping artefacts by version, then size, then flavour.

    Args:
        artefact: An artefact dict from :func:`discover_artefacts`.

    Returns:
        ``(version, size_index, flavour)``.
    """
    version, size_letter = split_model_name(str(artefact["model"]))
    return version, _SIZE_ORDER.index(size_letter), str(artefact["flavour"])


# %%
def expected_shapes_for(version: str, verify_cfg: dict[str, Any]) -> list[tuple[int, ...]]:
    """Return the ONNX output-shape contract for one YOLO version.

    YOLO11 and YOLO12 share the anchor-grid head, ``(1, 37, 8400)``. YOLO26 is
    NMS-free end to end and emits 300 already-selected detections instead, so
    its contract is declared as a per-version override in the config rather
    than being force-fitted to the default.

    Args:
        version: Version prefix, e.g. ``yolo26``.
        verify_cfg: The ``verify`` block from the config.

    Returns:
        The expected output shapes, in output order.
    """
    overrides = verify_cfg.get("expected_shapes_by_version") or {}
    shapes = overrides.get(version, verify_cfg["expected_shapes"])
    return [tuple(int(dim) for dim in shape) for shape in shapes]


# %%
def discover_artefacts(
    models_dir: Path,
    naming: dict[str, str],
    include_jetson: bool,
) -> list[dict[str, Any]]:
    """Find every exported artefact worth verifying.

    Args:
        models_dir: Directory holding the exports.
        naming: The ``naming`` block from the config.
        include_jetson: Whether to include ``*_jetson.engine`` artefacts.

    Returns:
        One dict per artefact with ``path``, ``model``, ``flavour`` and ``kind``
        (``"onnx"`` or ``"engine"``) keys, ordered by model then flavour.
    """
    engine_suffixes = {
        "TRT FP16": naming["finetuned_suffix"],
        "TRT INT8": naming["int8_suffix"],
    }
    if include_jetson:
        engine_suffixes["TRT FP16 (jetson)"] = naming["jetson_fp16_suffix"]
        engine_suffixes["TRT INT8 (jetson)"] = naming["jetson_int8_suffix"]
    onnx_suffixes = {"ONNX FP32": naming["finetuned_suffix"]}

    artefacts: list[dict[str, Any]] = []
    if not models_dir.is_dir():
        return artefacts

    for path in sorted(models_dir.glob("*.onnx")):
        parsed = parse_artefact_stem(path.stem, onnx_suffixes)
        if parsed is not None:
            artefacts.append({
                "path": path, "model": parsed[0], "flavour": parsed[1], "kind": "onnx",
            })
    for path in sorted(models_dir.glob("*.engine")):
        parsed = parse_artefact_stem(path.stem, engine_suffixes)
        if parsed is not None:
            artefacts.append({
                "path": path, "model": parsed[0], "flavour": parsed[1], "kind": "engine",
            })

    artefacts.sort(key=artefact_sort_key)
    return artefacts


# %% ONNX verification
def verify_onnx(
    path: Path,
    expected_shapes: list[tuple[int, ...]],
    batch: int,
    imgsz: int,
    strict: bool,
    console: Console,
) -> tuple[str, str]:
    """Run one forward pass through an ONNX graph and check its output shapes.

    Args:
        path: The ONNX file to verify.
        expected_shapes: Output shapes the graph must produce.
        batch: Batch dimension of the dummy input.
        imgsz: Square input resolution.
        strict: Fail on a shape mismatch rather than merely reporting it.
        console: ``rich`` console for status output.

    Returns:
        ``(status, detail)`` where status is ``PASS`` or ``FAIL``.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return _STATUS_FAIL, "onnxruntime not installed"

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        dummy = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)
        outputs = session.run(None, {input_name: dummy})
    except Exception as exc:  # noqa: BLE001 - a bad file must not stop the sweep
        return _STATUS_FAIL, str(exc).splitlines()[0][:120]

    arrays = [np.asarray(output) for output in outputs]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        return _STATUS_FAIL, "output contains NaN or Inf"

    actual = [tuple(int(dim) for dim in array.shape) for array in arrays]
    expected = [tuple(shape) for shape in expected_shapes]
    if actual == expected:
        return _STATUS_PASS, f"outputs {actual}"

    detail = f"expected {expected}, got {actual}"
    if strict:
        return _STATUS_FAIL, detail
    console.print(f"[yellow]  {path.name}: {detail}[/yellow]")
    return _STATUS_PASS, detail


# %% Engine verification
def verify_engine(path: Path, imgsz: int) -> tuple[str, str]:
    """Load a TensorRT engine through Ultralytics and run one forward pass.

    Args:
        path: The ``.engine`` file to verify.
        imgsz: Square input resolution for the dummy frame.

    Returns:
        ``(status, detail)`` where status is ``PASS`` or ``FAIL``.
    """
    try:
        # Imported lazily so a machine without Ultralytics can still verify
        # ONNX graphs instead of failing at module import.
        from calibrate_int8 import patch_trt_nptype

        patch_trt_nptype()
        from ultralytics import YOLO
    except ImportError as exc:
        return _STATUS_FAIL, f"import failed: {exc}"

    try:
        model = YOLO(str(path))
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        results = model.predict(frame, verbose=False)
    except Exception as exc:  # noqa: BLE001 - a bad engine must not stop the sweep
        return _STATUS_FAIL, str(exc).splitlines()[0][:120]

    if results is None or len(results) == 0:
        return _STATUS_FAIL, "predict returned no result"
    return _STATUS_PASS, f"{len(results)} result object(s)"


# %% Reporting
def print_summary(rows: list[dict[str, Any]], console: Console) -> None:
    """Print the per-artefact verification table.

    Args:
        rows: Verification rows.
        console: ``rich`` console for output.
    """
    table = Table(title="Export Verification", title_style="bold white")
    table.add_column("Model", style="bold")
    table.add_column("Format")
    table.add_column("File")
    table.add_column("Result", justify="center")
    table.add_column("ms", justify="right")
    table.add_column("Detail", overflow="fold")

    colour = {
        _STATUS_PASS: "[green]PASS[/green]",
        _STATUS_FAIL: "[red]FAIL[/red]",
        _STATUS_SKIP: "[dim]SKIP[/dim]",
    }
    for row in rows:
        table.add_row(
            str(row["model"]),
            str(row["flavour"]),
            str(row["file"]),
            colour[str(row["status"])],
            f"{row['ms']:.0f}",
            str(row["detail"]),
        )
    console.print(table)


# %% Main function
def main(include_jetson: bool) -> int:
    """Verify every exported artefact found next to the trained weights.

    Args:
        include_jetson: Force verification of ``*_jetson.engine`` artefacts even
            when this machine is not a Jetson.

    Returns:
        Process exit code - ``0`` when every artefact passed.
    """
    console = Console()
    config = load_config(_CONFIG_PATH)

    models_dir = resolve_path(config["models_dir"])
    naming = config["naming"]
    verify_cfg = config["verify"]
    imgsz = int(config["export"]["imgsz"])
    batch = int(verify_cfg["batch"])
    strict = bool(verify_cfg["strict_onnx_shapes"])

    with_jetson = include_jetson or _TEGRA_RELEASE_PATH.is_file()

    console.print("[bold]Jetson pipeline - export verification[/bold]")
    console.print(f"[dim]Scanned: {models_dir}[/dim]")
    console.print(
        f"[dim]Expected ONNX outputs: {verify_cfg['expected_shapes']} "
        f"(strict={strict}), overrides: "
        f"{sorted((verify_cfg.get('expected_shapes_by_version') or {}).keys())}[/dim]"
    )
    if not with_jetson:
        console.print(
            "[dim]Skipping *_jetson.engine artefacts - they only load on the "
            "Jetson. Pass --include-jetson to force.[/dim]"
        )

    artefacts = discover_artefacts(models_dir, naming, with_jetson)
    if not artefacts:
        console.print("[red]No exported artefacts found - run export_all.py first.[/red]")
        return 1

    rows: list[dict[str, Any]] = []
    for artefact in artefacts:
        path: Path = artefact["path"]
        started = time.perf_counter()
        if artefact["kind"] == "onnx":
            version, _ = split_model_name(str(artefact["model"]))
            status, detail = verify_onnx(
                path, expected_shapes_for(version, verify_cfg),
                batch, imgsz, strict, console,
            )
        else:
            status, detail = verify_engine(path, imgsz)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        marker = "[green]PASS[/green]" if status == _STATUS_PASS else "[red]FAIL[/red]"
        console.print(f"  {marker}  {path.name} - {detail}")
        rows.append({
            "model": artefact["model"],
            "flavour": artefact["flavour"],
            "file": path.name,
            "status": status,
            "ms": elapsed_ms,
            "detail": detail,
        })

    print_summary(rows, console)
    failures = sum(1 for row in rows if row["status"] == _STATUS_FAIL)
    passes = sum(1 for row in rows if row["status"] == _STATUS_PASS)
    console.print(f"[bold]{passes} passed, {failures} failed[/bold] of {len(rows)}")
    if failures:
        console.print("[red]Verification failed.[/red]")
        return 1
    console.print("[bold green]All exports verified.[/bold green]")
    return 0


# %% Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load every exported artefact and run one forward pass.",
    )
    parser.add_argument(
        "--include-jetson", action="store_true",
        help="Also verify *_jetson.engine artefacts on a non-Jetson machine.",
    )
    args = parser.parse_args()
    sys.exit(main(include_jetson=bool(args.include_jetson)))
