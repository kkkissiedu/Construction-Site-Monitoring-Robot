# %%
"""export_on_device.py - re-compile transferred ONNX graphs on the Jetson.

TensorRT engines are tied to the GPU architecture, driver and TensorRT build
that produced them, so the laptop's ``.engine`` files will not load on an Orin.
This script takes the ``*_crack_finetuned.onnx`` graphs transferred from the
laptop and rebuilds them for the device:

* FP16 -> ``{name}_crack_finetuned_jetson.engine``
* INT8 -> ``{name}_crack_int8_jetson.engine`` (reusing the transferred
  calibration cache, or building one on-device from the test split)

Only the raw TensorRT Python API is used - neither Ultralytics nor PyTorch is
required, since Jetson inference does not go through the Ultralytics wrapper.
``rich`` is used when present and falls back to plain text when it is not.

Usage:
    python3 export_on_device.py
    python3 export_on_device.py --check-only   # print the transfer checklist and exit
"""

from __future__ import annotations

# %% Imports
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import tensorrt as trt
import yaml

from calibrate_int8 import (
    CrackCalibrator,
    build_engine_from_onnx,
    cache_path_for,
    collect_calibration_images,
    patch_trt_nptype,
    trt_major_version,
)

# %% Config
_HERE: Path = Path(__file__).parent
_CONFIG_PATH: Path = _HERE / "jetson_config.yaml"

# Canonical size ordering, shared with the rest of the pipeline.
_SIZE_ORDER: tuple[str, ...] = ("n", "s", "m", "l", "x")

# Sources consulted for the JetPack / L4T release, in order of preference.
_TEGRA_RELEASE_PATH: Path = Path("/etc/nv_tegra_release")
_BOOT_CONTROL_PATH: Path = Path("/etc/nv_boot_control.conf")
_CUDA_VERSION_JSON: Path = Path("/usr/local/cuda/version.json")

# L4T major release -> JetPack line. Orin Nano Super ships JetPack 6.x (L4T
# R36); older entries keep the report useful on a mixed fleet.
_L4T_TO_JETPACK: dict[str, str] = {
    "36": "6.x",
    "35": "5.x",
    "34": "5.0-DP",
    "32": "4.x",
}


# %%
def _encodable(text: str, fallback: str) -> str:
    """Return ``text`` when the active stdout encoding can render it.

    The Jetson's terminal is UTF-8, but a dry ``--check-only`` run on a Windows
    console lands on cp1252, where the tick and cross raise
    ``UnicodeEncodeError`` mid-report. Degrading to ASCII keeps the checklist
    readable everywhere instead of aborting the run over decoration.

    Args:
        text: The preferred glyph.
        fallback: ASCII replacement used when the glyph is not encodable.

    Returns:
        ``text`` or ``fallback``.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return fallback
    return text


# Marks used by the transfer checklist. The ASCII fallbacks are deliberately
# bracket-free: rich parses square brackets as style markup and would swallow
# them.
_MARK_OK: str = _encodable("✔", "OK")
_MARK_MISSING: str = _encodable("✗", "XX")

# Strips rich markup when rendering in plain-text mode.
_MARKUP_RE: re.Pattern[str] = re.compile(r"\[/?[a-z0-9_ #]+\]")

# TensorRT 10 removed trt.nptype; restore it before any engine work.
patch_trt_nptype()


# %% Optional rich reporting
class Reporter:
    """Console output that uses ``rich`` when available, plain text otherwise.

    JetPack images do not ship ``rich``, and this script must run on a stock
    device. Importing it optionally keeps the nicer tables on machines that
    have it without making the script undeployable on ones that do not.

    Attributes:
        console: The ``rich`` console, or ``None`` in plain-text mode.
    """

    def __init__(self) -> None:
        """Set up the reporter, preferring ``rich`` when it is importable."""
        self.console: Any = None
        self._table_cls: Any = None
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            return
        self.console = Console()
        self._table_cls = Table

    def line(self, message: str) -> None:
        """Print one line, stripping markup in plain-text mode.

        Args:
            message: The line to print, optionally carrying rich markup.
        """
        if self.console is not None:
            self.console.print(message)
        else:
            print(_MARKUP_RE.sub("", message))

    def table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        """Render a table with ``rich`` or as aligned plain text.

        Args:
            title: Table title.
            columns: Column headers.
            rows: Row cells, one list per row.
        """
        if self.console is not None and self._table_cls is not None:
            table = self._table_cls(title=title, title_style="bold white")
            for index, column in enumerate(columns):
                table.add_column(column, style="bold" if index == 0 else "")
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
            return

        plain = [[_MARKUP_RE.sub("", cell) for cell in row] for row in rows]
        widths = [
            max([len(columns[index])] + [len(row[index]) for row in plain])
            for index in range(len(columns))
        ]
        print(f"\n{title}")
        print("  ".join(header.ljust(widths[index]) for index, header in enumerate(columns)))
        print("  ".join("-" * width for width in widths))
        for row in plain:
            print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
        print()


# %% Environment detection
def read_l4t_release() -> str:
    """Read the L4T release string from the Tegra release file.

    Returns:
        The raw release text, or an empty string when no such file exists
        (i.e. this is not a Jetson).
    """
    for path in (_TEGRA_RELEASE_PATH, _BOOT_CONTROL_PATH):
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
    return ""


# %%
def parse_jetpack_version(release_text: str) -> str:
    """Derive a JetPack version from an L4T release string.

    ``/etc/nv_tegra_release`` looks like
    ``# R36 (release), REVISION: 4.0, GCID: ..., BOARD: generic, ...`` - the
    R-number identifies the L4T major release, which maps onto a JetPack line.

    Args:
        release_text: Raw contents of the release file.

    Returns:
        A version string such as ``"6.x (L4T R36.4.0)"``, or ``"unknown"``.
    """
    if not release_text:
        return "unknown"
    match = re.search(r"R(\d+)\D+REVISION:\s*([\d.]+)", release_text)
    if match is not None:
        major, revision = match.group(1), match.group(2)
        return f"{_L4T_TO_JETPACK.get(major, 'unknown')} (L4T R{major}.{revision})"
    match = re.search(r"R(\d+)", release_text)
    if match is None:
        return "unknown"
    major = match.group(1)
    return f"{_L4T_TO_JETPACK.get(major, 'unknown')} (L4T R{major})"


# %%
def detect_cuda_version() -> str:
    """Detect the installed CUDA toolkit version without requiring PyTorch.

    Three sources are tried: ``torch.version.cuda`` when torch happens to be
    installed, then ``/usr/local/cuda/version.json``, then ``nvcc --version``.

    Returns:
        The CUDA version string, or ``"unknown"``.
    """
    try:
        import torch

        if getattr(torch.version, "cuda", None):
            return str(torch.version.cuda)
    except Exception:  # noqa: BLE001 - torch is not expected on the Jetson
        pass

    try:
        payload = json.loads(_CUDA_VERSION_JSON.read_text(encoding="utf-8"))
        version = payload.get("cuda", {}).get("version")
        if version:
            return str(version)
    except (OSError, ValueError):
        pass

    try:
        completed = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10, check=False,
        )
        match = re.search(r"release\s+([\d.]+)", completed.stdout)
        if match is not None:
            return match.group(1)
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


# %%
def detect_gpu_properties() -> dict[str, Any]:
    """Detect GPU name, compute capability and total memory.

    ``pycuda`` is the JetPack default; ``cuda-python`` and ``torch`` are tried
    afterwards so the same report works on the laptop.

    Returns:
        A dict with ``name``, ``sm`` and ``vram_mib`` keys, falling back to
        ``"unknown"`` / ``0`` when no CUDA runtime is importable.
    """
    try:
        import pycuda.autoinit  # noqa: F401 - the import creates the CUDA context
        import pycuda.driver as cuda

        device = cuda.Device(0)
        major, minor = device.compute_capability()
        return {
            "name": device.name(),
            "sm": f"{major}.{minor}",
            "vram_mib": int(device.total_memory() // (1024 * 1024)),
        }
    except Exception:  # noqa: BLE001 - fall through to the next runtime
        pass

    try:
        from cuda import cudart

        status, properties = cudart.cudaGetDeviceProperties(0)
        if int(status) == 0:
            name = properties.name
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            return {
                "name": str(name).strip("\x00"),
                "sm": f"{properties.major}.{properties.minor}",
                "vram_mib": int(properties.totalGlobalMem // (1024 * 1024)),
            }
    except Exception:  # noqa: BLE001 - fall through to the next runtime
        pass

    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            return {
                "name": properties.name,
                "sm": f"{properties.major}.{properties.minor}",
                "vram_mib": int(properties.total_memory // (1024 * 1024)),
            }
    except Exception:  # noqa: BLE001 - no CUDA runtime at all
        pass

    return {"name": "unknown", "sm": "unknown", "vram_mib": 0}


# %%
def detect_jetson_environment() -> dict[str, Any]:
    """Detect JetPack version, TensorRT version, CUDA version, available VRAM.

    Returns:
        A dict with ``jetpack``, ``tensorrt``, ``trt_major``, ``cuda``,
        ``gpu_name``, ``gpu_sm``, ``vram_mib`` and ``is_jetson`` keys.
    """
    release_text = read_l4t_release()
    gpu = detect_gpu_properties()
    return {
        "jetpack": parse_jetpack_version(release_text),
        "tensorrt": str(getattr(trt, "__version__", "unknown")),
        "trt_major": trt_major_version(),
        "cuda": detect_cuda_version(),
        "gpu_name": gpu["name"],
        "gpu_sm": gpu["sm"],
        "vram_mib": gpu["vram_mib"],
        "is_jetson": bool(release_text),
    }


# %%
def print_environment_report(
    environment: dict[str, Any],
    batch: int,
    reporter: Reporter,
) -> None:
    """Print the full environment report before any export runs.

    Args:
        environment: The dict from :func:`detect_jetson_environment`.
        batch: The fixed inference batch size from the config.
        reporter: Output reporter.
    """
    reporter.line(f"[jetson-env] JetPack: {environment['jetpack']}")
    reporter.line(f"[jetson-env] TensorRT: {environment['tensorrt']}")
    reporter.line(f"[jetson-env] CUDA: {environment['cuda']}")
    reporter.line(
        f"[jetson-env] GPU: {environment['gpu_name']} (SM {environment['gpu_sm']}), "
        f"{environment['vram_mib']} MiB VRAM"
    )
    reporter.line(f"[jetson-env] Recommended batch size: {batch} (fixed for inference)")
    if int(environment["trt_major"]) >= 10:
        api = "10.x (implicit explicit-batch, memory-pool workspace)"
    else:
        api = "8.x (EXPLICIT_BATCH flag, max_workspace_size)"
    reporter.line(f"[jetson-env] Builder API path: TensorRT {api}")
    if not environment["is_jetson"]:
        reporter.line(
            "[jetson-env] [yellow]No /etc/nv_tegra_release - this does not look "
            "like a Jetson. Engines built here will not run on the device.[/yellow]"
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
def parse_onnx_stem(stem: str, finetuned_suffix: str) -> tuple[str, str] | None:
    """Parse a transferred ONNX stem into ``(version, size_letter)``.

    Args:
        stem: Filename without extension.
        finetuned_suffix: Stem suffix marking a fine-tuned export.

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
        model: A model dict from :func:`discover_transferred_models`.

    Returns:
        ``(version, size_index)``.
    """
    return str(model["version"]), _SIZE_ORDER.index(str(model["size"]))


# %%
def discover_transferred_models(
    models_dir: Path,
    finetuned_suffix: str,
) -> list[dict[str, Any]]:
    """Find every ONNX graph transferred onto the device.

    Args:
        models_dir: Directory the ONNX files were copied into.
        finetuned_suffix: Stem suffix marking a fine-tuned export.

    Returns:
        One dict per model with ``name``, ``version``, ``size`` and ``onnx``
        keys, sorted by version then size.
    """
    models: list[dict[str, Any]] = []
    if not models_dir.is_dir():
        return models
    for onnx_path in sorted(models_dir.glob(f"*{finetuned_suffix}.onnx")):
        parsed = parse_onnx_stem(onnx_path.stem, finetuned_suffix)
        if parsed is None:
            continue
        version, size_letter = parsed
        models.append({
            "name": f"{version}{size_letter}-seg",
            "version": version,
            "size": size_letter,
            "onnx": onnx_path,
        })
    models.sort(key=model_sort_key)
    return models


# %% Transfer checklist
def print_transfer_checklist(
    config: dict[str, Any],
    models_dir: Path,
    data_dir: Path,
    models: list[dict[str, Any]],
    reporter: Reporter,
) -> bool:
    """Print exactly which files must already be on the Jetson.

    Args:
        config: The full pipeline configuration.
        models_dir: Directory the ONNX files and caches were copied into.
        data_dir: Root of the processed dataset, needed only for on-device
            calibration.
        models: Discovered models.
        reporter: Output reporter.

    Returns:
        ``True`` when at least one ONNX graph is present. A missing calibration
        cache is reported but is not fatal - it can be rebuilt on-device.
    """
    naming = config["naming"]
    export_cfg = config["export"]
    cache_dir = models_dir / export_cfg["int8_calib_cache"]
    split_dir = data_dir / "images" / export_cfg["int8_calib_split"]

    reporter.line("[transfer-check] Required files on Jetson:")
    if not models:
        reporter.line(
            f"  {_MARK_MISSING} no *{naming['finetuned_suffix']}.onnx under "
            f"{models_dir} - MISSING"
        )
    for model in models:
        name = str(model["name"])
        onnx_path: Path = model["onnx"]
        reporter.line(f"  {_MARK_OK} {onnx_path.name} - found")
        cache_file = cache_path_for(name, cache_dir, naming["cache_suffix"])
        label = f"{export_cfg['int8_calib_cache']}/{cache_file.name}"
        if cache_file.is_file():
            reporter.line(f"  {_MARK_OK} {label} - found")
        else:
            reporter.line(
                f"  {_MARK_MISSING} {label} - MISSING (will be built on-device)"
            )

    images = collect_calibration_images(split_dir, int(export_cfg["int8_calib_images"]))
    if images:
        reporter.line(f"  {_MARK_OK} {split_dir} - found ({len(images)} image(s))")
    else:
        reporter.line(
            f"  {_MARK_MISSING} {split_dir} - MISSING (only needed when a "
            "calibration cache has to be rebuilt here)"
        )
    return bool(models)


# %% Engine building
def build_fp16_engine(
    model: dict[str, Any],
    models_dir: Path,
    config: dict[str, Any],
    reporter: Reporter,
) -> bool:
    """Build the device-native FP16 engine for one model.

    Args:
        model: The model dict being built.
        models_dir: Directory the engine is written into.
        config: The full pipeline configuration.
        reporter: Output reporter.

    Returns:
        ``True`` when a plausible engine exists afterwards.
    """
    name = str(model["name"])
    naming = config["naming"]
    jetson_cfg = config["jetson"]
    engine_path = models_dir / f"{name}{naming['jetson_fp16_suffix']}.engine"

    try:
        build_engine_from_onnx(
            onnx_path=model["onnx"],
            engine_path=engine_path,
            workspace_mib=int(jetson_cfg["workspace_mib"]),
            fp16=True,
            int8=False,
            calibrator=None,
            log=reporter.line,
        )
    except Exception as exc:  # noqa: BLE001 - one bad model must not stop the rest
        reporter.line(f"[red]  FP16  {name}: {exc}[/red]")
        return False

    min_bytes = int(jetson_cfg["min_engine_bytes"])
    if not engine_path.is_file() or engine_path.stat().st_size < min_bytes:
        reporter.line(f"[red]  FP16  {name}: engine missing or implausibly small[/red]")
        return False
    reporter.line(
        f"[green]  FP16  {name}[/green] -> {engine_path.name} "
        f"({engine_path.stat().st_size / 1e6:.1f} MB)"
    )
    return True


# %%
def build_int8_engine(
    model: dict[str, Any],
    models_dir: Path,
    data_dir: Path,
    config: dict[str, Any],
    reporter: Reporter,
) -> bool:
    """Build the device-native INT8 engine for one model.

    The transferred calibration cache is reused when present. When it is
    absent, the same :class:`CrackCalibrator` used on the laptop rebuilds it
    here from the transferred test split, so a missing cache degrades to a
    slower build rather than a failure.

    Args:
        model: The model dict being built.
        models_dir: Directory the engine is written into.
        data_dir: Root of the processed dataset.
        config: The full pipeline configuration.
        reporter: Output reporter.

    Returns:
        ``True`` when a plausible engine exists afterwards.
    """
    name = str(model["name"])
    naming = config["naming"]
    export_cfg = config["export"]
    jetson_cfg = config["jetson"]

    cache_dir = models_dir / export_cfg["int8_calib_cache"]
    cache_file = cache_path_for(name, cache_dir, naming["cache_suffix"])
    engine_path = models_dir / f"{name}{naming['jetson_int8_suffix']}.engine"

    images: list[Path] = []
    if cache_file.is_file():
        reporter.line(f"[dim]  INT8  {name}: reusing {cache_file.name}[/dim]")
    else:
        split_dir = data_dir / "images" / export_cfg["int8_calib_split"]
        images = collect_calibration_images(split_dir, int(export_cfg["int8_calib_images"]))
        if not images:
            reporter.line(
                f"[red]  INT8  {name}: no cache and no images under {split_dir}[/red]"
            )
            return False
        reporter.line(
            f"[dim]  INT8  {name}: calibrating on-device with {len(images)} image(s)[/dim]"
        )

    calibrator: CrackCalibrator | None = None
    try:
        calibrator = CrackCalibrator(
            images, cache_file,
            imgsz=int(export_cfg["imgsz"]),
            batch_size=int(jetson_cfg["batch"]),
        )
        build_engine_from_onnx(
            onnx_path=model["onnx"],
            engine_path=engine_path,
            workspace_mib=int(jetson_cfg["workspace_mib"]),
            fp16=True,
            int8=True,
            calibrator=calibrator,
            log=reporter.line,
        )
    except Exception as exc:  # noqa: BLE001 - one bad model must not stop the rest
        reporter.line(f"[red]  INT8  {name}: {exc}[/red]")
        return False
    finally:
        if calibrator is not None:
            calibrator.close()

    min_bytes = int(jetson_cfg["min_engine_bytes"])
    if not engine_path.is_file() or engine_path.stat().st_size < min_bytes:
        reporter.line(f"[red]  INT8  {name}: engine missing or implausibly small[/red]")
        return False
    reporter.line(
        f"[green]  INT8  {name}[/green] -> {engine_path.name} "
        f"({engine_path.stat().st_size / 1e6:.1f} MB)"
    )
    return True


# %% Reporting
def print_summary(
    models: list[dict[str, Any]],
    results: dict[str, dict[str, bool]],
    models_dir: Path,
    reporter: Reporter,
) -> None:
    """Print the final per-model on-device build summary.

    Args:
        models: Discovered models.
        results: Per-model outcome map.
        models_dir: Directory the engines were written into.
        reporter: Output reporter.
    """
    rows: list[list[str]] = []
    for model in models:
        outcome = results[str(model["name"])]
        rows.append([
            str(model["name"]),
            "[green]OK[/green]" if outcome["fp16"] else "[red]FAIL[/red]",
            "[green]OK[/green]" if outcome["int8"] else "[red]FAIL[/red]",
        ])
    reporter.table("On-device Engine Build", ["Model", "FP16 jetson", "INT8 jetson"], rows)
    reporter.line(f"[dim]Engines written to {models_dir}[/dim]")


# %% Main function
def main(check_only: bool) -> int:
    """Detect the environment, verify transfers, and rebuild every engine.

    Args:
        check_only: Print the environment report and transfer checklist, then
            exit without building anything.

    Returns:
        Process exit code - ``0`` when every attempted build succeeded.
    """
    reporter = Reporter()
    config = load_config(_CONFIG_PATH)

    models_dir = resolve_path(config["models_dir"])
    data_dir = resolve_path(config["data_dir"])
    naming = config["naming"]
    export_cfg = config["export"]

    environment = detect_jetson_environment()
    print_environment_report(environment, int(config["jetson"]["batch"]), reporter)

    models = discover_transferred_models(models_dir, naming["finetuned_suffix"])
    has_work = print_transfer_checklist(config, models_dir, data_dir, models, reporter)
    if check_only:
        return 0 if has_work else 1
    if not has_work:
        reporter.line(
            "[red]Nothing to build - transfer the ONNX files listed above first.[/red]"
        )
        return 1

    results: dict[str, dict[str, bool]] = {
        str(model["name"]): {"fp16": False, "int8": False} for model in models
    }

    reporter.line("[bold]Stage 1/2 - FP16 engines[/bold]")
    for model in models:
        results[str(model["name"])]["fp16"] = build_fp16_engine(
            model, models_dir, config, reporter,
        )

    reporter.line("[bold]Stage 2/2 - INT8 engines[/bold]")
    if export_cfg["trt_int8"]:
        for model in models:
            results[str(model["name"])]["int8"] = build_int8_engine(
                model, models_dir, data_dir, config, reporter,
            )
    else:
        reporter.line("[yellow]  trt_int8 disabled in config - stage skipped[/yellow]")

    print_summary(models, results, models_dir, reporter)
    failures = sum(
        1
        for model in models
        for key in ("fp16", "int8")
        if not results[str(model["name"])][key]
    )
    if failures:
        reporter.line(f"[red]{failures} build(s) failed - see the table above.[/red]")
        return 1
    reporter.line("[bold green]All on-device engines built.[/bold green]")
    return 0


# %% Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild transferred ONNX graphs into Jetson-native TensorRT engines.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Print the environment report and transfer checklist, then exit.",
    )
    args = parser.parse_args()
    sys.exit(main(check_only=bool(args.check_only)))
