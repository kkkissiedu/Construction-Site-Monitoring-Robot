# %%
"""GhanaCrack desktop application.

A tkinter-shelled, OpenCV-rendered crack-detection application with three
screens (Main Menu, Run Screen, History) and on-demand per-model parallel
inference via dedicated worker threads with per-model CUDA streams.

Run with::

    python app.py
"""

from __future__ import annotations

import ctypes
import sys




def enable_dpi_awareness() -> None:
    """Enable Per-Monitor DPI awareness on Windows to prevent blurry scaling."""
    if sys.platform == "win32":
        try:
            # Per-Monitor V2 awareness - best mode, Windows 10 1703+.
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                # Fallback: system DPI awareness.
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


# Must run before tkinter is imported - awareness is locked once Tk inits.
enable_dpi_awareness()


def get_dpi_scale() -> float:
    """Return the DPI scale factor for the primary monitor (e.g. 1.5 for 150%)."""
    if sys.platform == "win32":
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
            return dpi / 96.0
        except Exception:
            return 1.0
    return 1.0


DPI_SCALE: float = get_dpi_scale()


def px(value: int) -> int:
    """Scale a pixel value by the DPI scale factor."""
    return int(value * DPI_SCALE)


import csv
import functools
import json
import queue
import shutil
import threading
import time
import tkinter as tk
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import customtkinter as ctk
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import torch
import yaml
from PIL import Image, ImageTk
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as RLTable,
    TableStyle,
)
from ultralytics import FastSAM, YOLO

# ``onnxruntime`` is optional. If absent the ONNX backend simply stays
# unavailable rather than crashing the whole app on import.
try:
    import onnxruntime as ort  # type: ignore[import-not-found]
    _ONNXRUNTIME_AVAILABLE: bool = True
except Exception:  # noqa: BLE001 - presence check only
    ort = None  # type: ignore[assignment]
    _ONNXRUNTIME_AVAILABLE = False


# Shim: silence Ultralytics' ``check_requirements(['onnxruntime'])`` when ORT
# is already importable. Users on CUDA usually have ``onnxruntime-gpu``
# installed - that's a different distribution name, so Ultralytics' default
# pkg-metadata check fails and triggers a noisy ``pip install onnxruntime``
# auto-update loop every time a ``YOLO('*.onnx')`` is constructed. We patch
# the check so the runtime requirement is treated as met whenever the module
# can be imported, regardless of which distribution provides it.
if _ONNXRUNTIME_AVAILABLE:
    try:
        import sys as _sys

        import ultralytics.utils.checks as _ult_checks

        _orig_check_requirements = _ult_checks.check_requirements

        def _requirement_is_onnxruntime(requirement: Any) -> bool:
            """Return True if ``requirement`` names the onnxruntime package.

            Args:
                requirement: A requirement spec as Ultralytics passes it
                    (``"onnxruntime"``, ``"onnxruntime>=1.16"``, etc.).
            """
            token = (
                str(requirement).lower().strip()
                .split(">", 1)[0].split("=", 1)[0]
                .split("<", 1)[0].split("!", 1)[0].split("~", 1)[0]
                .strip()
            )
            return token == "onnxruntime"

        def _check_requirements_no_ort(
            requirements: Any = "requirements.txt",
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Wrapper around Ultralytics' check_requirements.

            Drops any ``onnxruntime`` entry from the requirement list when
            ``onnxruntime`` is already importable - this avoids the noisy
            ``pip install onnxruntime`` retry loop on systems that ship
            ``onnxruntime-gpu`` instead. Other requirements still go
            through the original check.
            """
            if isinstance(requirements, (list, tuple)):
                filtered = [
                    requirement
                    for requirement in requirements
                    if not _requirement_is_onnxruntime(requirement)
                ]
                if not filtered:
                    return True
                return _orig_check_requirements(filtered, *args, **kwargs)
            if isinstance(requirements, str) and _requirement_is_onnxruntime(
                requirements,
            ):
                return True
            return _orig_check_requirements(requirements, *args, **kwargs)

        _ult_checks.check_requirements = _check_requirements_no_ort
        # Several Ultralytics submodules (e.g. ``nn.autobackend``) do
        # ``from ultralytics.utils.checks import check_requirements`` at
        # module-load time, which captures a reference to the *original*
        # function before our patch lands. Re-bind those module-level
        # references so the patch actually takes effect at call time.
        for _module_name, _module in list(_sys.modules.items()):
            if (
                _module is None
                or not _module_name.startswith("ultralytics")
                or _module_name == "ultralytics.utils.checks"
            ):
                continue
            if getattr(_module, "check_requirements", None) is _orig_check_requirements:
                _module.check_requirements = _check_requirements_no_ort
    except Exception:  # noqa: BLE001 - best-effort patch; never block startup
        pass


# Shim: TensorRT 10 removed trt.nptype; Ultralytics still calls it internally.
# Injecting a compatible replacement before any YOLO(.engine) call.
try:
    import tensorrt as trt
    if not hasattr(trt, "nptype"):
        import numpy as np
        def _trt_dtype_to_np(dtype) -> "np.dtype":
            """Map TensorRT DataType to numpy dtype (replaces removed trt.nptype)."""
            mapping = {
                trt.DataType.FLOAT: np.float32,
                trt.DataType.HALF:  np.float16,
                trt.DataType.INT8:  np.int8,
                trt.DataType.INT32: np.int32,
                trt.DataType.BOOL:  np.bool_,
            }
            return mapping.get(dtype, np.float32)
        trt.nptype = _trt_dtype_to_np
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Directory / file locations
# ---------------------------------------------------------------------------
_BASE_DIR: Path = Path(__file__).parent
_MODELS_DIR: Path = _BASE_DIR / "models"
_INPUTS_DIR: Path = _BASE_DIR / "inputs"
_OUTPUTS_DIR: Path = _BASE_DIR / "outputs"
_SNAPSHOTS_DIR: Path = _OUTPUTS_DIR / "snapshots"
_RESULTS_DIR: Path = _BASE_DIR / "results"
_EXPORTS_DIR: Path = _RESULTS_DIR / "exports"
_HISTORY_PATH: Path = _RESULTS_DIR / "history.json"
_BENCHMARK_HISTORY_PATH: Path = _RESULTS_DIR / "benchmark_history.json"

# ---------------------------------------------------------------------------
# Window / theme constants
# ---------------------------------------------------------------------------
_WINDOW_TITLE: str = "GhanaCrack"
_BG_DARK: str = "#0D0D0D"
_BG_PANEL: str = "#1A1A1A"
_BG_HOVER: str = "#222222"
_FG_DEFAULT: str = "#EEEEEE"
_FG_GREY: str = "#888888"
_ORANGE: str = "#E87722"
_TEAL: str = "#00B4A6"
_BLUE: str = "#3498DB"
_DARK_RED: str = "#7E2A1B"
_YELLOW: str = "#F1C40F"

# ---------------------------------------------------------------------------
# Main-menu (customtkinter) style constants
# ---------------------------------------------------------------------------
BG_PRIMARY: str = "#0D0D0D"
BG_SECONDARY: str = "#141414"
BG_CARD: str = "#1A1A1A"
BG_HOVER: str = "#1E1E1E"
ACCENT_ORANGE: str = "#E87722"
ACCENT_TEAL: str = "#00B4A6"
ACCENT_BLUE: str = "#3498DB"
ACCENT_GREEN: str = "#2ECC71"
ACCENT_GREEN_LARGE: str = "#2ECC71"   # Large model accent colour
ACCENT_RED: str = "#E74C3C"
ACCENT_YELLOW: str = "#F39C12"
TEXT_PRIMARY: str = "#FFFFFF"
TEXT_MUTED: str = "#888888"
BORDER: str = "#2A2A2A"
APP_VERSION: str = "v1.0.0"

# Full-benchmark chart: fixed combo order + per-combo bar colours. The chart
# spec is shared between the post-run modal preview and the Benchmarks
# history screen so colours stay consistent everywhere a chart is drawn.
FULL_BENCH_COMBO_ORDER: tuple[tuple[str, str], ...] = (
    ("pytorch", "fp32"),
    ("pytorch", "fp16"),
    ("onnx", "fp32"),
    ("tensorrt", "fp16"),
    ("tensorrt", "int8"),
)
FULL_BENCH_COMBO_COLORS: dict[tuple[str, str], str] = {
    ("pytorch", "fp32"): "#E87722",
    ("pytorch", "fp16"): "#F5A245",
    ("onnx", "fp32"): "#00B4A6",
    ("tensorrt", "fp16"): "#2ECC71",
    ("tensorrt", "int8"): "#7BE495",
}
FULL_BENCH_NA_COLOR: str = "#333333"

# DPI-aware font sizes - tuned for 150% Windows scaling on high-res displays.
FONT_TITLE      = ("Segoe UI", 28, "bold")   # GhanaCrack title in sidebar
FONT_SUBTITLE   = ("Segoe UI", 13)            # "Crack Detection" subtitle
FONT_NAV        = ("Segoe UI", 14)            # sidebar nav items
FONT_STATUS     = ("Segoe UI", 12)            # sidebar model status rows
FONT_VERSION    = ("Segoe UI", 11)            # v1.0.0
FONT_SECTION    = ("Segoe UI", 14, "bold")    # orange card section titles
FONT_SUBLABEL   = ("Segoe UI", 12)            # subsection labels inside cards
FONT_WIDGET     = ("Segoe UI", 13)            # radios, checkboxes, switches, option menus
FONT_BUTTON     = ("Segoe UI", 14, "bold")    # Start Run, Run Benchmark, Quit
FONT_SMALL      = ("Segoe UI", 11)            # selected count, muted labels
FONT_HUD        = ("Segoe UI", 12)            # benchmark modal log, progress text
FONT_MODAL_TITLE= ("Segoe UI", 20, "bold")    # Benchmark Running modal title

# Configure customtkinter globally before any widgets are constructed.
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

_DISPLAY_WIDTH: int = 1280
_DISPLAY_HEIGHT: int = 720
# Physical canvas size at native pixel density (DPI-scaled).
CANVAS_W: int = px(_DISPLAY_WIDTH)
CANVAS_H: int = px(_DISPLAY_HEIGHT)
_PANEL_WIDTH: int = CANVAS_W // 3
_PANEL_HEIGHT: int = CANVAS_H


def _split_panel_dims(model_count: int) -> tuple[int, int]:
    """Return per-panel ``(width, height)`` for the split-screen layout.

    1, 2 or 3 selected models keep the historic horizontal strip; 4 models
    switch to a 2x2 grid so every quadrant gets ``CANVAS_W // 2`` by
    ``CANVAS_H // 2``.

    Args:
        model_count: Number of models being rendered in the split.

    Returns:
        ``(panel_w, panel_h)`` in pixels.
    """
    if model_count <= 1:
        return CANVAS_W, CANVAS_H
    if model_count == 4:
        return CANVAS_W // 2, CANVAS_H // 2
    return CANVAS_W // model_count, CANVAS_H


def _composite_split_panels(
    panels: list[np.ndarray], panel_w: int, panel_h: int,
) -> np.ndarray:
    """Composite per-model panels into a single frame for display.

    Args:
        panels: Pre-rendered panels in selection order.
        panel_w: Width each panel was rendered at.
        panel_h: Height each panel was rendered at.

    Returns:
        The composited frame. Empty input falls back to a black canvas.
    """
    if not panels:
        return np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    if len(panels) == 4:
        top_row = np.hstack([panels[0], panels[1]])
        bottom_row = np.hstack([panels[2], panels[3]])
        return np.vstack([top_row, bottom_row])
    composite = np.zeros((panel_h, panel_w * len(panels), 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        composite[:, index * panel_w : (index + 1) * panel_w] = panel
    return composite

_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov")

# Rendering / inference constants
_MASK_ALPHA: float = 0.4
_BBOX_THICKNESS: int = 2
_MASK_BINARY_THRESHOLD: float = 0.5
_IOU_DEFAULT: float = 0.45
_CONF_STEP: float = 0.05
_CONF_MIN: float = 0.05
_CONF_MAX: float = 0.95
_FPS_WINDOW: int = 30
_WARMUP_ITERATIONS: int = 10
# Frame budget for the kernel-only second pass inside ``benchmark_one_combo``.
# 200 frames balances statistical noise against the extra dialog wait time.
_KERNEL_TIMING_FRAMES: int = 200
_DISPLAY_TICK_MS: int = 15
_VIDEO_TRANSITION_SECONDS: float = 2.0
_SUMMARY_AUTO_RETURN_SECONDS: int = 15
_SPLIT_WORKER_TIMEOUT: float = 0.5
_WORKER_POLL_TIMEOUT: float = 0.1

# OpenCV BGR colours
_BGR_WHITE: tuple[int, int, int] = (255, 255, 255)
_BGR_RED: tuple[int, int, int] = (0, 0, 255)
_BGR_PROGRESS_BG: tuple[int, int, int] = (60, 60, 60)
_HUD_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
_HUD_PADDING: int = 8
_HUD_LINE_HEIGHT: int = 22
_HUD_PANEL_ALPHA: float = 0.55
_PROGRESS_BAR_HEIGHT: int = 4

# Model registry. ``finetuned`` lists candidate fine-tuned filenames in
# priority order; the first one found wins, otherwise ``pretrained`` is used
# and a yellow warning badge is shown on the main menu.
# Model registry: three seg model sizes (nano / small / medium). Every backend
# variant of each size shares this spec - render mode, default confidence and
# colour live here so the app's rendering helpers stay agnostic to which
# backend produced the inference result.
_MODEL_SPECS: list[dict[str, Any]] = [
    {
        "key": "yolo11n-seg",
        "display": "YOLO11n-seg",
        "size": "nano",
        "size_label": "Nano",
        "finetuned": ["yolo11n-seg_crack_finetuned.pt", "yolo11n_seg_crack_finetuned.pt"],
        "pretrained": "yolo11n-seg.pt",
        "is_fastsam": False,
        "default_conf": 0.25,
        "color_hex": _ORANGE,
        "render": "polygon",
    },
    {
        "key": "yolo11s-seg",
        "display": "YOLO11s-seg",
        "size": "small",
        "size_label": "Small",
        "finetuned": ["yolo11s-seg_crack_finetuned.pt"],
        "pretrained": "yolo11s-seg.pt",
        "is_fastsam": False,
        "default_conf": 0.25,
        "color_hex": _TEAL,
        "render": "polygon",
    },
    {
        "key": "yolo11m-seg",
        "display": "YOLO11m-seg",
        "size": "medium",
        "size_label": "Medium",
        "finetuned": ["yolo11m-seg_crack_finetuned.pt"],
        "pretrained": "yolo11m-seg.pt",
        "is_fastsam": False,
        "default_conf": 0.25,
        "color_hex": _BLUE,
        "render": "polygon",
    },
    {
        "key": "yolo11l-seg",
        "display": "YOLO11l-seg (Large)",
        "size": "large",
        "size_label": "Large",
        "finetuned": ["yolo11l-seg_crack_finetuned.pt"],
        "pretrained": "yolo11l-seg.pt",
        "is_fastsam": False,
        "default_conf": 0.25,
        "color_hex": ACCENT_GREEN_LARGE,
        "render": "polygon",
        "default_checked": False,
    },
]
_MODEL_BY_KEY: dict[str, dict[str, Any]] = {spec["key"]: spec for spec in _MODEL_SPECS}

# ---------------------------------------------------------------------------
# Auto-discovery: filesystem-based model registry
# ---------------------------------------------------------------------------
# Filename convention scanned by ``discover_models``:
#     {version}{size_letter}-seg_crack_finetuned.{ext}
# Example: ``yolo11n-seg_crack_finetuned.pt`` -> version=yolo11, size=n.
_SIZE_LETTER_TO_KEY: dict[str, str] = {
    "n": "nano",
    "s": "small",
    "m": "medium",
    "l": "large",
    "x": "xlarge",
}
_SIZE_LETTER_TO_LABEL: dict[str, str] = {
    "n": "Nano",
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "x": "XLarge",
}
_SIZE_ORDER: tuple[str, ...] = ("n", "s", "m", "l", "x")

# Per-version accent palette. Known versions get a fixed colour; unknown
# versions cycle through ``_VERSION_PALETTE_FALLBACK`` in discovery order.
_VERSION_ACCENT: dict[str, str] = {
    "yolo11": "#E87722",   # orange
    "yolo12": "#9B59B6",   # purple
    "yolo26": "#00B4A6",   # teal
}
# Purple moved into _VERSION_ACCENT for yolo12, so an unknown version cannot
# claim it and collide.
_VERSION_PALETTE_FALLBACK: tuple[str, ...] = (
    "#3498DB",   # blue (ACCENT_BLUE)
    "#2ECC71",   # green
    "#E91E63",   # pink
)

# Per-backend extension recognised by discovery. Order matters only for the
# discovery summary log line.
_BACKEND_TO_EXTENSION: dict[str, str] = {
    "pytorch": ".pt",
    "onnx": ".onnx",
    "tensorrt": ".engine",
}
_FINETUNED_STEM_SUFFIX: str = "_crack_finetuned"

# INT8 engines carry their own stem suffix so they sit alongside the FP16
# engine instead of overwriting it. They are tracked under a dedicated
# registry key because ``(model, "tensorrt")`` already points at the FP16
# engine; the UI still presents INT8 as a *precision* of the TensorRT backend.
_INT8_STEM_SUFFIX: str = "_crack_int8"
_BACKEND_TENSORRT_INT8: str = "tensorrt_int8"


# %%
def _parse_model_filename(stem: str) -> tuple[str, str] | None:
    """Parse a ``*_crack_finetuned`` stem into ``(version, size_letter)``.

    Args:
        stem: Filename without extension, e.g. ``yolo11n-seg_crack_finetuned``.

    Returns:
        ``(version, size_letter)`` if the stem matches the convention,
        otherwise ``None``. Unknown size letters return ``None`` so unrelated
        files do not pollute the registry.
    """
    if not stem.endswith(_FINETUNED_STEM_SUFFIX):
        return None
    head = stem[: -len(_FINETUNED_STEM_SUFFIX)]
    if not head.endswith("-seg"):
        return None
    head = head[: -len("-seg")]
    if not head:
        return None
    size_letter = head[-1]
    if size_letter not in _SIZE_LETTER_TO_KEY:
        return None
    version = head[:-1]
    if not version:
        return None
    return version, size_letter


# %%
def discover_models(models_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Scan ``models_dir`` for finetuned weights and return a nested registry.

    The registry shape is::

        {
          "yolo11": {
            "n": {
              "label": "Nano",
              "key": "yolo11n-seg",
              "pytorch": Path("models/yolo11n-seg_crack_finetuned.pt"),
              "onnx":    Path("models/yolo11n-seg_crack_finetuned.onnx") or None,
              "tensorrt":Path("models/yolo11n-seg_crack_finetuned.engine") or None,
              "tensorrt_int8": Path("models/yolo11n-seg_crack_int8.engine") or None,
            },
            ...
          },
          "yolo26": {...},
        }

    Args:
        models_dir: Directory to scan (non-recursively).

    Returns:
        The nested ``{version: {size_letter: spec}}`` mapping. Empty when the
        directory does not exist or contains no recognised files - never
        raises.
    """
    registry: dict[str, dict[str, dict[str, Any]]] = {}
    if not models_dir.is_dir():
        return registry
    try:
        candidates = sorted(models_dir.glob(f"*{_FINETUNED_STEM_SUFFIX}.pt"))
    except OSError:
        return registry
    for pt_path in candidates:
        parsed = _parse_model_filename(pt_path.stem)
        if parsed is None:
            continue
        version, size_letter = parsed
        size_entry = registry.setdefault(version, {}).setdefault(
            size_letter,
            {
                "label": _SIZE_LETTER_TO_LABEL[size_letter],
                "key": f"{version}{size_letter}-seg",
                "pytorch": None,
                "onnx": None,
                "tensorrt": None,
                _BACKEND_TENSORRT_INT8: None,
            },
        )
        size_entry["pytorch"] = pt_path
        for backend, ext in _BACKEND_TO_EXTENSION.items():
            if backend == "pytorch":
                continue
            sibling = pt_path.with_suffix(ext)
            if sibling.is_file():
                size_entry[backend] = sibling
        # The INT8 engine has a different stem, so it is resolved by name
        # rather than by swapping the extension.
        int8_engine = pt_path.with_name(
            f"{size_entry['key']}{_INT8_STEM_SUFFIX}.engine"
        )
        if int8_engine.is_file():
            size_entry[_BACKEND_TENSORRT_INT8] = int8_engine
    return registry


# %%
def format_discovery_summary(
    discovered: dict[str, dict[str, dict[str, Any]]],
) -> str:
    """Render the one-line ``[discovery]`` summary printed on startup.

    Args:
        discovered: The mapping returned by :func:`discover_models`.

    Returns:
        A human-readable summary, e.g. ``"yolo11n-seg (PT+ONNX+TRT), ..."``.
        Empty string when nothing was discovered.
    """
    parts: list[str] = []
    backend_short = {
        "pytorch": "PT",
        "onnx": "ONNX",
        "tensorrt": "TRT",
        _BACKEND_TENSORRT_INT8: "TRT-INT8",
    }
    for version in sorted(discovered.keys()):
        for letter in _SIZE_ORDER:
            entry = discovered[version].get(letter)
            if entry is None:
                continue
            tags = [
                backend_short[backend]
                for backend in ("pytorch", "onnx", "tensorrt", _BACKEND_TENSORRT_INT8)
                if entry.get(backend) is not None
            ]
            parts.append(f"{entry['key']} ({'+'.join(tags) or '-'})")
    return ", ".join(parts)


# %%
def version_display(version: str) -> str:
    """Return the title-case display string for a YOLO version key.

    Args:
        version: Lower-case version key, e.g. ``"yolo11"``.

    Returns:
        ``"YOLO11"``-style display string.
    """
    return version.upper()


# %%
def version_accent_color(
    version: str, discovery_order: list[str] | None = None,
) -> str:
    """Return the accent colour for a YOLO version.

    Known versions (``yolo11``, ``yolo26``) get a fixed colour. Unknown
    versions cycle through :data:`_VERSION_PALETTE_FALLBACK` in their
    discovery order so colours stay stable across runs.

    Args:
        version: Version key, e.g. ``"yolo11"``.
        discovery_order: Optional sorted list of versions used to derive a
            stable fallback index.

    Returns:
        A ``#RRGGBB`` colour string.
    """
    if version in _VERSION_ACCENT:
        return _VERSION_ACCENT[version]
    if discovery_order and version in discovery_order:
        index = discovery_order.index(version)
    else:
        index = 0
    return _VERSION_PALETTE_FALLBACK[index % len(_VERSION_PALETTE_FALLBACK)]


# %%
def build_specs_from_discovery(
    discovered: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Compose render-ready model specs from a discovery registry.

    Each spec mirrors the shape of the static :data:`_MODEL_SPECS` entries so
    existing rendering, history, snapshot and worker code keeps working
    without further adaptation. Specs are emitted in (version, size) order.

    Args:
        discovered: The mapping returned by :func:`discover_models`.

    Returns:
        A list of spec dicts. Empty when discovery found nothing.
    """
    specs: list[dict[str, Any]] = []
    sorted_versions = sorted(discovered.keys())
    for version in sorted_versions:
        accent = version_accent_color(version, sorted_versions)
        for letter in _SIZE_ORDER:
            entry = discovered[version].get(letter)
            if entry is None:
                continue
            key = entry["key"]
            size_name = _SIZE_LETTER_TO_KEY[letter]
            size_label = _SIZE_LETTER_TO_LABEL[letter]
            pt_path: Path | None = entry.get("pytorch")
            specs.append({
                "key": key,
                "version": version,
                "version_display": version_display(version),
                "size_letter": letter,
                "size": size_name,
                "size_label": size_label,
                "display": f"{version_display(version)}{letter}-seg",
                "finetuned": [pt_path.name] if pt_path is not None else [
                    f"{key}_crack_finetuned.pt",
                ],
                "pretrained": f"{key.split('-')[0]}-seg.pt",
                "is_fastsam": False,
                "default_conf": 0.25,
                "color_hex": accent,
                "render": "polygon",
            })
    return specs


# ---------------------------------------------------------------------------
# Backend / precision / device constants
# ---------------------------------------------------------------------------
_BACKEND_PYTORCH: str = "pytorch"
_BACKEND_ONNX: str = "onnx"
_BACKEND_TENSORRT: str = "tensorrt"
_BACKENDS: tuple[str, ...] = (_BACKEND_PYTORCH, _BACKEND_ONNX, _BACKEND_TENSORRT)

_BACKEND_DISPLAY: dict[str, str] = {
    _BACKEND_PYTORCH: "PyTorch",
    _BACKEND_ONNX: "ONNX",
    _BACKEND_TENSORRT: "TensorRT",
}

# Accent colours used by the HUD and backend-selector dots.
_BACKEND_COLOR_HEX: dict[str, str] = {
    _BACKEND_PYTORCH: "#E87722",   # orange
    _BACKEND_ONNX: "#00B4A6",      # teal
    _BACKEND_TENSORRT: "#2ECC71",  # green
}

_PRECISION_FP32: str = "fp32"
_PRECISION_FP16: str = "fp16"
_PRECISION_INT8: str = "int8"
_PRECISIONS: tuple[str, ...] = (_PRECISION_FP32, _PRECISION_FP16, _PRECISION_INT8)

# Valid (backend, precision) pairs. Anything missing here is greyed out.
# INT8 exists only as a TensorRT engine - there is no INT8 PyTorch or ONNX
# path in this app - so it is the sole TensorRT-only precision.
_SUPPORTED_BACKEND_PRECISIONS: set[tuple[str, str]] = {
    (_BACKEND_PYTORCH, _PRECISION_FP32),
    (_BACKEND_PYTORCH, _PRECISION_FP16),
    (_BACKEND_ONNX, _PRECISION_FP32),
    (_BACKEND_TENSORRT, _PRECISION_FP16),
    (_BACKEND_TENSORRT, _PRECISION_INT8),
}

# Tooltip shown on the greyed-out INT8 radio.
_INT8_REQUIRES_TRT_TOOLTIP: str = "INT8 requires TensorRT"


# %%
def backend_storage_key(backend: str, precision: str) -> str:
    """Map a UI ``(backend, precision)`` pair onto its weights-registry key.

    TensorRT FP16 and TensorRT INT8 are separate engine files, so they occupy
    separate registry slots even though the UI presents INT8 as a precision of
    the single TensorRT backend.

    Args:
        backend: Backend identifier from the UI.
        precision: Precision identifier from the UI.

    Returns:
        The key under which ``discover_models`` filed that engine.
    """
    if backend == _BACKEND_TENSORRT and precision == _PRECISION_INT8:
        return _BACKEND_TENSORRT_INT8
    return backend

# Default backend/precision shown in the menu on first launch.
_DEFAULT_BACKEND: str = _BACKEND_PYTORCH
_DEFAULT_PRECISION: str = _PRECISION_FP32

# Status-dot colours for backend availability.
_DOT_GREEN: str = "#2ECC71"
_DOT_YELLOW: str = "#F1C40F"
_DOT_RED: str = "#E74C3C"
_DOT_GREY: str = "#444444"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_CONFIG_PATH: Path = _BASE_DIR / "config.yaml"


# %%
def load_app_config(config_path: Path) -> dict[str, Any]:
    """Load and parse ``config.yaml`` with graceful fallback defaults.

    The benchmark/back-end blocks the app relies on are filled in with sane
    defaults when missing so a stripped-down config file still boots.

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        The parsed configuration as a nested dict.
    """
    defaults: dict[str, Any] = {
        "backends": {
            _BACKEND_PYTORCH: {
                spec["key"]: f"models/{spec['finetuned'][0]}"
                for spec in _MODEL_SPECS
            },
            _BACKEND_ONNX: {
                spec["key"]: f"models/{spec['key']}_crack_finetuned.onnx"
                for spec in _MODEL_SPECS
            },
            _BACKEND_TENSORRT: {
                spec["key"]: f"models/{spec['key']}_crack_fp16.engine"
                for spec in _MODEL_SPECS
            },
            _BACKEND_TENSORRT_INT8: {
                spec["key"]: f"models/{spec['key']}{_INT8_STEM_SUFFIX}.engine"
                for spec in _MODEL_SPECS
            },
        },
        "benchmark": {
            "frame_count": 300,
            "warmup_frames": 30,
            "output_csv": "results/benchmark_results.csv",
        },
    }
    if not config_path.is_file():
        return defaults
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return defaults
    merged: dict[str, Any] = dict(raw)
    for key, default_value in defaults.items():
        if key not in merged or not isinstance(merged[key], dict):
            merged[key] = default_value
        elif isinstance(default_value, dict):
            merged_block = dict(default_value)
            merged_block.update(merged[key])
            merged[key] = merged_block
    return merged


# %%
def resolve_backend_paths(
    app_config: dict[str, Any],
    base_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Resolve config-defined backend weight paths to absolute ``Path`` objects.

    Args:
        app_config: The parsed ``config.yaml`` content.
        base_dir: The benchmark directory used as the root for relative paths.

    Returns:
        A ``{backend: {model_key: Path}}`` mapping with every entry resolved.
    """
    backends_cfg = app_config.get("backends", {}) or {}
    resolved: dict[str, dict[str, Path]] = {}
    for backend in _BACKENDS:
        entries = backends_cfg.get(backend, {}) or {}
        resolved[backend] = {
            str(model_key): (base_dir / Path(str(path)) if not Path(str(path)).is_absolute() else Path(str(path)))
            for model_key, path in entries.items()
        }
    return resolved


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
# %%
def hex_to_bgr(hex_string: str) -> tuple[int, int, int]:
    """Convert an ``#RRGGBB`` hex colour to an OpenCV ``(B, G, R)`` tuple.

    Args:
        hex_string: Colour string in ``"#RRGGBB"`` form.

    Returns:
        The colour as ``(B, G, R)`` with components in ``0..255``.
    """
    cleaned = hex_string.lstrip("#")
    r = int(cleaned[0:2], 16)
    g = int(cleaned[2:4], 16)
    b = int(cleaned[4:6], 16)
    return b, g, r


# %%
def resolve_device() -> str:
    """Return ``"cuda"`` when a CUDA GPU is available, otherwise ``"cpu"``.

    Returns:
        ``"cuda"`` if a CUDA-capable GPU is visible, otherwise ``"cpu"``.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# %%
def list_videos(inputs_dir: Path) -> list[Path]:
    """Return every supported video file directly inside ``inputs_dir``.

    Args:
        inputs_dir: Directory to scan (non-recursively).

    Returns:
        A sorted list of video paths.
    """
    if not inputs_dir.is_dir():
        return []
    return sorted(
        path
        for path in inputs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
    )


# %%
def load_model_with_fallback(
    spec: dict[str, Any],
    device: str,
) -> tuple[Any, str, bool]:
    """Load fine-tuned weights for a model, falling back to pretrained on miss.

    Args:
        spec: One of the entries in :data:`_MODEL_SPECS`.
        device: The compute device to move the loaded model onto.

    Returns:
        ``(model, weights_filename, used_finetuned)``.
    """
    for candidate in spec["finetuned"]:
        candidate_path = _MODELS_DIR / candidate
        if candidate_path.is_file():
            model = (
                FastSAM(str(candidate_path))
                if spec["is_fastsam"]
                else YOLO(str(candidate_path))
            )
            model.to(device)
            return model, candidate, True

    pretrained_path = _MODELS_DIR / spec["pretrained"]
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"No weights for {spec['display']} - tried {spec['finetuned']} "
            f"and pretrained {spec['pretrained']} in {_MODELS_DIR}."
        )
    model = (
        FastSAM(str(pretrained_path))
        if spec["is_fastsam"]
        else YOLO(str(pretrained_path))
    )
    model.to(device)
    return model, spec["pretrained"], False


# %%
def warmup_model(model: Any, device: str, iterations: int) -> None:
    """Run repeated dummy inferences so the first real frame isn't a cold start.

    Args:
        model: An Ultralytics model wrapper.
        device: The compute device string passed to ``model.predict``.
        iterations: Number of dummy inferences to execute.
    """
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(max(0, iterations)):
        model.predict(dummy, device=device, verbose=False)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
# %%
def overlay_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Blend a coloured semi-transparent mask onto a BGR frame.

    Args:
        frame_bgr: The base BGR image.
        mask: ``HxW`` mask; pixels above ``_MASK_BINARY_THRESHOLD`` are blended.
        color_bgr: Overlay colour as ``(B, G, R)``.
        alpha: Blend weight applied to the colour (``0 - 1``).

    Returns:
        A new BGR image with the mask blended on top.
    """
    out = frame_bgr.copy()
    binary = mask > _MASK_BINARY_THRESHOLD
    if not binary.any():
        return out
    color_array = np.array(color_bgr, dtype=np.float32)
    out_float = out.astype(np.float32)
    out_float[binary] = alpha * color_array + (1.0 - alpha) * out_float[binary]
    return np.clip(out_float, 0.0, 255.0).astype(np.uint8)


# %%
def draw_bbox(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a single bounding-box rectangle onto a frame in place.

    Args:
        frame: The BGR image to draw on, modified in place.
        xyxy: The box as ``(x1, y1, x2, y2)`` in pixel coordinates.
        color_bgr: Outline colour as ``(B, G, R)``.
        thickness: Line thickness in pixels.
    """
    x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)


# %%
def put_confidence_text(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    confidence: float,
) -> None:
    """Render the per-box confidence score in white text above the box.

    Args:
        frame: The BGR image to draw on, modified in place.
        xyxy: The corresponding bounding box ``(x1, y1, x2, y2)``.
        confidence: Confidence score in ``0..1``.
    """
    x1, y1, _, _ = (int(round(value)) for value in xyxy)
    text = f"{confidence:.2f}"
    (_, text_h), _ = cv2.getTextSize(text, _HUD_FONT, 0.5, 1)
    text_y = max(y1 - 4, text_h + 2)
    cv2.putText(
        frame, text, (x1, text_y), _HUD_FONT, 0.5, _BGR_WHITE, 1, cv2.LINE_AA
    )


# %%
def render_detect(
    frame: np.ndarray,
    result: Any,
    color_bgr: tuple[int, int, int],
) -> tuple[np.ndarray, int]:
    """Draw detection bounding boxes plus per-box confidence text.

    Args:
        frame: The base BGR frame.
        result: The Ultralytics ``Results`` object for this frame.
        color_bgr: Box outline colour as ``(B, G, R)``.

    Returns:
        ``(annotated_frame, detection_count)``.
    """
    annotated = frame.copy()
    boxes = getattr(result, "boxes", None)
    count = 0
    if boxes is not None and len(boxes) > 0:
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for xyxy, conf_value in zip(xyxys, confs):
            draw_bbox(annotated, xyxy, color_bgr, _BBOX_THICKNESS)
            put_confidence_text(annotated, xyxy, float(conf_value))
            count += 1
    return annotated, count


# %%
def render_polygon_segment(
    frame: np.ndarray,
    result: Any,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> tuple[np.ndarray, int]:
    """Render YOLO11n-seg masks via polygon vertices for sharp edges.

    Args:
        frame: The base BGR frame.
        result: The Ultralytics ``Results`` object for this frame.
        color_bgr: Mask and box colour as ``(B, G, R)``.
        alpha: Mask blend weight (``0 - 1``).

    Returns:
        ``(annotated_frame, detection_count)``.
    """
    annotated = frame.copy()
    height, width = frame.shape[:2]
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)

    polygons = getattr(masks, "xy", None) if masks is not None else None
    if polygons:
        mask_layer = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            if polygon is None or len(polygon) < 3:
                continue
            pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask_layer, [pts], 255)
        annotated = overlay_mask(
            annotated, mask_layer.astype(np.float32) / 255.0, color_bgr, alpha
        )

    count = 0
    if boxes is not None and len(boxes) > 0:
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for xyxy, conf_value in zip(xyxys, confs):
            draw_bbox(annotated, xyxy, color_bgr, _BBOX_THICKNESS)
            put_confidence_text(annotated, xyxy, float(conf_value))
            count += 1
    elif polygons:
        count = len(polygons)

    return annotated, count


# %%
def render_bitmap_segment(
    frame: np.ndarray,
    result: Any,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> tuple[np.ndarray, int]:
    """Render FastSAM masks via the bitmap mask tensor.

    Args:
        frame: The base BGR frame.
        result: The Ultralytics ``Results`` object for this frame.
        color_bgr: Mask and box colour as ``(B, G, R)``.
        alpha: Mask blend weight (``0 - 1``).

    Returns:
        ``(annotated_frame, detection_count)``.
    """
    annotated = frame.copy()
    height, width = frame.shape[:2]
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)

    if masks is not None and getattr(masks, "data", None) is not None and len(masks.data) > 0:
        mask_array = masks.data.cpu().numpy()
        combined = np.zeros((height, width), dtype=np.float32)
        for plane in mask_array:
            resized = cv2.resize(
                plane.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            combined = np.maximum(combined, resized.astype(np.float32))
        annotated = overlay_mask(annotated, combined, color_bgr, alpha)

    count = 0
    if boxes is not None and len(boxes) > 0:
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for xyxy, conf_value in zip(xyxys, confs):
            draw_bbox(annotated, xyxy, color_bgr, _BBOX_THICKNESS)
            put_confidence_text(annotated, xyxy, float(conf_value))
            count += 1
    elif masks is not None and getattr(masks, "data", None) is not None:
        count = int(len(masks.data))

    return annotated, count


# %%
def extract_max_confidence(result: Any) -> float:
    """Return the highest detection confidence in a result, or ``0`` if empty.

    Args:
        result: The Ultralytics ``Results`` object.

    Returns:
        Max confidence over all detections.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        return float(np.max(confs))
    return 0.0


# %%
def apply_overlays(
    frame: np.ndarray,
    result: Any,
    spec: dict[str, Any],
) -> tuple[np.ndarray, int, float]:
    """Dispatch to the right renderer for a model spec.

    Args:
        frame: The base BGR frame.
        result: The Ultralytics ``Results`` object.
        spec: One of the entries in :data:`_MODEL_SPECS`.

    Returns:
        ``(annotated_frame, detection_count, max_confidence)``.
    """
    color_bgr = hex_to_bgr(spec["color_hex"])
    render_mode = spec["render"]
    if render_mode == "detect":
        annotated, count = render_detect(frame, result, color_bgr)
    elif render_mode == "polygon":
        annotated, count = render_polygon_segment(frame, result, color_bgr, _MASK_ALPHA)
    else:
        annotated, count = render_bitmap_segment(frame, result, color_bgr, _MASK_ALPHA)
    return annotated, count, extract_max_confidence(result)


# %%
def letterbox(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Scale-fit a frame into ``target_width`` x ``target_height``, padded black.

    The result is always exactly ``target_width`` x ``target_height``.

    Args:
        frame: BGR source frame.
        target_width: Canvas width in pixels.
        target_height: Canvas height in pixels.

    Returns:
        A BGR frame at exactly the target dimensions.
    """
    height, width = frame.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


# %%
def draw_hud_panel(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int], int]],
    position: tuple[int, int] = (10, 10),
) -> None:
    """Draw a dark semi-transparent HUD panel of text lines onto a frame.

    Args:
        frame: The BGR frame to draw on, modified in place.
        lines: HUD text lines as ``(text, color_bgr, thickness)`` tuples.
        position: Top-left corner of the panel in pixels.
    """
    if not lines:
        return

    measured: list[tuple[str, tuple[int, int, int], int, float, int]] = []
    max_text_w = 0
    for text, color, thickness in lines:
        scale = 0.62 if thickness > 1 else 0.5
        (text_w, _text_h), _baseline = cv2.getTextSize(
            text, _HUD_FONT, scale, thickness
        )
        max_text_w = max(max_text_w, text_w)
        measured.append((text, color, thickness, scale, text_w))

    panel_w = max_text_w + 2 * _HUD_PADDING
    panel_h = len(measured) * _HUD_LINE_HEIGHT + 2 * _HUD_PADDING
    x0, y0 = position

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), thickness=-1
    )
    cv2.addWeighted(
        overlay, _HUD_PANEL_ALPHA, frame, 1.0 - _HUD_PANEL_ALPHA, 0, frame
    )

    for index, (text, color, thickness, scale, _) in enumerate(measured):
        baseline_y = y0 + _HUD_PADDING + index * _HUD_LINE_HEIGHT + 16
        cv2.putText(
            frame,
            text,
            (x0 + _HUD_PADDING, baseline_y),
            _HUD_FONT,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


# %%
def draw_progress_bar(
    frame: np.ndarray,
    current_frame: int,
    total_frames: int,
    color_bgr: tuple[int, int, int],
) -> None:
    """Draw the bottom-edge playback progress bar in place.

    Args:
        frame: The BGR frame to draw on, modified in place.
        current_frame: Index of the most recent frame.
        total_frames: Total frames in the source video.
        color_bgr: Fill colour of the progress portion.
    """
    height, width = frame.shape[:2]
    top = height - _PROGRESS_BAR_HEIGHT
    cv2.rectangle(frame, (0, top), (width, height), _BGR_PROGRESS_BG, thickness=-1)
    if total_frames <= 0:
        return
    fraction = max(0.0, min(1.0, current_frame / total_frames))
    filled = int(round(width * fraction))
    if filled > 0:
        cv2.rectangle(frame, (0, top), (filled, height), color_bgr, thickness=-1)


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------
# %%
@dataclass
class ModelBackend:
    """Unified wrapper around a loaded model regardless of the backend used.

    A single ``ModelBackend`` ties together one model size (e.g. ``yolo11s-seg``)
    with one inference backend (PyTorch / ONNX / TensorRT) and one numeric
    precision (FP32 / FP16). :meth:`predict` hides the per-backend differences
    in calling convention, output format and timing strategy.

    Attributes:
        name: The model key, e.g. ``"yolo11s-seg"``.
        size: Size category (``"nano"`` / ``"small"`` / ``"medium"``).
        backend: Backend identifier (``"pytorch"`` / ``"onnx"`` / ``"tensorrt"``).
        precision: Precision identifier (``"fp32"`` / ``"fp16"``).
        device: Compute device (``"cuda"`` / ``"cpu"``).
        model: The underlying loaded model object (Ultralytics ``YOLO`` for
            PyTorch/TensorRT, ``onnxruntime.InferenceSession`` for ONNX).
        warmup_done: Set to ``True`` once :meth:`warmup` has run at least once.
        spec: Optional reference to the matching entry in :data:`_MODEL_SPECS`,
            used by rendering helpers.
        imgsz: Inference image size baked into the wrapper.
        active_provider: The actual execution provider Ultralytics /
            ``onnxruntime`` ended up using for ONNX wrappers (e.g.
            ``"CUDAExecutionProvider"``). ``"N/A"`` for non-ONNX backends.
    """

    name: str
    size: str
    backend: str
    precision: str
    device: str
    model: Any
    warmup_done: bool = False
    spec: dict[str, Any] | None = None
    imgsz: int = 640
    active_provider: str = "N/A"

    # %%
    def predict(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        iou: float = _IOU_DEFAULT,
    ) -> tuple[Any, float]:
        """Run inference on a single BGR frame.

        The timing strategy follows the backend:

        * PyTorch CUDA / TensorRT: ``torch.cuda.synchronize()`` before and
          after to capture true GPU wall time. For PyTorch FP16 this is
          mandatory - without explicit sync, ``time.perf_counter`` returns
          before the kernel finishes and reports inflated FPS.
        * ONNX (via Ultralytics): ``time.perf_counter`` only. ONNX Runtime
          manages its own synchronisation under the hood so an explicit CUDA
          sync isn't required.
        * PyTorch CPU: ``time.perf_counter`` only.

        Args:
            frame: Source BGR frame as a ``HxWx3`` ``uint8`` ndarray.
            conf: Confidence threshold passed to the underlying model.
            iou: NMS IoU threshold passed to the underlying model.

        Returns:
            ``(result, inference_time_ms)`` where ``result`` is the
            Ultralytics ``Results`` object for the frame.
        """
        on_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        if self.backend in (_BACKEND_PYTORCH, _BACKEND_TENSORRT):
            kwargs: dict[str, Any] = {
                "conf": conf,
                "iou": iou,
                "device": self.device,
                "verbose": False,
                "imgsz": self.imgsz,
            }
            # FP16 weights demand an FP16 input on CUDA; Ultralytics honours
            # ``half=True`` for that conversion internally.
            if self.precision == _PRECISION_FP16 and on_cuda:
                kwargs["half"] = True
            # PyTorch FP16 needs an explicit pair of ``cuda.synchronize`` to
            # capture true wall time - ``model.half()`` alone doesn't engage
            # the Tensor Cores at batch=1, and skipping the sync makes the
            # latency look artificially good. FP32 + TensorRT also benefit
            # from sync-bracketed timing so we apply it to every CUDA path.
            if on_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = self.model.predict(frame, **kwargs)[0]
            if on_cuda:
                torch.cuda.synchronize()
            return result, (time.perf_counter() - t0) * 1000.0

        if self.backend == _BACKEND_ONNX:
            # Ultralytics-native ONNX path: identical call signature to the
            # PyTorch branch, no custom Python post-processing. ORT manages
            # its own CUDA synchronisation so ``perf_counter`` alone is the
            # correct timing primitive here. ``device=self.device`` is
            # passed so a runtime CPU-mode toggle actually flips ORT onto
            # the CPU provider instead of silently staying on CUDA.
            kwargs = {
                "conf": conf,
                "iou": iou,
                "device": self.device,
                "verbose": False,
                "imgsz": self.imgsz,
            }
            t0 = time.perf_counter()
            result = self.model(frame, **kwargs)[0]
            inference_ms = (time.perf_counter() - t0) * 1000.0
            return result, inference_ms

        raise ValueError(f"Unknown backend: {self.backend}")

    # %%
    def warmup(self, iterations: int) -> None:
        """Run ``iterations`` throwaway inferences to remove cold-start cost.

        Args:
            iterations: Number of dummy forward passes.
        """
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(max(0, iterations)):
            self.predict(dummy)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.warmup_done = True

    # %%
    def predict_kernel_only(
        self,
        frame: np.ndarray,
    ) -> tuple[Any, float]:
        """Time only the raw forward pass, excluding Ultralytics' pre/post.

        Frame preprocessing is done once *before* the timed call so the
        returned latency corresponds to just the engine forward. This is
        the number most published benchmarks quote.

        Args:
            frame: Source BGR frame.

        Returns:
            ``(result_or_none, kernel_time_ms)``.
        """
        on_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        half = (
            self.backend == _BACKEND_PYTORCH
            and self.precision == _PRECISION_FP16
            and on_cuda
        )
        tensor = _preprocess_to_tensor(frame, self.imgsz, self.device, half=half)
        if self.backend == _BACKEND_PYTORCH:
            inner = getattr(self.model, "model", self.model)
            if on_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output = inner(tensor)
            if on_cuda:
                torch.cuda.synchronize()
            return output, (time.perf_counter() - t0) * 1000.0
        if self.backend == _BACKEND_TENSORRT:
            if on_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = self.model(tensor, verbose=False)
            if on_cuda:
                torch.cuda.synchronize()
            return output, (time.perf_counter() - t0) * 1000.0
        if self.backend == _BACKEND_ONNX:
            # Ultralytics-wrapped ONNX: fall back to its full predict()
            # (we don't have a raw session here). The benchmark runner
            # uses BenchmarkOnnxBackend.predict_kernel_only instead so
            # this branch is only hit for live wrappers.
            return self.predict(frame)
        raise ValueError(f"Unknown backend: {self.backend}")


# %%
class _OnnxResultBox:
    """Lightweight stand-in for Ultralytics ``Results.boxes`` (ONNX path)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        """Bind the parsed bounding-box arrays.

        Args:
            xyxy: ``(N, 4)`` array of box corners.
            conf: ``(N,)`` array of confidence scores.
        """
        self.xyxy: Any = _OnnxTensor(xyxy.astype(np.float32))
        self.conf: Any = _OnnxTensor(conf.astype(np.float32))

    def __len__(self) -> int:
        return int(self.xyxy.raw.shape[0])


# %%
class _OnnxResultMasks:
    """Lightweight stand-in for Ultralytics ``Results.masks`` (ONNX path)."""

    def __init__(self, xy_polys: list[np.ndarray], data: np.ndarray | None) -> None:
        """Bind the parsed polygon list and optional bitmap stack.

        Args:
            xy_polys: List of ``(K, 2)`` polygon arrays in image coordinates.
            data: Optional ``(N, H, W)`` mask stack as a numpy array.
        """
        self.xy: list[np.ndarray] = xy_polys
        self.data: Any = _OnnxTensor(data) if data is not None else None


# %%
class _OnnxTensor:
    """Thin wrapper exposing ``.cpu().numpy()`` over a numpy array.

    The rendering helpers in this file call ``.cpu().numpy()`` on Ultralytics
    tensors. To stay compatible without forking the renderers, the ONNX path
    wraps its outputs in this wrapper.
    """

    def __init__(self, array: np.ndarray) -> None:
        """Bind a numpy array.

        Args:
            array: The underlying ``ndarray``.
        """
        self.raw: np.ndarray = array

    def cpu(self) -> "_OnnxTensor":
        """Return ``self`` - already on host memory."""
        return self

    def numpy(self) -> np.ndarray:
        """Return the wrapped numpy array."""
        return self.raw

    def __len__(self) -> int:
        return int(self.raw.shape[0]) if self.raw is not None else 0


# %%
class _OnnxResult:
    """Ultralytics-shaped result object built from raw ONNX outputs."""

    def __init__(
        self,
        boxes: _OnnxResultBox | None,
        masks: _OnnxResultMasks | None,
    ) -> None:
        """Bind the parsed boxes and masks.

        Args:
            boxes: The boxes wrapper, or ``None`` if no detections.
            masks: The masks wrapper, or ``None`` if no masks.
        """
        self.boxes: _OnnxResultBox | None = boxes
        self.masks: _OnnxResultMasks | None = masks


# %%
def _onnx_letterbox(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """Letterbox a BGR frame into a square ``imgsz`` canvas for ONNX input.

    Args:
        frame: BGR source frame.
        imgsz: Target square side length in pixels.

    Returns:
        ``(canvas, scale, x_offset, y_offset)`` where the canvas is a
        ``imgsz x imgsz x 3`` BGR ``uint8`` image and the scale/offsets let
        the caller un-letterbox detections back to the original frame.
    """
    h, w = frame.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    x_offset = (imgsz - new_w) // 2
    y_offset = (imgsz - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas, scale, x_offset, y_offset


# %%
def _onnx_predict(
    session: Any,
    frame: np.ndarray,
    imgsz: int,
    conf_threshold: float,
    iou_threshold: float,
) -> _OnnxResult:
    """**Unused** - kept for reference / fallback. Ultralytics now handles ONNX.

    The :class:`ModelBackend` predict path routes ONNX through ``YOLO(...)``
    (Ultralytics' native loader) which is materially faster than this pure
    Python post-processor at batch=1. This helper is retained because other
    parts of the codebase reference its return type, but no live code calls
    it any longer.

    Run one ONNX inference and assemble an Ultralytics-shaped result.

    Supports YOLO11 segmentation ONNX exports: outputs ``(1, 4+1+nm, N)``
    detections plus ``(1, nm, mh, mw)`` mask prototypes. Detections are
    NMS-filtered and mapped back to original frame coordinates.

    Args:
        session: An ``onnxruntime.InferenceSession``.
        frame: Source BGR frame.
        imgsz: Inference image size baked into the ONNX model.
        conf_threshold: Confidence threshold for filtering.
        iou_threshold: IoU threshold for NMS.

    Returns:
        An :class:`_OnnxResult` mirroring the Ultralytics ``Results`` shape.
    """
    canvas, scale, x_off, y_off = _onnx_letterbox(frame, imgsz)
    blob = canvas[..., ::-1].astype(np.float32) / 255.0      # BGR -> RGB, normalise
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1))[None]  # HWC -> NCHW

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    detection_tensor = outputs[0]
    proto_tensor = outputs[1] if len(outputs) > 1 else None

    if detection_tensor.ndim == 3:
        det = detection_tensor[0]
    else:
        det = detection_tensor

    # Ultralytics YOLO11 segmentation layout (channel-first):
    #   rows = 4 (cx,cy,w,h) + nc (class scores) + nm (mask coeffs)
    # Class count ``nc`` varies per export (single-crack -> 1; multi-class ->
    # more), so deriving nm as ``rows - 5`` was wrong whenever nc != 1 and
    # caused matmul shape errors against the prototype tensor. The mask
    # coefficients are ALWAYS the trailing ``nm`` channels, where ``nm`` is
    # the prototype tensor's channel count.
    rows, count = det.shape
    if rows < 5:
        return _OnnxResult(boxes=None, masks=None)

    proto_for_shape = (
        proto_tensor[0] if (proto_tensor is not None and proto_tensor.ndim == 4)
        else proto_tensor
    )
    nm = int(proto_for_shape.shape[0]) if proto_for_shape is not None else 0
    boxes_cxcywh = det[0:4, :].T
    # ``confidence`` is the max class score per anchor; for single-class
    # segmentation models this is just det[4, :], for multi-class it's the
    # per-anchor max across class channels.
    class_rows = max(rows - 4 - nm, 1)
    if class_rows == 1:
        confidence = det[4, :]
    else:
        confidence = det[4 : 4 + class_rows, :].max(axis=0)
    if nm > 0:
        # Always grab the last ``nm`` rows so we do not depend on the class
        # count being exactly 1.
        mask_coeffs = det[-nm:, :].T
    else:
        mask_coeffs = np.zeros((count, 0), dtype=np.float32)

    keep = confidence >= conf_threshold
    if not bool(np.any(keep)):
        return _OnnxResult(boxes=None, masks=None)

    boxes_cxcywh = boxes_cxcywh[keep]
    confidence = confidence[keep]
    mask_coeffs = mask_coeffs[keep]

    # cxcywh -> xyxy in letterboxed pixel space.
    xy = boxes_cxcywh[:, :2]
    wh = boxes_cxcywh[:, 2:]
    xyxy_lb = np.concatenate([xy - wh / 2.0, xy + wh / 2.0], axis=1)

    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=xyxy_lb.tolist(),
        scores=confidence.tolist(),
        score_threshold=float(conf_threshold),
        nms_threshold=float(iou_threshold),
    )
    if len(keep_indices) == 0:
        return _OnnxResult(boxes=None, masks=None)
    if isinstance(keep_indices, np.ndarray):
        keep_idx = keep_indices.flatten().tolist()
    else:
        keep_idx = [int(idx) for idx in keep_indices]

    xyxy_lb = xyxy_lb[keep_idx]
    confidence = confidence[keep_idx]
    mask_coeffs = mask_coeffs[keep_idx]

    # Un-letterbox to original frame coordinates.
    xyxy_orig = xyxy_lb.copy()
    xyxy_orig[:, [0, 2]] -= x_off
    xyxy_orig[:, [1, 3]] -= y_off
    xyxy_orig /= max(scale, 1e-9)

    boxes_obj = _OnnxResultBox(xyxy_orig, confidence)

    masks_obj: _OnnxResultMasks | None = None
    if nm > 0 and proto_tensor is not None:
        polygons, mask_stack = _onnx_decode_masks(
            mask_coeffs,
            proto_tensor[0] if proto_tensor.ndim == 4 else proto_tensor,
            xyxy_lb,
            (frame.shape[0], frame.shape[1]),
            (imgsz, imgsz),
            scale,
            x_off,
            y_off,
        )
        masks_obj = _OnnxResultMasks(xy_polys=polygons, data=mask_stack)

    return _OnnxResult(boxes=boxes_obj, masks=masks_obj)


# %%
def _onnx_decode_masks(
    mask_coeffs: np.ndarray,
    proto: np.ndarray,
    xyxy_lb: np.ndarray,
    image_shape_hw: tuple[int, int],
    canvas_shape_hw: tuple[int, int],
    scale: float,
    x_off: int,
    y_off: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """**Unused** - kept for reference. See :func:`_onnx_predict`.

    Reconstruct binary masks + outline polygons from YOLO seg coefficients.

    Args:
        mask_coeffs: ``(N, nm)`` per-instance mask coefficients.
        proto: ``(nm, mh, mw)`` mask prototype tensor.
        xyxy_lb: ``(N, 4)`` boxes in letterbox space, used for cropping.
        image_shape_hw: ``(H, W)`` of the original frame.
        canvas_shape_hw: ``(H, W)`` of the letterboxed canvas.
        scale: Letterbox scale factor.
        x_off: Horizontal letterbox offset.
        y_off: Vertical letterbox offset.

    Returns:
        ``(polygons, mask_stack)`` where polygons is a list of ``(K, 2)``
        contour arrays in original-image pixel space and ``mask_stack`` is a
        ``(N, H, W)`` ``uint8`` bitmap stack at original-image resolution.
    """
    nm, mh, mw = proto.shape
    # Guard: catches future schema drift (e.g. extra class channels sneaking
    # into the mask-coefficient slice) with a clearer error than the bare
    # numpy matmul mismatch.
    assert mask_coeffs.shape[1] == proto.shape[0], (
        f"mask_coeffs cols {mask_coeffs.shape[1]} != proto rows {proto.shape[0]}"
    )
    # (N, mh*mw) = (N, nm) @ (nm, mh*mw) -> (N, mh, mw)
    masks_low = mask_coeffs @ proto.reshape(nm, -1)
    masks_low = 1.0 / (1.0 + np.exp(-masks_low))   # sigmoid
    masks_low = masks_low.reshape(-1, mh, mw)

    canvas_h, canvas_w = canvas_shape_hw
    orig_h, orig_w = image_shape_hw

    polygons: list[np.ndarray] = []
    output_stack = np.zeros((masks_low.shape[0], orig_h, orig_w), dtype=np.uint8)

    for index, (mask_low, box_lb) in enumerate(zip(masks_low, xyxy_lb)):
        # Upsample to canvas resolution.
        canvas_mask = cv2.resize(
            mask_low.astype(np.float32),
            (canvas_w, canvas_h),
            interpolation=cv2.INTER_LINEAR,
        )
        # Crop with the (letterboxed) bounding box - YOLO standard practice.
        cmask = np.zeros_like(canvas_mask)
        x1, y1, x2, y2 = (int(round(value)) for value in box_lb)
        x1 = max(0, min(x1, canvas_w))
        x2 = max(0, min(x2, canvas_w))
        y1 = max(0, min(y1, canvas_h))
        y2 = max(0, min(y2, canvas_h))
        if x2 > x1 and y2 > y1:
            cmask[y1:y2, x1:x2] = canvas_mask[y1:y2, x1:x2]
        # Remove letterbox padding then resize to original resolution.
        unpadded = cmask[
            y_off : canvas_h - y_off if (canvas_h - 2 * y_off) > 0 else canvas_h,
            x_off : canvas_w - x_off if (canvas_w - 2 * x_off) > 0 else canvas_w,
        ]
        if unpadded.size == 0:
            output_stack[index] = 0
            continue
        full = cv2.resize(unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = (full > _MASK_BINARY_THRESHOLD).astype(np.uint8) * 255
        output_stack[index] = binary

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            polygon = largest.reshape(-1, 2).astype(np.float32)
            polygons.append(polygon)

    return polygons, output_stack


# %%
def _preprocess_to_tensor(
    frame: np.ndarray,
    imgsz: int,
    device: str,
    half: bool = False,
) -> "torch.Tensor":
    """Letterbox + normalise ``frame`` to a model-ready GPU tensor.

    Args:
        frame: Source BGR frame as a ``HxWx3`` ``uint8`` ndarray.
        imgsz: Target square side length the model expects.
        device: ``"cuda"`` / ``"cpu"`` - where to put the returned tensor.
        half: When ``True`` cast the tensor to ``float16`` (used by the
            PyTorch kernel-only timing path when measuring FP16).

    Returns:
        A ``1x3xHxW`` float tensor on ``device``, normalised to ``[0, 1]``.
    """
    canvas, _scale, _x_off, _y_off = _onnx_letterbox(frame, imgsz)
    blob = canvas[..., ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1))[None]
    tensor = torch.from_numpy(blob)
    if device.startswith("cuda") and torch.cuda.is_available():
        tensor = tensor.to(device, non_blocking=True)
    if half:
        tensor = tensor.half()
    return tensor


# %%
def _preprocess_to_numpy(frame: np.ndarray, imgsz: int) -> np.ndarray:
    """Letterbox + normalise to a ``1x3xHxW`` ``float32`` numpy blob.

    Used by :class:`BenchmarkOnnxBackend` to feed ``onnxruntime``'s
    ``OrtValue.ortvalue_from_numpy`` without an extra torch round-trip.

    Args:
        frame: Source BGR frame.
        imgsz: Target square side length the ONNX model expects.

    Returns:
        ``np.ndarray`` shaped ``(1, 3, imgsz, imgsz)``, ``float32``.
    """
    canvas, _scale, _x_off, _y_off = _onnx_letterbox(frame, imgsz)
    blob = canvas[..., ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob.transpose(2, 0, 1))[None]


# %%
@dataclass
class BenchmarkOnnxBackend(ModelBackend):
    """Benchmark-only ONNX wrapper using raw ``onnxruntime`` + IO binding.

    Constructed by :func:`_load_benchmark_onnx_backend`. ``predict`` performs
    a real letterbox + normalise + ``session.run_with_iobinding`` call so the
    end-to-end timing reflects ORT's true cost without the
    ``tensor.cpu().numpy()`` round-trip Ultralytics' AutoBackend introduces.

    Attributes (in addition to :class:`ModelBackend`):
        session: The pinned-CUDA ``ort.InferenceSession`` instance.
        input_name: Input tensor name (typically ``"images"``).
        output_names: Output tensor names discovered from the session.
        io_binding_available: ``True`` when ``session.io_binding()`` is
            usable on this ORT install; ``False`` triggers the
            ``session.run`` fallback (with a one-shot warning).
    """

    session: Any = None
    input_name: str = "images"
    output_names: tuple[str, ...] = ()
    io_binding_available: bool = True

    # %%
    def _run_engine(self, blob: np.ndarray) -> int:
        """Run one ORT inference on ``blob`` and return a sanity detection count.

        Args:
            blob: ``(1, 3, H, W)`` ``float32`` preprocessed input.

        Returns:
            Number of anchor proposals in the first output tensor (used as
            a sanity check in the live log).
        """
        on_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if self.io_binding_available:
            try:
                io_binding = self.session.io_binding()
                ort_input = ort.OrtValue.ortvalue_from_numpy(
                    blob, "cuda" if on_cuda else "cpu", 0,
                )
                io_binding.bind_ortvalue_input(self.input_name, ort_input)
                for name in self.output_names:
                    io_binding.bind_output(name, "cuda" if on_cuda else "cpu")
                self.session.run_with_iobinding(io_binding)
                if on_cuda:
                    torch.cuda.synchronize()
                return 0
            except Exception as exc:  # noqa: BLE001 - keep run alive
                print(
                    f"[onnx-warning] {self.name} IO binding failed: {exc} "
                    "(falling back to session.run)"
                )
                self.io_binding_available = False
        outputs = self.session.run(
            list(self.output_names), {self.input_name: blob},
        )
        if on_cuda:
            torch.cuda.synchronize()
        detections = 0
        if outputs:
            first = outputs[0]
            if isinstance(first, np.ndarray) and first.ndim == 3:
                detections = int(first.shape[2])
        return detections

    # %%
    def predict(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        iou: float = _IOU_DEFAULT,
    ) -> tuple[Any, float]:
        """Time only the ORT engine forward (preprocess outside the window).

        For raw ONNX we measure just ``session.run`` / ``run_with_iobinding``
        - this matches the engine-time numbers most published benchmarks
        quote. Preprocess (letterbox + normalise) runs *before* the timed
        window. Postprocess is skipped entirely in benchmark mode.

        Args:
            frame: Source BGR frame.
            conf: Confidence threshold (ignored; kept for interface parity).
            iou: NMS IoU threshold (ignored; kept for interface parity).

        Returns:
            ``(detections_count, inference_time_ms)``.
        """
        del conf, iou
        blob = _preprocess_to_numpy(frame, self.imgsz)
        t0 = time.perf_counter()
        detections = self._run_engine(blob)
        t1 = time.perf_counter()
        return detections, (t1 - t0) * 1000.0

    # %%
    def predict_kernel_only(self, frame: np.ndarray) -> tuple[Any, float]:
        """Engine-only timing (identical to :meth:`predict` for raw ONNX).

        Args:
            frame: Source BGR frame.

        Returns:
            ``(detections_count, kernel_time_ms)``.
        """
        return self.predict(frame)


# %%
def _load_benchmark_onnx_backend(
    spec: dict[str, Any],
    weights_path: Path,
    cpu_mode: bool,
) -> BenchmarkOnnxBackend:
    """Build a raw-ORT ONNX wrapper pinned to CUDA EP for benchmark mode.

    Args:
        spec: The model spec being loaded.
        weights_path: Path to the ``.onnx`` file.
        cpu_mode: When ``True`` skip CUDA EP and use CPU EP only.

    Returns:
        A :class:`BenchmarkOnnxBackend` ready for timed calls.
    """
    if ort is None:
        raise RuntimeError("onnxruntime is not installed in this environment.")
    cuda_available = (
        not cpu_mode
        and torch.cuda.is_available()
        and "CUDAExecutionProvider" in ort.get_available_providers()
    )
    providers: list[Any]
    if cuda_available:
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(weights_path), sess_options=sess_opts, providers=providers,
    )
    active_providers = session.get_providers()
    active = active_providers[0] if active_providers else "unknown"
    print(f"[onnx-provider] {spec['key']}: {active}")
    if cuda_available and active != "CUDAExecutionProvider":
        print(
            f"[onnx-warning] {spec['key']} fell back to CPU despite explicit "
            "CUDA EP request - VRAM exhausted"
        )
    # Infer imgsz from the model's input shape.
    imgsz = 640
    try:
        shape = session.get_inputs()[0].shape
        if len(shape) >= 4 and isinstance(shape[-1], int):
            imgsz = int(shape[-1])
    except Exception:  # noqa: BLE001 - keep default on any parse error
        imgsz = 640
    input_name = session.get_inputs()[0].name
    output_names = tuple(output.name for output in session.get_outputs())
    # IO binding availability sanity-check (some old ORT builds lack it).
    io_binding_available = True
    try:
        _ = session.io_binding()
    except Exception:  # noqa: BLE001 - graceful fallback path
        io_binding_available = False
        print(
            f"[onnx-warning] {spec['key']} ORT build lacks io_binding(); "
            "falling back to session.run for benchmarking."
        )
    return BenchmarkOnnxBackend(
        name=spec["key"],
        size=spec["size"],
        backend=_BACKEND_ONNX,
        precision=_PRECISION_FP32,
        device="cuda" if cuda_available else "cpu",
        model=session,
        spec=spec,
        imgsz=imgsz,
        active_provider=active,
        session=session,
        input_name=input_name,
        output_names=output_names,
        io_binding_available=io_binding_available,
    )


# %%
@dataclass
class BenchmarkTensorRTBackend(ModelBackend):
    """Benchmark-only TensorRT wrapper using the raw runtime API.

    Bypasses Ultralytics' AutoBackend entirely so the recorded latency is
    just the engine's ``execute_async_v3`` (TRT 10) or ``execute_async_v2``
    (TRT 8/9) call, bracketed by ``cuda.synchronize``. Input/output GPU
    tensors are pre-allocated at load time so per-frame allocation cost
    doesn't pollute the measurement.

    Attributes (in addition to :class:`ModelBackend`):
        engine: The deserialised ``trt.ICudaEngine``.
        context: The execution context built from ``engine``.
        input_name: Name of the input tensor (typically ``"images"``).
        output_names: Tuple of output tensor names (``"output0"`` / ``"output1"``).
        input_tensor: Pre-allocated GPU input tensor.
        output_tensors: ``{name: torch.Tensor}`` map of output buffers.
        input_torch_dtype: Whether to feed FP16 or FP32 input.
        api_v3_available: ``True`` if ``execute_async_v3`` is callable on
            this TensorRT install.
    """

    engine: Any = None
    context: Any = None
    input_name: str = "images"
    output_names: tuple[str, ...] = ()
    input_tensor: Any = None
    output_tensors: dict[str, Any] = field(default_factory=dict)
    input_torch_dtype: Any = None
    api_v3_available: bool = True

    # %%
    def _stage_input(self, frame: np.ndarray) -> None:
        """Preprocess ``frame`` and copy it into the pre-allocated GPU buffer.

        Args:
            frame: Source BGR frame.
        """
        tensor = _preprocess_to_tensor(frame, self.imgsz, self.device)
        if self.input_torch_dtype is torch.float16:
            tensor = tensor.half()
        elif self.input_torch_dtype is torch.float32:
            tensor = tensor.float()
        self.input_tensor.copy_(tensor)

    # %%
    def _execute(self) -> int:
        """Launch one engine forward on the current CUDA stream and sync.

        Returns:
            Sanity detection count read off the first output tensor.
        """
        stream_handle = torch.cuda.current_stream().cuda_stream
        if self.api_v3_available:
            ok = self.context.execute_async_v3(stream_handle=stream_handle)
        else:
            bindings = [int(self.input_tensor.data_ptr())]
            for name in self.output_names:
                bindings.append(int(self.output_tensors[name].data_ptr()))
            ok = self.context.execute_async_v2(
                bindings=bindings, stream_handle=stream_handle,
            )
        torch.cuda.synchronize()
        if not ok:
            print(f"[trt-warning] {self.name} execute returned False")
        detections = 0
        first_out = (
            self.output_tensors[self.output_names[0]]
            if self.output_names else None
        )
        if first_out is not None and first_out.ndim >= 3:
            detections = int(first_out.shape[-1])
        return detections

    # %%
    def predict(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        iou: float = _IOU_DEFAULT,
    ) -> tuple[Any, float]:
        """Time only the engine execute (preprocess outside the window).

        For raw TensorRT we measure just ``execute_async_v3`` between
        ``cuda.synchronize`` brackets - this is the engine-time number
        most published benchmarks quote. Preprocess (letterbox + dtype
        cast + GPU staging) runs *before* the timed window. Postprocess
        is skipped entirely in benchmark mode.

        Args:
            frame: Source BGR frame.
            conf: Confidence threshold (ignored; bookkeeping only).
            iou: NMS IoU threshold (ignored; bookkeeping only).

        Returns:
            ``(detections, inference_time_ms)``.
        """
        del conf, iou
        self._stage_input(frame)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        detections = self._execute()
        t1 = time.perf_counter()
        return detections, (t1 - t0) * 1000.0

    # %%
    def predict_kernel_only(self, frame: np.ndarray) -> tuple[Any, float]:
        """Engine-only timing (identical to :meth:`predict` for raw TRT).

        Args:
            frame: Source BGR frame.

        Returns:
            ``(detections, kernel_time_ms)``.
        """
        return self.predict(frame)


# %%
def _load_benchmark_tensorrt_backend(
    spec: dict[str, Any],
    weights_path: Path,
) -> BenchmarkTensorRTBackend:
    """Build a raw-TensorRT wrapper with pre-allocated GPU bindings.

    Args:
        spec: The model spec being loaded.
        weights_path: Path to the ``.engine`` file.

    Returns:
        A :class:`BenchmarkTensorRTBackend` ready for timed calls.

    Raises:
        RuntimeError: If TensorRT or CUDA are unavailable, or the engine
            file fails to deserialize.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "tensorrt is not installed in this environment."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT engines require a CUDA-capable GPU.")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    # Ultralytics exports ``.engine`` files with a metadata prefix:
    #   [4 bytes meta_len LE][meta_len bytes of JSON][actual engine bytes]
    # The raw TRT runtime errors with ``magicTag != rt::kPLAN_MAGIC_TAG`` if
    # we deserialize the whole file. Detect the prefix and skip past it
    # before calling ``deserialize_cuda_engine``; fall back to a pure-engine
    # load when the metadata is missing or corrupt.
    raw_bytes = weights_path.read_bytes()
    prefix_stripped: bytes | None = None
    try:
        meta_len = int.from_bytes(raw_bytes[:4], byteorder="little")
        if 0 < meta_len < 1_000_000 and 4 + meta_len < len(raw_bytes):
            try:
                json.loads(raw_bytes[4 : 4 + meta_len].decode("utf-8"))
                prefix_stripped = raw_bytes[4 + meta_len :]
            except (UnicodeDecodeError, json.JSONDecodeError):
                prefix_stripped = None
    except Exception:  # noqa: BLE001 - any parse error -> pure-engine fallback
        prefix_stripped = None

    # Prefer the prefix-stripped buffer (Ultralytics-style export); if that
    # fails to deserialize, retry with the whole file in case the metadata
    # heuristic was a false positive.
    engine = None
    if prefix_stripped is not None:
        engine = runtime.deserialize_cuda_engine(prefix_stripped)
    if engine is None:
        engine = runtime.deserialize_cuda_engine(raw_bytes)
    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize TensorRT engine: {weights_path}"
        )
    context = engine.create_execution_context()

    # Discover IO tensors. TRT 10 uses ``num_io_tensors`` + ``get_tensor_*``;
    # TRT 8/9 uses ``num_bindings`` + ``get_binding_*``. Try the new API
    # first, fall back to the old.
    input_name: str | None = None
    output_names_list: list[str] = []
    input_dtype: Any = trt.DataType.FLOAT
    input_shape: tuple[int, ...] | None = None
    output_shapes: dict[str, tuple[int, ...]] = {}
    output_dtypes: dict[str, Any] = {}

    if hasattr(engine, "num_io_tensors"):
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_name = name
                input_dtype = engine.get_tensor_dtype(name)
                input_shape = tuple(engine.get_tensor_shape(name))
            else:
                output_names_list.append(name)
                output_dtypes[name] = engine.get_tensor_dtype(name)
                output_shapes[name] = tuple(engine.get_tensor_shape(name))
        api_v3_available = True
    else:
        for index in range(engine.num_bindings):
            name = engine.get_binding_name(index)
            shape = tuple(engine.get_binding_shape(index))
            dtype = engine.get_binding_dtype(index)
            if engine.binding_is_input(index):
                input_name = name
                input_dtype = dtype
                input_shape = shape
            else:
                output_names_list.append(name)
                output_dtypes[name] = dtype
                output_shapes[name] = shape
        api_v3_available = False

    if input_name is None or input_shape is None:
        raise RuntimeError(
            f"TensorRT engine for {spec['key']} exposed no input tensor."
        )

    # Resolve the input torch dtype from the engine's expectation.
    if input_dtype == trt.DataType.HALF:
        torch_input_dtype = torch.float16
        precision_label = _PRECISION_FP16
    elif input_dtype == trt.DataType.INT8:
        # Engines exported with INT8 quantisation still typically take FP32
        # input and quantise internally; keep the input FP32 here.
        torch_input_dtype = torch.float32
        precision_label = _PRECISION_INT8
    else:
        torch_input_dtype = torch.float32
        precision_label = _PRECISION_FP32

    imgsz = 640
    if len(input_shape) >= 4 and input_shape[-1] > 0:
        imgsz = int(input_shape[-1])

    # Replace dynamic dims (-1 / 0) with concrete sizes so we can allocate
    # the buffers up-front.
    def materialise_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Substitute non-positive dims with batch=1, channels=3, side=imgsz."""
        out: list[int] = []
        for index, dim in enumerate(shape):
            if dim > 0:
                out.append(int(dim))
            elif index == 0:
                out.append(1)
            elif index == 1:
                out.append(3)
            else:
                out.append(imgsz)
        return tuple(out)

    materialised_input_shape = materialise_shape(input_shape)
    input_tensor = torch.zeros(
        materialised_input_shape, dtype=torch_input_dtype, device="cuda",
    )
    output_tensors: dict[str, Any] = {}
    for name in output_names_list:
        shape = materialise_shape(output_shapes[name])
        dtype = output_dtypes[name]
        if dtype == trt.DataType.HALF:
            torch_out_dtype = torch.float16
        elif dtype == trt.DataType.INT32:
            torch_out_dtype = torch.int32
        else:
            torch_out_dtype = torch.float32
        output_tensors[name] = torch.zeros(
            shape, dtype=torch_out_dtype, device="cuda",
        )

    if api_v3_available:
        # Bind each tensor to its pre-allocated GPU buffer now so subsequent
        # ``execute_async_v3`` calls don't have to rebind.
        try:
            context.set_input_shape(input_name, materialised_input_shape)
        except Exception:  # noqa: BLE001 - dynamic-shape calls may not exist
            pass
        context.set_tensor_address(input_name, int(input_tensor.data_ptr()))
        for name in output_names_list:
            context.set_tensor_address(
                name, int(output_tensors[name].data_ptr()),
            )

    print(
        f"[trt-engine] {spec['key']}: api={'v3' if api_v3_available else 'v2'} "
        f"input_dtype={input_dtype} imgsz={imgsz}"
    )

    return BenchmarkTensorRTBackend(
        name=spec["key"],
        size=spec["size"],
        backend=_BACKEND_TENSORRT,
        precision=precision_label,
        device="cuda",
        model=engine,
        spec=spec,
        imgsz=imgsz,
        active_provider=f"TensorRT-{('v3' if api_v3_available else 'v2')}",
        engine=engine,
        context=context,
        input_name=input_name,
        output_names=tuple(output_names_list),
        input_tensor=input_tensor,
        output_tensors=output_tensors,
        input_torch_dtype=torch_input_dtype,
        api_v3_available=api_v3_available,
    )


# %%
def load_backend(
    spec: dict[str, Any],
    backend: str,
    weights_path: Path,
    cpu_mode: bool,
    *,
    benchmark_mode: bool = False,
) -> ModelBackend:
    """Load one ``(model size, backend)`` combination from disk.

    Args:
        spec: One of the entries in :data:`_MODEL_SPECS`.
        backend: Backend identifier (``"pytorch"`` / ``"onnx"`` / ``"tensorrt"``
            / ``"tensorrt_int8"``).
        weights_path: Path to the backend's weight file.
        cpu_mode: If ``True``, force CPU even when CUDA is available.

    Returns:
        A fully-initialised :class:`ModelBackend`.

    Raises:
        FileNotFoundError: If ``weights_path`` does not exist.
        RuntimeError: If the requested backend is unavailable on this system.
    """
    # An INT8 engine loads exactly like an FP16 engine; only the file on disk
    # and the precision it reports differ.
    engine_precision = _PRECISION_FP16
    if backend == _BACKEND_TENSORRT_INT8:
        backend = _BACKEND_TENSORRT
        engine_precision = _PRECISION_INT8

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"{_BACKEND_DISPLAY[backend]} weights for {spec['display']} not found "
            f"at {weights_path}."
        )

    device = "cuda" if (not cpu_mode and torch.cuda.is_available()) else "cpu"

    if backend == _BACKEND_PYTORCH:
        model = YOLO(str(weights_path))
        model.to(device)
        # Native precision starts at FP32; FP16 is requested per-predict via
        # ``half=True`` so the loaded weights can be reused either way.
        return ModelBackend(
            name=spec["key"],
            size=spec["size"],
            backend=backend,
            precision=_PRECISION_FP32,
            device=device,
            model=model,
            spec=spec,
        )

    if backend == _BACKEND_ONNX:
        if not _ONNXRUNTIME_AVAILABLE or ort is None:
            raise RuntimeError("onnxruntime is not installed in this environment.")
        if benchmark_mode:
            # Bypass Ultralytics' AutoBackend entirely for benchmarking:
            # pinned CUDA EP with a fixed VRAM cap so large sessions can't
            # silently fall back to CPU under memory pressure.
            return _load_benchmark_onnx_backend(spec, weights_path, cpu_mode)
        # Live path: keep using Ultralytics' wrapper so the run-screen
        # gets a fully-decoded Results object back. The active provider
        # is discovered lazily once Ultralytics constructs its internal
        # session (logged when ``_log_live_onnx_provider`` is called).
        model = YOLO(str(weights_path))
        device = "cpu" if cpu_mode or not torch.cuda.is_available() else "cuda"
        print(
            f"[onnx-provider] {spec['key']}: (live path - "
            f"resolved on first predict call, device={device})"
        )
        return ModelBackend(
            name=spec["key"],
            size=spec["size"],
            backend=backend,
            precision=_PRECISION_FP32,
            device=device,
            model=model,
            spec=spec,
            active_provider="UltralyticsAutoBackend",
        )

    if backend == _BACKEND_TENSORRT:
        if cpu_mode:
            raise RuntimeError("TensorRT engines require CUDA.")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT engines require a CUDA-capable GPU.")
        if benchmark_mode:
            # Bypass Ultralytics for fair engine-vs-engine timing.
            return _load_benchmark_tensorrt_backend(spec, weights_path)
        # Live path keeps Ultralytics so the run-screen still draws Results.
        model = YOLO(str(weights_path))
        return ModelBackend(
            name=spec["key"],
            size=spec["size"],
            backend=backend,
            precision=engine_precision,
            device="cuda",
            model=model,
            spec=spec,
        )

    raise ValueError(f"Unknown backend: {backend}")


# %%
def backend_availability(
    selected_model_keys: list[str],
    backend_paths: dict[str, dict[str, Path]],
    cpu_mode: bool,
) -> dict[str, str]:
    """Compute per-backend availability dot colours for the menu UI.

    Args:
        selected_model_keys: Currently selected model size keys.
        backend_paths: Resolved ``{backend: {key: path}}`` mapping.
        cpu_mode: Whether CPU-only mode is currently active.

    Returns:
        A ``{backend: dot_color_hex}`` mapping for each backend.
    """
    status: dict[str, str] = {}
    for backend in _BACKENDS:
        if backend == _BACKEND_TENSORRT and cpu_mode:
            status[backend] = _DOT_RED
            continue
        if backend == _BACKEND_ONNX and not _ONNXRUNTIME_AVAILABLE:
            status[backend] = _DOT_RED
            continue
        paths = backend_paths.get(backend, {})
        if not selected_model_keys:
            status[backend] = _DOT_GREY
            continue
        present = sum(1 for key in selected_model_keys if paths.get(key) and paths[key].is_file())
        if present == len(selected_model_keys):
            status[backend] = _DOT_GREEN
        elif present == 0:
            status[backend] = _DOT_RED
        else:
            status[backend] = _DOT_YELLOW
    return status


# %%
def precision_availability(
    selected_model_keys: list[str],
    backend_paths: dict[str, dict[str, Path]],
    backend: str,
    cpu_mode: bool,
) -> dict[str, str]:
    """Compute per-precision availability dot colours for the menu UI.

    Only INT8 has a distinct file on disk - FP32 and FP16 are reached from the
    same weights as the backend itself, so they simply mirror the backend's own
    availability once the ``(backend, precision)`` pair is supported.

    Args:
        selected_model_keys: Currently selected model size keys.
        backend_paths: Resolved ``{backend: {key: path}}`` mapping.
        backend: The backend currently selected in the UI.
        cpu_mode: Whether CPU-only mode is currently active.

    Returns:
        A ``{precision: dot_color_hex}`` mapping for each precision.
    """
    backend_status = backend_availability(selected_model_keys, backend_paths, cpu_mode)
    status: dict[str, str] = {}
    for precision in _PRECISIONS:
        if (backend, precision) not in _SUPPORTED_BACKEND_PRECISIONS:
            status[precision] = _DOT_GREY
            continue
        if precision != _PRECISION_INT8:
            status[precision] = backend_status.get(backend, _DOT_GREY)
            continue
        if cpu_mode:
            status[precision] = _DOT_RED
            continue
        paths = backend_paths.get(_BACKEND_TENSORRT_INT8, {})
        if not selected_model_keys:
            status[precision] = _DOT_GREY
            continue
        present = sum(
            1 for key in selected_model_keys if paths.get(key) and paths[key].is_file()
        )
        if present == len(selected_model_keys):
            status[precision] = _DOT_GREEN
        elif present == 0:
            status[precision] = _DOT_RED
        else:
            status[precision] = _DOT_YELLOW
    return status


# %%
class Tooltip:
    """A hover tooltip for a single tk/customtkinter widget.

    Used to explain why a control is greyed out - the INT8 precision radio is
    disabled whenever a non-TensorRT backend is selected, and a disabled radio
    cannot otherwise say why. Setting :attr:`text` to an empty string silences
    the tooltip without unbinding it.

    Attributes:
        text: The message shown on hover. Empty disables the tooltip.
    """

    def __init__(self, widget: Any, text: str = "") -> None:
        """Bind the tooltip to ``widget``.

        Args:
            widget: The widget to attach hover handlers to.
            text: Initial tooltip text.
        """
        self.text: str = text
        self._widget: Any = widget
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, event: Any) -> None:
        """Show the tooltip near the cursor.

        Args:
            event: The tk enter event carrying the pointer position.
        """
        if not self.text or self._window is not None:
            return
        window = tk.Toplevel(self._widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{event.x_root + px(12)}+{event.y_root + px(16)}")
        tk.Label(
            window,
            text=self.text,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            font=FONT_SMALL,
            padx=px(8),
            pady=px(4),
            borderwidth=1,
            relief=tk.SOLID,
        ).pack()
        self._window = window

    def _on_leave(self, event: Any) -> None:
        """Destroy the tooltip window.

        Args:
            event: The tk leave event (unused).
        """
        if self._window is not None:
            self._window.destroy()
            self._window = None


# ---------------------------------------------------------------------------
# Benchmark runner (headless inference timing)
# ---------------------------------------------------------------------------
# %%
@dataclass
class BenchmarkResult:
    """One row of the benchmark results table.

    Attributes:
        model_key: Model size key, e.g. ``"yolo11s-seg"``.
        size_label: Human label (``"Nano"`` / ``"Small"`` / ``"Medium"``).
        backend: Backend identifier.
        precision: Precision identifier.
        device: Compute device.
        mean_fps: Pipeline mean throughput including pre/post-processing.
        p50_ms: 50th-percentile pipeline latency.
        p95_ms: 95th-percentile pipeline latency.
        p99_ms: 99th-percentile pipeline latency.
        min_ms: Minimum pipeline latency.
        max_ms: Maximum pipeline latency.
        cold_start_ms: First-frame pipeline latency.
        throughput_fps: ``frame_count / total_seconds`` independent of mean.
        frames: Number of timed pipeline frames.
        timestamp: ISO timestamp when this row was produced.
        status: ``"ok"`` for completed runs, otherwise an explanatory string.
        kernel_mean_fps: Raw-forward mean throughput (excludes pre/post).
        kernel_p50_ms: 50th-percentile kernel-only latency.
        kernel_p95_ms: 95th-percentile kernel-only latency.
        kernel_frames: Number of frames timed in the kernel-only pass.
        active_provider: Execution provider actually used for ONNX combos.
            ``"N/A"`` for PyTorch / TensorRT.
    """

    model_key: str
    size_label: str
    backend: str
    precision: str
    device: str
    mean_fps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    cold_start_ms: float
    throughput_fps: float
    frames: int
    timestamp: str
    status: str = "ok"
    kernel_mean_fps: float = 0.0
    kernel_p50_ms: float = 0.0
    kernel_p95_ms: float = 0.0
    kernel_frames: int = 0
    active_provider: str = "N/A"


# %%
def benchmark_one_combo(
    spec: dict[str, Any],
    backend: str,
    precision: str,
    weights_path: Path,
    video_paths: list[Path],
    frame_count: int,
    warmup_frames: int,
    cpu_mode: bool,
    log_fn: Any,
    cancel_event: threading.Event,
) -> BenchmarkResult | None:
    """Time one ``(model, backend, precision)`` combination over real frames.

    Args:
        spec: The model spec being benchmarked.
        backend: Backend identifier.
        precision: Precision identifier.
        weights_path: Path to the weights for this combo.
        video_paths: Videos to read frames from (cycled to fill the budget).
        frame_count: Total timed frames after warmup.
        warmup_frames: Pre-timing warmup frames that are discarded.
        cpu_mode: Whether to force CPU even when CUDA is present.
        log_fn: Callable accepting one log line (queued onto the UI thread).
        cancel_event: Set by the UI when the user clicks Cancel.

    Returns:
        A populated :class:`BenchmarkResult`, or ``None`` if cancelled.
    """
    label = (
        f"{spec['display']} / {_BACKEND_DISPLAY[backend]} / "
        f"{precision.upper()}"
    )

    if not weights_path.is_file():
        log_fn(f"[skip] {label}: file missing ({weights_path.name})")
        return None

    log_fn(f"[load] {label}")
    try:
        wrapper = load_backend(
            spec, backend, weights_path, cpu_mode, benchmark_mode=True,
        )
        if backend == _BACKEND_PYTORCH and precision == _PRECISION_FP16:
            wrapper.precision = _PRECISION_FP16
        elif backend == _BACKEND_TENSORRT:
            # The engine file already fixes the precision, so report what the
            # combo asked for - otherwise INT8 rows are mislabelled as FP16.
            wrapper.precision = precision
        elif backend == _BACKEND_ONNX:
            wrapper.precision = _PRECISION_FP32
        else:
            wrapper.precision = precision
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going
        log_fn(f"[fail] {label}: {exc}")
        return None

    # Collect a flat list of BGR frames cycled across the input videos.
    log_fn(f"[frames] sampling {frame_count + warmup_frames} frames")
    frames = _sample_frames(video_paths, frame_count + warmup_frames)
    if not frames:
        log_fn(f"[fail] {label}: no frames could be read from inputs/")
        return None

    log_fn(f"[warmup] {warmup_frames} frames")
    for i in range(min(warmup_frames, len(frames))):
        if cancel_event.is_set():
            return None
        wrapper.predict(frames[i % len(frames)])

    log_fn(f"[timed] {frame_count} frames (pipeline)")
    latencies: list[float] = []
    total_t0 = time.perf_counter()
    cold_start_ms: float = 0.0
    for i in range(frame_count):
        if cancel_event.is_set():
            return None
        frame = frames[(warmup_frames + i) % len(frames)]
        _result, inference_ms = wrapper.predict(frame)
        latencies.append(inference_ms)
        if i == 0:
            cold_start_ms = inference_ms
        if (i + 1) % 50 == 0:
            log_fn(f"  ...{i + 1}/{frame_count} frames done")
    total_seconds = time.perf_counter() - total_t0

    arr = np.asarray(latencies, dtype=np.float64)
    mean_ms = float(arr.mean()) if arr.size else 0.0
    mean_fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0
    throughput = (frame_count / total_seconds) if total_seconds > 0 else 0.0

    # ---- Second pass: kernel-only timing -----------------------------
    kernel_frames_target = min(frame_count, _KERNEL_TIMING_FRAMES)
    kernel_latencies: list[float] = []
    log_fn(f"[timed] {kernel_frames_target} frames (kernel-only)")
    # Re-warm with the kernel-only path so the first call's overhead
    # doesn't pollute the kernel mean.
    for i in range(min(5, len(frames))):
        if cancel_event.is_set():
            break
        try:
            wrapper.predict_kernel_only(frames[i % len(frames)])
        except Exception as exc:  # noqa: BLE001 - log and abort kernel pass
            log_fn(f"[kernel-warn] {label}: kernel-only warmup failed: {exc}")
            kernel_frames_target = 0
            break
    for i in range(kernel_frames_target):
        if cancel_event.is_set():
            return None
        frame = frames[(warmup_frames + i) % len(frames)]
        try:
            _kresult, kernel_ms = wrapper.predict_kernel_only(frame)
        except Exception as exc:  # noqa: BLE001 - never fail the row
            log_fn(f"[kernel-warn] {label}: kernel-only timing aborted: {exc}")
            kernel_latencies = []
            break
        kernel_latencies.append(kernel_ms)

    karr = np.asarray(kernel_latencies, dtype=np.float64)
    kernel_mean_ms = float(karr.mean()) if karr.size else 0.0
    kernel_mean_fps = (1000.0 / kernel_mean_ms) if kernel_mean_ms > 0 else 0.0
    kernel_p50_ms = float(np.percentile(karr, 50)) if karr.size else 0.0
    kernel_p95_ms = float(np.percentile(karr, 95)) if karr.size else 0.0

    result = BenchmarkResult(
        model_key=spec["key"],
        size_label=spec["size_label"],
        backend=backend,
        precision=precision,
        device=wrapper.device,
        mean_fps=mean_fps,
        p50_ms=float(np.percentile(arr, 50)) if arr.size else 0.0,
        p95_ms=float(np.percentile(arr, 95)) if arr.size else 0.0,
        p99_ms=float(np.percentile(arr, 99)) if arr.size else 0.0,
        min_ms=float(arr.min()) if arr.size else 0.0,
        max_ms=float(arr.max()) if arr.size else 0.0,
        cold_start_ms=cold_start_ms,
        throughput_fps=throughput,
        frames=int(arr.size),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        kernel_mean_fps=kernel_mean_fps,
        kernel_p50_ms=kernel_p50_ms,
        kernel_p95_ms=kernel_p95_ms,
        kernel_frames=int(karr.size),
        active_provider=getattr(wrapper, "active_provider", "N/A"),
    )
    log_fn(
        f"[done] {label}: pipeline {mean_fps:.1f} FPS "
        f"({result.p50_ms:.1f}/{result.p95_ms:.1f}/{result.p99_ms:.1f} ms p50/p95/p99) "
        f"| kernel {kernel_mean_fps:.1f} FPS"
        f" | provider={result.active_provider}"
    )
    return result


# %%
def _sample_frames(video_paths: list[Path], wanted: int) -> list[np.ndarray]:
    """Read up to ``wanted`` BGR frames across the given videos in order.

    Args:
        video_paths: Source videos to read from.
        wanted: Maximum number of frames to return.

    Returns:
        A list of BGR ``ndarray`` frames (length may be less than ``wanted``
        if the videos run out before the budget is hit).
    """
    frames: list[np.ndarray] = []
    for path in video_paths:
        if len(frames) >= wanted:
            break
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            continue
        while len(frames) < wanted:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
    return frames


# %%
def append_benchmark_csv(
    csv_path: Path,
    results: list[BenchmarkResult],
) -> None:
    """Append benchmark rows to a CSV, writing the header if the file is new.

    Args:
        csv_path: Destination CSV path.
        results: Rows to append.
    """
    if not results:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "model",
        "size",
        "backend",
        "precision",
        "device",
        "mean_fps",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "cold_start_ms",
        "throughput_fps",
        "frames",
    ]
    file_exists = csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(header)
        for row in results:
            writer.writerow(
                [
                    row.timestamp,
                    row.model_key,
                    row.size_label,
                    _BACKEND_DISPLAY[row.backend],
                    row.precision.upper(),
                    row.device,
                    f"{row.mean_fps:.2f}",
                    f"{row.p50_ms:.2f}",
                    f"{row.p95_ms:.2f}",
                    f"{row.p99_ms:.2f}",
                    f"{row.min_ms:.2f}",
                    f"{row.max_ms:.2f}",
                    f"{row.cold_start_ms:.2f}",
                    f"{row.throughput_fps:.2f}",
                    row.frames,
                ]
            )


# ---------------------------------------------------------------------------
# Inference worker (split-screen mode)
# ---------------------------------------------------------------------------
# %%
class ModelWorker(threading.Thread):
    """Persistent per-model inference thread with its own CUDA stream.

    The worker pulls frames off ``input_queue`` (maxsize=1), runs inference on
    its own ``torch.cuda.Stream`` so the three models can overlap, and pushes
    ``(result, inference_ms)`` onto ``output_queue``. Both queues are bounded
    to 1 so the producer naturally back-pressures: dropping new frames when a
    model is still busy with the previous one.
    """

    def __init__(
        self,
        spec: dict[str, Any],
        backend: ModelBackend,
        app_stop_event: threading.Event,
    ) -> None:
        """Create a worker bound to one ``ModelBackend``.

        Args:
            spec: The model spec dict from :data:`_MODEL_SPECS`.
            backend: The already-loaded :class:`ModelBackend` instance the
                worker should drive. May be replaced at any time via
                :meth:`set_backend` to switch inference backends.
            app_stop_event: Set on app shutdown to make the worker exit.
        """
        super().__init__(daemon=True, name=f"worker-{spec['key']}")
        self.spec: dict[str, Any] = spec
        self.backend: ModelBackend = backend
        self.device: str = backend.device
        self.app_stop_event: threading.Event = app_stop_event
        self.input_queue: queue.Queue = queue.Queue(maxsize=1)
        self.output_queue: queue.Queue = queue.Queue(maxsize=1)
        self.stream: Any = (
            torch.cuda.Stream()
            if backend.device.startswith("cuda")
            and torch.cuda.is_available()
            and backend.backend == _BACKEND_PYTORCH
            else None
        )
        self._conf: float = float(spec["default_conf"])
        self._conf_lock: threading.Lock = threading.Lock()
        self._backend_lock: threading.Lock = threading.Lock()

    # %%
    def set_backend(self, backend: ModelBackend) -> None:
        """Atomically swap the underlying :class:`ModelBackend`.

        Args:
            backend: The new backend to use for subsequent inferences.
        """
        with self._backend_lock:
            self.backend = backend
            self.device = backend.device
            self.stream = (
                torch.cuda.Stream()
                if backend.device.startswith("cuda")
                and torch.cuda.is_available()
                and backend.backend == _BACKEND_PYTORCH
                else None
            )

    # %%
    def set_conf(self, conf: float) -> None:
        """Thread-safely update the confidence threshold for future inferences.

        Args:
            conf: New confidence threshold in ``0..1``.
        """
        with self._conf_lock:
            self._conf = float(conf)

    # %%
    def submit(self, frame: np.ndarray) -> bool:
        """Submit a frame for inference, dropping it if the worker is busy.

        Args:
            frame: BGR frame to enqueue.

        Returns:
            ``True`` if accepted, ``False`` if dropped (back-pressure).
        """
        try:
            self.input_queue.put_nowait(frame)
            return True
        except queue.Full:
            return False

    # %%
    def collect(self, timeout: float) -> tuple[Any, float] | None:
        """Block on the worker's output queue with a timeout.

        Args:
            timeout: Maximum seconds to wait for a result.

        Returns:
            ``(result, inference_ms)`` or ``None`` on timeout.
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # %%
    def drain_output(self) -> None:
        """Discard any stale result still sitting in the output queue."""
        try:
            self.output_queue.get_nowait()
        except queue.Empty:
            pass

    # %%
    def run(self) -> None:
        """Worker main loop - runs until ``app_stop_event`` is set."""
        while not self.app_stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=_WORKER_POLL_TIMEOUT)
            except queue.Empty:
                continue

            with self._conf_lock:
                conf = self._conf
            with self._backend_lock:
                backend = self.backend
                stream = self.stream

            try:
                if stream is not None:
                    # Per-worker CUDA stream lets the three split-screen
                    # PyTorch workers overlap on the same GPU. The ``predict``
                    # method does its own internal cuda.synchronize() before
                    # and after timing, so timings stay accurate.
                    with torch.cuda.stream(stream):
                        result, inference_ms = backend.predict(frame, conf=conf)
                else:
                    result, inference_ms = backend.predict(frame, conf=conf)
                try:
                    self.output_queue.put_nowait((result, inference_ms))
                except queue.Full:
                    # Display is behind; drop this result rather than block.
                    pass
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                print(f"[worker {self.spec['key']}] error during predict: {exc}")
                continue


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------
# %%
class HistoryStore:
    """Tiny JSON-backed history of completed runs."""

    def __init__(self, path: Path) -> None:
        """Bind the store to a JSON file (created lazily).

        Args:
            path: Path to the history JSON file.
        """
        self.path: Path = path
        self._lock: threading.Lock = threading.Lock()

    # %%
    def load(self) -> list[dict[str, Any]]:
        """Read every history entry from disk.

        Returns:
            A list of run-entry dicts (possibly empty).
        """
        if not self.path.is_file():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    # %%
    def save_all(self, entries: list[dict[str, Any]]) -> None:
        """Atomically rewrite the history file with the given entries.

        Args:
            entries: Full list of entries to persist.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(entries, handle, indent=2)

    # %%
    def add(self, entry: dict[str, Any]) -> None:
        """Insert a new entry at the top of the history.

        Args:
            entry: The run-entry dict to persist.
        """
        entries = self.load()
        entries.insert(0, entry)
        self.save_all(entries)

    # %%
    def delete(self, run_id: str) -> None:
        """Remove a run from history by its ``run_id``.

        Args:
            run_id: UUID of the run to remove.
        """
        entries = [entry for entry in self.load() if entry.get("run_id") != run_id]
        self.save_all(entries)


# %%
class BenchmarkHistoryStore:
    """JSON-backed store of completed benchmark runs.

    Schema for every entry::

        {
          "run_id":      "<uuid4>",
          "timestamp":   "<ISO 8601>",
          "yolo_version":"yolo11" | "yolo26" | ...,
          "combos": [
              {"model_key", "size", "backend", "precision", "device",
               "mean_fps", "p50_ms", "p95_ms", "p99_ms",
               "cold_start_ms", "frame_count"},
              ...,
          ],
          "video_files": ["...mp4", ...],
          "notes": "",
        }
    """

    def __init__(self, path: Path) -> None:
        """Bind the store to a JSON file (created lazily).

        Args:
            path: Path to the benchmark-history JSON file.
        """
        self.path: Path = path
        self._lock: threading.Lock = threading.Lock()

    # %%
    def load(self) -> list[dict[str, Any]]:
        """Read every benchmark-history entry from disk.

        Returns:
            A list of run-entry dicts. Empty list if the file is missing or
            malformed - never raises.
        """
        if not self.path.is_file():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    # %%
    def save_all(self, entries: list[dict[str, Any]]) -> None:
        """Atomically rewrite the benchmark-history file.

        Args:
            entries: Full list of entries to persist.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(entries, handle, indent=2)

    # %%
    def add(self, entry: dict[str, Any]) -> None:
        """Insert a new entry at the top of the benchmark history.

        Args:
            entry: The benchmark-run dict to persist.
        """
        entries = self.load()
        entries.insert(0, entry)
        self.save_all(entries)

    # %%
    def delete(self, run_id: str) -> None:
        """Remove a benchmark run from history by its ``run_id``.

        Args:
            run_id: UUID of the run to remove.
        """
        entries = [entry for entry in self.load() if entry.get("run_id") != run_id]
        self.save_all(entries)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------
# %%
def _snapshot_confidence_key(item: tuple[Path, float]) -> float:
    """Sort key returning the confidence of a ``(path, conf)`` pair.

    Args:
        item: A ``(snapshot_path, max_confidence)`` tuple.

    Returns:
        The confidence value, for use as a ``sorted`` key.
    """
    return item[1]


# %%
def gather_snapshot_confidences(snapshot_paths: list[str]) -> list[tuple[Path, float]]:
    """Pair each snapshot path with the max confidence from its JSON sidecar.

    Args:
        snapshot_paths: Paths to snapshot JPEGs.

    Returns:
        ``[(path, max_confidence), ...]`` sorted by descending confidence.
    """
    annotated: list[tuple[Path, float]] = []
    for raw in snapshot_paths:
        path = Path(raw)
        if not path.is_file():
            continue
        sidecar = path.with_suffix(".json")
        max_conf = 0.0
        if sidecar.is_file():
            try:
                with sidecar.open("r", encoding="utf-8") as handle:
                    sidecar_data = json.load(handle)
                max_conf = float(sidecar_data.get("max_confidence", 0.0))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                max_conf = 0.0
        annotated.append((path, max_conf))
    annotated.sort(key=_snapshot_confidence_key, reverse=True)
    return annotated


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------
# %%
def export_run_pdf(entry: dict[str, Any], destination: Path) -> None:
    """Generate a one-page PDF run report at ``destination``.

    Args:
        entry: The history-entry dict for the run.
        destination: PDF file path (parent dirs created if missing).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    story: list[Any] = []

    story.append(Paragraph("GhanaCrack Run Report", styles["Title"]))
    story.append(Spacer(1, 4 * mm))

    metadata_rows: list[list[str]] = [
        ["Run ID", str(entry.get("run_id", ""))],
        ["Timestamp", str(entry.get("timestamp", ""))],
        ["Models", ", ".join(entry.get("model_list", []))],
        ["Videos", ", ".join(entry.get("video_list", []))],
        ["Duration (s)", f"{float(entry.get('duration_seconds', 0.0)):.1f}"],
        ["Snapshots", str(len(entry.get("snapshot_paths", [])))],
    ]
    metadata_table = RLTable(metadata_rows, hAlign="LEFT", colWidths=[35 * mm, 130 * mm])
    metadata_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.grey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (0, -1), rl_colors.whitesmoke),
            ]
        )
    )
    story.append(metadata_table)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Per-Model Statistics", styles["Heading2"]))
    stats_rows: list[list[str]] = [
        ["Model", "Frames", "Mean FPS", "p50 ms", "p95 ms", "Detections", "Detect %"]
    ]
    for name, stats in entry.get("per_model_stats", {}).items():
        stats_rows.append(
            [
                name,
                str(int(stats.get("frames", 0))),
                f"{float(stats.get('mean_fps', 0.0)):.1f}",
                f"{float(stats.get('p50_latency_ms', 0.0)):.1f}",
                f"{float(stats.get('p95_latency_ms', 0.0)):.1f}",
                str(int(stats.get("total_detections", 0))),
                f"{100.0 * float(stats.get('detection_rate', 0.0)):.1f}",
            ]
        )
    stats_table = RLTable(stats_rows, hAlign="LEFT")
    stats_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(stats_table)
    story.append(Spacer(1, 6 * mm))

    snapshots_sorted = gather_snapshot_confidences(entry.get("snapshot_paths", []))
    top = snapshots_sorted[:5]
    if top:
        story.append(Paragraph("Top Snapshots", styles["Heading2"]))
        grid_rows: list[list[Any]] = []
        current_row: list[Any] = []
        for path, _confidence in top:
            try:
                image = RLImage(str(path), width=55 * mm, height=40 * mm)
                current_row.append(image)
            except Exception:  # noqa: BLE001 - skip a broken snapshot
                current_row.append("")
            if len(current_row) == 3:
                grid_rows.append(current_row)
                current_row = []
        if current_row:
            while len(current_row) < 3:
                current_row.append("")
            grid_rows.append(current_row)
        grid = RLTable(grid_rows, hAlign="LEFT")
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(grid)
        story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            f"Generated {datetime.now().isoformat(timespec='seconds')}",
            styles["Italic"],
        )
    )
    doc.build(story)


# ---------------------------------------------------------------------------
# The application class
# ---------------------------------------------------------------------------
# %%
class App:
    """The GhanaCrack desktop application root."""

    def __init__(self, root: tk.Tk) -> None:
        """Wire up the root window and kick off background model loading.

        Args:
            root: The toplevel ``tk.Tk`` window.
        """
        self.root: tk.Tk = root
        # Align tkinter's point-to-pixel mapping with the physical display.
        self.root.tk.call("tk", "scaling", DPI_SCALE * 96 / 72)
        self.root.title(_WINDOW_TITLE)
        self.root.configure(bg=_BG_DARK)
        self.root.geometry(f"{px(1400)}x{px(900)}")
        self.root.minsize(px(1000), px(700))

        self.device: str = resolve_device()
        self.history: HistoryStore = HistoryStore(_HISTORY_PATH)
        self.benchmark_history: BenchmarkHistoryStore = BenchmarkHistoryStore(
            _BENCHMARK_HISTORY_PATH,
        )

        # Auto-discovery: scan ``models/`` for ``*_crack_finetuned.{pt,onnx,
        # engine}``. The static ``_MODEL_SPECS`` is still used as a fallback
        # when discovery finds nothing (fresh checkout with no weights yet).
        self.discovered_models: dict[
            str, dict[str, dict[str, Any]]
        ] = discover_models(_MODELS_DIR)
        summary = format_discovery_summary(self.discovered_models)
        print(f"[discovery] found: {summary or '(no models)'}")
        self.model_specs: list[dict[str, Any]] = (
            build_specs_from_discovery(self.discovered_models) or list(_MODEL_SPECS)
        )
        self.model_by_key: dict[str, dict[str, Any]] = {
            spec["key"]: spec for spec in self.model_specs
        }
        self.versions: list[str] = sorted(self.discovered_models.keys()) or sorted(
            {spec.get("version", "yolo11") for spec in self.model_specs}
        )

        # Config loading (benchmark parameters only - backend paths are
        # discovered above).
        self.app_config: dict[str, Any] = load_app_config(_CONFIG_PATH)
        bench_cfg = self.app_config.get("benchmark", {}) or {}
        self.benchmark_frame_count: int = int(bench_cfg.get("frame_count", 300))
        self.benchmark_warmup_frames: int = int(bench_cfg.get("warmup_frames", 30))
        bench_csv_raw = str(bench_cfg.get("output_csv", "results/benchmark_results.csv"))
        bench_csv_path = Path(bench_csv_raw)
        self.benchmark_csv_path: Path = (
            bench_csv_path if bench_csv_path.is_absolute() else _BASE_DIR / bench_csv_path
        )

        # Compatibility shim: existing call sites read ``self.backend_paths``
        # in the ``{backend: {model_key: Path}}`` shape. We populate it once
        # from discovery and rebuild it whenever discovery is refreshed.
        self.backend_paths: dict[str, dict[str, Path]] = (
            self._build_backend_paths_from_discovery()
        )

        # Models + workers (populated by the background loader thread).
        self.app_stop_event: threading.Event = threading.Event()
        self.models: dict[str, dict[str, Any]] = {}
        self.workers: dict[str, ModelWorker] = {}
        self.models_loaded: threading.Event = threading.Event()
        # Loaded ModelBackend wrappers keyed by (model_key, backend).
        self.backends: dict[tuple[str, str], ModelBackend] = {}
        # Tracks which cpu-mode the cached ``self.backends`` is currently
        # configured for. ``False`` at startup because the loader always
        # picks the best-available device (CUDA when present). Flipped by
        # ``_apply_cpu_mode_to_backends`` whenever a run/benchmark needs
        # the opposite mode.
        self._current_cpu_mode: bool = False

        # Shared mutable confidence dict - written by main thread, read by
        # video / worker threads.
        self.live_conf: dict[str, float] = {
            spec["key"]: float(spec["default_conf"]) for spec in self.model_specs
        }

        # Main-menu tk vars.
        self.loading_status: tk.StringVar = tk.StringVar(value="Loading models...")
        self.mode_var: tk.StringVar = tk.StringVar(value="single")
        first_display = (
            self.model_specs[0]["display"] if self.model_specs else "YOLO11n-seg"
        )
        self.single_model_var: tk.StringVar = tk.StringVar(value=first_display)
        self.save_snapshots_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.save_history_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.selected_video_count_var: tk.StringVar = tk.StringVar(
            value="0 video(s) selected"
        )

        # Version selector: default to first discovered version.
        default_version = self.versions[0] if self.versions else "yolo11"
        self.version_var: tk.StringVar = tk.StringVar(value=default_version)

        # Backend / Precision / CPU state. Default: nano-size checked across
        # every discovered version so render code can rely on at least one
        # checked entry being present.
        self.selected_size_vars: dict[str, tk.BooleanVar] = {
            spec["key"]: tk.BooleanVar(
                value=(
                    spec["size"] == "nano"
                    and spec.get("version", default_version) == default_version
                ),
            )
            for spec in self.model_specs
        }
        self.backend_var: tk.StringVar = tk.StringVar(value=_DEFAULT_BACKEND)
        self.precision_var: tk.StringVar = tk.StringVar(value=_DEFAULT_PRECISION)
        self.cpu_mode_var: tk.BooleanVar = tk.BooleanVar(value=False)

        # customtkinter main-menu state. ``menu_view`` is "run" (default) or
        # "settings"; ``_video_checkbox_vars`` holds the per-video BooleanVars
        # that replaced the old tk.Listbox.
        self.menu_view: str = "run"
        self._video_checkbox_vars: dict[str, tk.BooleanVar] = {}

        # Menu widget refs that need to be reachable from callbacks / loader.
        self._menu_widgets: dict[str, Any] = {}

        # Run-time state (set when a run starts, cleared when it ends).
        self.run_state: dict[str, Any] | None = None
        self.video_thread: threading.Thread | None = None
        self._display_queue: queue.Queue = queue.Queue(maxsize=1)
        self._canvas_photo: ImageTk.PhotoImage | None = None
        self._transition_overlay: tk.Frame | None = None

        # Run-summary timer state.
        self._summary_visible: bool = False
        self._summary_countdown_var: tk.StringVar = tk.StringVar(value="")
        self._summary_thumb_refs: list[ImageTk.PhotoImage] = []

        # History detail state.
        self._history_thumb_refs: list[ImageTk.PhotoImage] = []
        self._history_entries_cache: list[dict[str, Any]] = []

        # Screen swapping.
        self.current_frame: tk.Frame | None = None
        self.current_screen: str = "main_menu"

        # Benchmarks-screen state (populated when the screen builds itself);
        # accessed by ``_on_chart_metric_changed`` to redraw the chart.
        self._benchmarks_screen_state: dict[str, Any] | None = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.show_main_menu()
        self._start_model_loading()

    # ------------------------------------------------------------------
    # Lifecycle / model loading
    # ------------------------------------------------------------------
    # %%
    def _build_backend_paths_from_discovery(self) -> dict[str, dict[str, Path]]:
        """Flatten ``self.discovered_models`` to ``{backend: {key: Path}}``.

        Existing call sites (loader, availability, benchmark runner) read
        backend paths in this flat shape. The discovery registry remains the
        source of truth; this method just adapts it.

        Returns:
            A mapping with one entry per ``_BACKENDS`` value plus the
            ``tensorrt_int8`` slot. Missing files simply omit that
            ``(backend, key)`` pair.
        """
        slots = (*_BACKENDS, _BACKEND_TENSORRT_INT8)
        flat: dict[str, dict[str, Path]] = {backend: {} for backend in slots}
        for version_entries in self.discovered_models.values():
            for size_entry in version_entries.values():
                key = size_entry["key"]
                for backend in slots:
                    path = size_entry.get(backend)
                    if isinstance(path, Path) and path.is_file():
                        flat[backend][key] = path
        return flat

    # %%
    def lookup_weights(self, model_key: str, backend: str) -> Path | None:
        """Return the discovered weights path for ``(model_key, backend)``.

        Args:
            model_key: Composite key, e.g. ``"yolo11n-seg"``.
            backend: One of ``_BACKENDS``, or ``"tensorrt_int8"``.

        Returns:
            The on-disk ``Path`` if discovery saw the file, otherwise ``None``.
        """
        spec = self.model_by_key.get(model_key)
        if spec is None:
            return None
        version = spec.get("version")
        size_letter = spec.get("size_letter")
        if not version or not size_letter:
            return None
        size_entry = self.discovered_models.get(version, {}).get(size_letter)
        if size_entry is None:
            return None
        path = size_entry.get(backend)
        return path if isinstance(path, Path) else None

    # %%
    def refresh_discovery(self) -> None:
        """Rescan ``models/`` and rebuild the discovery-derived registries.

        Called on startup and whenever the user clicks Refresh in the Video
        Selection card. Safe to call from the main thread only.
        """
        self.discovered_models = discover_models(_MODELS_DIR)
        summary = format_discovery_summary(self.discovered_models)
        print(f"[discovery] found: {summary or '(no models)'}")
        new_specs = build_specs_from_discovery(self.discovered_models)
        if new_specs:
            self.model_specs = new_specs
            self.model_by_key = {spec["key"]: spec for spec in self.model_specs}
        self.versions = sorted(self.discovered_models.keys()) or self.versions
        self.backend_paths = self._build_backend_paths_from_discovery()
        # Keep tk vars consistent with the new spec set.
        for spec in self.model_specs:
            key = spec["key"]
            if key not in self.selected_size_vars:
                self.selected_size_vars[key] = tk.BooleanVar(value=False)
            if key not in self.live_conf:
                self.live_conf[key] = float(spec["default_conf"])
        # If the previously-active version vanished, snap to the first one.
        if self.version_var.get() not in self.versions and self.versions:
            self.version_var.set(self.versions[0])

    # %%
    def _apply_cpu_mode_to_backends(self, cpu_mode: bool) -> None:
        """Move every cached ``ModelBackend`` to match the requested cpu_mode.

        The startup loader bakes wrappers onto the best available device
        (CUDA when present, otherwise CPU). When the user toggles CPU-only
        the cached wrappers have to actually relocate, otherwise the
        toggle is purely cosmetic.

        Per-backend behaviour:

        * **PyTorch** - ``wrapper.model.to(target)`` swaps the weights and
          ``wrapper.device`` is updated. Ultralytics' YOLO class supports
          ``.to()`` on its embedded ``nn.Module`` directly.
        * **ONNX (Ultralytics-wrapped)** - the ORT session's provider is
          fixed at construction time, so we rebuild the wrapper via
          :func:`load_backend` with the new ``cpu_mode`` flag. The fresh
          wrapper replaces the cached one and is warmed up.
        * **TensorRT** - engines are device-bound; we leave them alone.
          The Backend & Precision card already prevents the user from
          picking TensorRT while CPU-only is on.

        Every affected ``ModelWorker`` is notified via ``set_backend`` so
        its internal CUDA stream is rebuilt (or torn down on CPU).

        Args:
            cpu_mode: ``True`` to force CPU, ``False`` to use CUDA when
                available.
        """
        if self._current_cpu_mode == cpu_mode:
            return
        # If CUDA is unavailable, the wrappers are already on CPU and we
        # only need to flip the bookkeeping flag.
        if not torch.cuda.is_available():
            self._current_cpu_mode = True
            return
        target_device = "cpu" if cpu_mode else "cuda"
        for (model_key, backend), wrapper in list(self.backends.items()):
            if backend in (_BACKEND_TENSORRT, _BACKEND_TENSORRT_INT8):
                # TensorRT engines stay on CUDA; the picker disallows CPU
                # mode for them so they should never be selected anyway.
                continue
            if backend == _BACKEND_PYTORCH:
                try:
                    wrapper.model.to(target_device)
                except Exception as exc:  # noqa: BLE001 - keep run alive
                    print(f"[cpu-mode] {model_key}/PyTorch to({target_device}) failed: {exc}")
                    continue
                wrapper.device = target_device
            elif backend == _BACKEND_ONNX:
                weights_path = self.lookup_weights(model_key, backend)
                if weights_path is None:
                    continue
                spec = self.model_by_key.get(model_key)
                if spec is None:
                    continue
                try:
                    new_wrapper = load_backend(
                        spec, backend, weights_path, cpu_mode,
                    )
                    new_wrapper.warmup(_WARMUP_ITERATIONS)
                except Exception as exc:  # noqa: BLE001 - keep run alive
                    print(f"[cpu-mode] {model_key}/ONNX reload failed: {exc}")
                    continue
                wrapper = new_wrapper
                self.backends[(model_key, backend)] = wrapper
                # Carry the precision flag from the previous wrapper.
                existing = self.backends.get((model_key, backend))
                if existing is not None:
                    wrapper.precision = existing.precision
            # Tell the per-key worker (if any) that the underlying backend
            # is on a different device now.
            worker = self.workers.get(model_key)
            if worker is not None and worker.backend is not None:
                if worker.backend.backend == backend:
                    worker.set_backend(wrapper)
        self._current_cpu_mode = cpu_mode

    # %%
    def _start_model_loading(self) -> None:
        """Spawn the background model-loading thread."""
        loader = threading.Thread(
            target=self._load_models, name="model-loader", daemon=True
        )
        loader.start()

    # %%
    def _load_models(self) -> None:
        """Background thread: load every available ``(size, backend)`` combo.

        For each model size declared in :data:`_MODEL_SPECS` the loader walks
        every backend listed in ``config.yaml`` and tries to load it. Missing
        files are skipped silently (no crash). The first successfully-loaded
        backend per size also serves as the default ``self.models[size]``
        entry used by the existing live-run code paths; the per-size
        :class:`ModelWorker` is seeded with the active backend selection.
        """
        # The startup loader always boots wrappers onto the best-available
        # device (CUDA when present). The CPU-only switch is then honoured
        # at run-start by ``_apply_cpu_mode_to_backends`` which moves the
        # cached wrappers to CPU on the fly (and rebuilds ONNX sessions
        # with the CPU provider).
        cpu_mode = False
        total = len(self.model_specs)
        for index, spec in enumerate(self.model_specs, start=1):
            self._post_status(
                f"Loading {spec['display']}... {index}/{total}"
            )
            primary: ModelBackend | None = None
            primary_used_finetuned = False
            primary_weights_label = ""

            for backend in (*_BACKENDS, _BACKEND_TENSORRT_INT8):
                weights_path = self.lookup_weights(spec["key"], backend)
                if weights_path is None or not weights_path.is_file():
                    continue
                try:
                    wrapper = load_backend(spec, backend, weights_path, cpu_mode)
                    wrapper.warmup(_WARMUP_ITERATIONS)
                except Exception as exc:  # noqa: BLE001 - keep the loader alive
                    print(f"[loader] skipping {spec['key']}/{backend}: {exc}")
                    continue
                self.backends[(spec["key"], backend)] = wrapper
                if primary is None and backend == _BACKEND_PYTORCH:
                    primary = wrapper
                    primary_used_finetuned = True  # config points at the finetuned file
                    primary_weights_label = weights_path.name

            # Fall back to the legacy finetuned/pretrained loader if no
            # config-driven PyTorch entry resolved (e.g. an unedited config).
            if primary is None:
                try:
                    model, weights_label, used_finetuned = load_model_with_fallback(
                        spec, self.device
                    )
                    warmup_model(model, self.device, _WARMUP_ITERATIONS)
                except Exception as exc:  # noqa: BLE001 - surface but continue
                    self._post_status(
                        f"Failed to load {spec['display']}: {exc}"
                    )
                    continue
                primary = ModelBackend(
                    name=spec["key"],
                    size=spec["size"],
                    backend=_BACKEND_PYTORCH,
                    precision=_PRECISION_FP32,
                    device=self.device,
                    model=model,
                    spec=spec,
                )
                primary_used_finetuned = used_finetuned
                primary_weights_label = weights_label
                self.backends[(spec["key"], _BACKEND_PYTORCH)] = primary

            self.models[spec["key"]] = {
                "spec": spec,
                "model": primary.model,
                "weights": primary_weights_label,
                "used_finetuned": primary_used_finetuned,
                "backend_wrapper": primary,
            }
            worker = ModelWorker(spec, primary, self.app_stop_event)
            worker.start()
            self.workers[spec["key"]] = worker
            self.root.after(0, self._apply_loaded_badge, spec["key"])

        self._post_status("All models ready.")
        self.models_loaded.set()
        self.root.after(0, self._on_all_models_loaded)

    # %%
    def _post_status(self, message: str) -> None:
        """Schedule a loading-status update on the tk main thread.

        Args:
            message: New status text.
        """
        self.root.after(0, self._apply_status, message)

    # %%
    def _apply_status(self, message: str) -> None:
        """Apply a loading-status update on the main thread.

        Args:
            message: New status text.
        """
        self.loading_status.set(message)

    # %%
    def _apply_loaded_badge(self, model_key: str) -> None:
        """Update the sidebar status label once a model finishes loading.

        Args:
            model_key: The model key whose sidebar row to update.
        """
        if self.current_screen != "main_menu":
            return
        status_labels = self._menu_widgets.get("sidebar_status", {}) or {}
        label = status_labels.get(model_key)
        if label is None:
            return
        info = self.models.get(model_key)
        if info is None:
            label.configure(text="Failed", text_color=ACCENT_RED)
            return
        if info["used_finetuned"]:
            label.configure(text="Ready", text_color=ACCENT_GREEN)
        else:
            label.configure(text="Pretrained", text_color=ACCENT_YELLOW)

    # %%
    def _on_all_models_loaded(self) -> None:
        """Main-thread callback: enable run buttons, refresh the action bar."""
        if self.current_screen != "main_menu":
            return
        progress = self._menu_widgets.get("loading_progress")
        if progress is not None:
            try:
                progress.stop()
                progress.pack_forget()
            except Exception:  # noqa: BLE001 - widget may have been destroyed
                pass
        loading_label = self._menu_widgets.get("loading_summary")
        if loading_label is not None:
            try:
                loading_label.configure(text="All models ready", text_color=ACCENT_GREEN)
            except Exception:  # noqa: BLE001 - widget may have been destroyed
                pass
        for key in list(self.models.keys()):
            self._apply_loaded_badge(key)
        self._update_start_button_state()
        self._refresh_backend_status()

    # %%
    def _on_close(self) -> None:
        """Tear down threads cleanly when the user clicks the OS close button."""
        self.app_stop_event.set()
        if self.run_state is not None:
            self.run_state["stop_event"].set()
        # Give the video thread a moment to wind down.
        if self.video_thread is not None and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        self.root.destroy()

    # ------------------------------------------------------------------
    # Screen swapping
    # ------------------------------------------------------------------
    # %%
    def _swap_frame(self, new_frame: tk.Frame, screen_name: str) -> None:
        """Replace the current screen frame with ``new_frame``.

        Args:
            new_frame: The replacement frame, not yet packed.
            screen_name: Logical name of the new screen (``main_menu``,
                ``run``, ``history``, ``summary``).
        """
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_screen = screen_name
        new_frame.pack(fill=tk.BOTH, expand=True)
        # Reset menu widget refs when leaving the menu so stale handles don't
        # accidentally fire callbacks.
        if screen_name != "main_menu":
            self._menu_widgets = {}

    # ------------------------------------------------------------------
    # Main menu (customtkinter)
    # ------------------------------------------------------------------
    # %%
    def show_main_menu(self) -> None:
        """Build and display the customtkinter two-panel main menu.

        Sidebar (200px, left) holds branding, nav buttons and live model
        status; the content area (right, fills) hosts a swappable view (Run
        configuration or Settings). Clicking the History nav button performs
        a full-window screen swap into the existing history screen.
        """
        frame = ctk.CTkFrame(self.root, fg_color=BG_SECONDARY, corner_radius=0)

        sidebar = self._build_sidebar(frame)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        content = ctk.CTkFrame(frame, fg_color=BG_SECONDARY, corner_radius=0)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._menu_widgets["content_area"] = content

        self._swap_frame(frame, "main_menu")
        self._show_main_menu_view(self.menu_view)
        if self.models_loaded.is_set():
            self._on_all_models_loaded()

    # %%
    def _build_sidebar(self, parent: ctk.CTkBaseClass) -> ctk.CTkFrame:
        """Construct the persistent 200px sidebar.

        Args:
            parent: The main-menu container frame.

        Returns:
            The sidebar frame, packed by the caller.
        """
        sidebar = ctk.CTkFrame(
            parent, fg_color=BG_PRIMARY, corner_radius=0, width=220,
        )
        sidebar.pack_propagate(False)

        # 1px right border so the sidebar visually separates from content.
        border = ctk.CTkFrame(sidebar, width=1, fg_color=BORDER, corner_radius=0)
        border.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")

        # Branding
        branding = ctk.CTkFrame(sidebar, fg_color="transparent")
        branding.pack(fill=tk.X, padx=16, pady=(20, 4))
        ctk.CTkLabel(
            branding, text="GhanaCrack", text_color=ACCENT_ORANGE,
            font=FONT_TITLE, anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            branding, text="Crack Detection", text_color=TEXT_MUTED,
            font=FONT_SUBTITLE, anchor="w",
        ).pack(anchor="w")

        ctk.CTkFrame(sidebar, height=1, fg_color=BORDER, corner_radius=0).pack(
            fill=tk.X, padx=12, pady=(12, 8),
        )

        # Navigation
        nav_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        nav_frame.pack(fill=tk.X)
        nav_buttons: dict[str, dict[str, Any]] = {}
        nav_specs: list[tuple[str, str, Any]] = [
            ("run", "▶  Run", self._on_nav_run),
            ("history", "🕐  History", self._on_nav_history),
            ("benchmarks", "📊  Benchmarks", self._on_nav_benchmarks),
            ("settings", "⚙  Settings", self._on_nav_settings),
        ]
        for name, label, command in nav_specs:
            row = ctk.CTkFrame(nav_frame, fg_color="transparent", height=44)
            row.pack(fill=tk.X, pady=1)
            row.pack_propagate(False)
            bar = ctk.CTkFrame(row, width=3, fg_color=BG_PRIMARY, corner_radius=0)
            bar.pack(side=tk.LEFT, fill=tk.Y)
            btn = ctk.CTkButton(
                row, text=label, anchor="w",
                fg_color="transparent", hover_color=BG_HOVER,
                text_color=TEXT_MUTED,
                font=FONT_NAV,
                corner_radius=0, height=44,
                command=command,
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
            nav_buttons[name] = {"row": row, "bar": bar, "button": btn}
        self._menu_widgets["nav_buttons"] = nav_buttons

        # Bottom-pinned status block + version.
        bottom = ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
        ctk.CTkFrame(bottom, height=1, fg_color=BORDER, corner_radius=0).pack(
            fill=tk.X, padx=12, pady=(0, 8),
        )

        status_labels: dict[str, ctk.CTkLabel] = {}
        specs_by_version: dict[str, list[dict[str, Any]]] = {}
        for spec in self.model_specs:
            specs_by_version.setdefault(spec.get("version", "yolo11"), []).append(spec)
        for version in sorted(specs_by_version.keys()):
            divider = ctk.CTkLabel(
                bottom,
                text=f"-- {version_display(version)} --",
                text_color=TEXT_MUTED,
                font=("Segoe UI", 10),
                anchor="w",
            )
            divider.pack(fill=tk.X, padx=16, pady=(8, 2))
            for spec in specs_by_version[version]:
                row = ctk.CTkFrame(bottom, fg_color="transparent", height=32)
                row.pack(fill=tk.X, padx=16, pady=2)
                row.pack_propagate(False)
                ctk.CTkLabel(
                    row, text="●", text_color=spec["color_hex"],
                    font=ctk.CTkFont(size=12),
                ).pack(side=tk.LEFT, padx=(0, 6))
                ctk.CTkLabel(
                    row, text=spec["size_label"], text_color=TEXT_PRIMARY,
                    font=FONT_STATUS, anchor="w",
                ).pack(side=tk.LEFT)
                status = ctk.CTkLabel(
                    row, text="Loading...", text_color=TEXT_MUTED,
                    font=FONT_STATUS, anchor="e",
                )
                status.pack(side=tk.RIGHT)
                status_labels[spec["key"]] = status
        self._menu_widgets["sidebar_status"] = status_labels

        ctk.CTkLabel(
            bottom, text=APP_VERSION, text_color="#555555",
            font=FONT_VERSION,
        ).pack(pady=(8, 0))

        # Reflect already-loaded models (when returning to the menu).
        for key in list(self.models.keys()):
            self._apply_loaded_badge(key)
        return sidebar

    # %%
    def _on_nav_run(self) -> None:
        """Sidebar nav: show the Run configuration content view."""
        self._show_main_menu_view("run")

    # %%
    def _on_nav_settings(self) -> None:
        """Sidebar nav: show the Settings content view."""
        self._show_main_menu_view("settings")

    # %%
    def _on_nav_history(self) -> None:
        """Sidebar nav: full-window swap into the existing history screen."""
        self.show_history_screen()

    # %%
    def _on_nav_benchmarks(self) -> None:
        """Sidebar nav: full-window swap into the Benchmark History screen."""
        self.show_benchmarks_screen()

    # %%
    def _show_main_menu_view(self, view_name: str) -> None:
        """Rebuild the content area for the chosen view.

        Args:
            view_name: ``"run"`` or ``"settings"``.
        """
        self.menu_view = view_name
        content = self._menu_widgets.get("content_area")
        if content is None:
            return
        for child in list(content.winfo_children()):
            child.destroy()

        # Repaint the active-nav accent bar.
        nav_buttons = self._menu_widgets.get("nav_buttons", {}) or {}
        for name, parts in nav_buttons.items():
            if name == view_name:
                parts["bar"].configure(fg_color=ACCENT_ORANGE)
                parts["button"].configure(text_color=TEXT_PRIMARY)
            else:
                parts["bar"].configure(fg_color=BG_PRIMARY)
                parts["button"].configure(text_color=TEXT_MUTED)

        # Drop stale widget refs about to be re-created.
        for stale_key in (
            "model_dropdown", "backend_dots", "backend_radios",
            "precision_radios", "cpu_badge", "start_button",
            "benchmark_button", "video_list_container", "video_count_label",
            "loading_summary", "loading_progress",
            "version_radios", "size_container", "size_checkboxes",
        ):
            self._menu_widgets.pop(stale_key, None)

        if view_name == "settings":
            self._build_content_settings(content)
        else:
            self._build_content_run(content)

        self._refresh_backend_status()
        self._update_start_button_state()

    # %%
    def _build_content_run(self, parent: ctk.CTkBaseClass) -> None:
        """Build the Run configuration view inside the content area.

        Args:
            parent: The content-area frame.
        """
        action_bar = self._build_action_bar(parent)
        action_bar.pack(side=tk.BOTTOM, fill=tk.X)

        scroll = ctk.CTkScrollableFrame(
            parent, fg_color=BG_SECONDARY, corner_radius=0,
            scrollbar_fg_color=BG_SECONDARY,
            scrollbar_button_color=BORDER,
        )
        scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=18, pady=(18, 0))

        body = ctk.CTkFrame(scroll, fg_color=BG_SECONDARY, corner_radius=0)
        body.pack(fill=tk.BOTH, expand=True)
        # Columns share width 50/50; row 0 also expands so cards can stretch
        # vertically rather than floating at the top with empty space below.
        body.grid_columnconfigure(0, weight=1, uniform="cols")
        body.grid_columnconfigure(1, weight=1, uniform="cols")
        body.grid_rowconfigure(0, weight=1)

        left_col = ctk.CTkFrame(body, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_col = ctk.CTkFrame(body, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._build_model_selection_card(left_col)
        self._build_backend_precision_card(left_col)
        self._build_video_selection_card(right_col)

    # %%
    def _build_content_settings(self, parent: ctk.CTkBaseClass) -> None:
        """Build the Settings view inside the content area.

        Args:
            parent: The content-area frame.
        """
        scroll = ctk.CTkScrollableFrame(
            parent, fg_color=BG_SECONDARY, corner_radius=0,
            scrollbar_fg_color=BG_SECONDARY,
            scrollbar_button_color=BORDER,
        )
        scroll.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        card = self._make_card(scroll, "Detection Settings")
        card.pack(fill=tk.X, pady=(0, 16))

        ctk.CTkLabel(
            card, text="Confidence Thresholds", text_color=ACCENT_ORANGE,
            font=FONT_SECTION, anchor="w",
        ).pack(fill=tk.X, padx=20, pady=(8, 8))
        for spec in self.model_specs:
            self._build_conf_slider_row(card, spec)

        ctk.CTkLabel(
            card, text="Run Options", text_color=ACCENT_ORANGE,
            font=FONT_SECTION, anchor="w",
        ).pack(fill=tk.X, padx=20, pady=(16, 8))
        ctk.CTkSwitch(
            card, text="Save snapshots during run",
            variable=self.save_snapshots_var,
            progress_color=ACCENT_ORANGE,
            font=FONT_WIDGET,
        ).pack(anchor="w", padx=20, pady=8)
        ctk.CTkSwitch(
            card, text="Save run to history",
            variable=self.save_history_var,
            progress_color=ACCENT_ORANGE,
            font=FONT_WIDGET,
        ).pack(anchor="w", padx=20, pady=(8, 16))

    # %%
    def _make_card(self, parent: ctk.CTkBaseClass, title: str) -> ctk.CTkFrame:
        """Create a titled card frame in the standard dark style.

        Uses 16px horizontal padding on the title so card content lines up
        with the surrounding rows, plus 10px breathing room beneath the title
        before any widgets start.

        Args:
            parent: The container frame.
            title: Card title shown in orange at the top.

        Returns:
            The card frame; the caller packs it and adds children.
        """
        card = ctk.CTkFrame(
            parent, fg_color=BG_CARD, corner_radius=8,
            border_color=BORDER, border_width=1,
        )
        ctk.CTkLabel(
            card, text=title, text_color=ACCENT_ORANGE,
            font=FONT_SECTION, anchor="w",
        ).pack(anchor="w", padx=20, pady=(12, 8))
        return card

    # %%
    def _build_model_selection_card(self, parent: ctk.CTkBaseClass) -> None:
        """Build the Model Selection card (mode radios only).

        The model size is now driven by the size checkboxes in the Backend &
        Precision card; this card only chooses between Single-model and
        All-Models modes. ``_on_mode_change`` keeps the two stay-in-sync by
        flipping the size checkboxes into mutually-exclusive or all-locked
        behaviour as appropriate.

        Args:
            parent: The left-column container.
        """
        card = self._make_card(parent, "Model Selection")
        card.pack(fill=tk.X, pady=(0, 16))

        ctk.CTkRadioButton(
            card, text="Single Model", variable=self.mode_var, value="single",
            command=self._on_mode_change,
            fg_color=ACCENT_ORANGE, hover_color=ACCENT_ORANGE,
            font=FONT_WIDGET,
            radiobutton_width=20, radiobutton_height=20,
        ).pack(anchor="w", fill="x", padx=20, pady=8)
        ctk.CTkRadioButton(
            card, text="All Models (Split Screen)", variable=self.mode_var,
            value="split", command=self._on_mode_change,
            fg_color=ACCENT_ORANGE, hover_color=ACCENT_ORANGE,
            font=FONT_WIDGET,
            radiobutton_width=20, radiobutton_height=20,
        ).pack(anchor="w", fill="x", padx=20, pady=(0, 16))
        # Apply the current mode's behaviour to whatever checkboxes already exist.
        self._on_mode_change()

    # %%
    def _build_backend_precision_card(self, parent: ctk.CTkBaseClass) -> None:
        """Build the Backend & Precision card with consistent 16px padding.

        The card now fills its column both horizontally and vertically and each
        subsection (sizes / backend / precision / CPU) uses an isolated row
        frame with 18px gaps between siblings for breathing room.

        Args:
            parent: The left-column container.
        """
        card = self._make_card(parent, "Backend & Precision")
        card.pack(fill=tk.BOTH, expand=True, pady=(0, 16))

        # YOLO version selector --------------------------------------
        self._build_subsection_label(card, "YOLO version")
        version_row = ctk.CTkFrame(card, fg_color="transparent")
        version_row.pack(fill="x", padx=20, pady=(0, 8))
        version_radios: dict[str, ctk.CTkRadioButton] = {}
        for version in sorted(self.versions):
            accent = version_accent_color(version, sorted(self.versions))
            radio = ctk.CTkRadioButton(
                version_row,
                text=version_display(version),
                variable=self.version_var,
                value=version,
                command=self._on_version_change,
                fg_color=accent,
                hover_color=accent,
                font=FONT_WIDGET,
                radiobutton_width=20, radiobutton_height=20,
            )
            radio.pack(side=tk.LEFT, padx=(0, 18))
            version_radios[version] = radio
        self._menu_widgets["version_radios"] = version_radios

        # Model sizes (rebuilt every time the version changes) -------
        self._build_subsection_label(card, "Model sizes")
        size_container = ctk.CTkFrame(card, fg_color="transparent")
        size_container.pack(fill="x", padx=20, pady=(0, 8))
        self._menu_widgets["size_container"] = size_container
        self._menu_widgets["size_checkboxes"] = {}
        self._rebuild_size_checkboxes()

        # Backend ----------------------------------------------------
        self._build_subsection_label(card, "Backend")
        backend_row = ctk.CTkFrame(card, fg_color="transparent")
        backend_row.pack(fill="x", padx=20, pady=(0, 8))
        backend_dots: dict[str, ctk.CTkLabel] = {}
        backend_radios: dict[str, ctk.CTkRadioButton] = {}
        for backend in _BACKENDS:
            cell = ctk.CTkFrame(backend_row, fg_color="transparent")
            cell.pack(side=tk.LEFT, padx=(0, 18))
            radio = ctk.CTkRadioButton(
                cell, text=_BACKEND_DISPLAY[backend],
                variable=self.backend_var, value=backend,
                command=self._on_backend_change,
                fg_color=ACCENT_ORANGE, hover_color=ACCENT_ORANGE,
                font=FONT_WIDGET,
                radiobutton_width=20, radiobutton_height=20,
            )
            radio.pack(side=tk.LEFT)
            dot = ctk.CTkLabel(
                cell, text="●", text_color=_DOT_GREY,
                font=("Segoe UI", 16),
            )
            dot.pack(side=tk.LEFT, padx=(8, 0))
            backend_dots[backend] = dot
            backend_radios[backend] = radio
        self._menu_widgets["backend_dots"] = backend_dots
        self._menu_widgets["backend_radios"] = backend_radios

        # Precision --------------------------------------------------
        self._build_subsection_label(card, "Precision")
        precision_row = ctk.CTkFrame(card, fg_color="transparent")
        precision_row.pack(fill="x", padx=20, pady=(0, 8))
        precision_radios: dict[str, ctk.CTkRadioButton] = {}
        precision_dots: dict[str, ctk.CTkLabel] = {}
        for precision in _PRECISIONS:
            cell = ctk.CTkFrame(precision_row, fg_color="transparent")
            cell.pack(side=tk.LEFT, padx=(0, 18))
            radio = ctk.CTkRadioButton(
                cell, text=precision.upper(),
                variable=self.precision_var, value=precision,
                command=self._on_precision_change,
                fg_color=ACCENT_ORANGE, hover_color=ACCENT_ORANGE,
                font=FONT_WIDGET,
                radiobutton_width=20, radiobutton_height=20,
            )
            radio.pack(side=tk.LEFT)
            precision_radios[precision] = radio
            # INT8 is the only precision backed by its own engine file, so it
            # is the only one that needs a file-availability dot.
            if precision == _PRECISION_INT8:
                dot = ctk.CTkLabel(
                    cell, text="●", text_color=_DOT_GREY, font=("Segoe UI", 16),
                )
                dot.pack(side=tk.LEFT, padx=(8, 0))
                precision_dots[precision] = dot
                # A disabled radio cannot explain itself, so hover text does.
                self._menu_widgets["int8_tooltip"] = Tooltip(cell)
        self._menu_widgets["precision_radios"] = precision_radios
        self._menu_widgets["precision_dots"] = precision_dots

        # CPU toggle + warning badge --------------------------------
        ctk.CTkSwitch(
            card, text="CPU-only mode",
            variable=self.cpu_mode_var,
            onvalue=True, offvalue=False,
            command=self._on_cpu_toggle,
            progress_color=ACCENT_ORANGE,
            font=FONT_WIDGET,
        ).pack(anchor="w", fill="x", padx=20, pady=8)
        cpu_badge = ctk.CTkLabel(
            card, text="", text_color=ACCENT_YELLOW, fg_color="transparent",
            font=FONT_WIDGET, anchor="w",
            corner_radius=4, padx=8, pady=6,
        )
        cpu_badge.pack(anchor="w", fill="x", padx=20, pady=(0, 16))
        self._menu_widgets["cpu_badge"] = cpu_badge

    # %%
    def _build_subsection_label(self, parent: ctk.CTkBaseClass, text: str) -> None:
        """Render a left-aligned subsection label inside a card.

        Packed with ``anchor="w"`` (no ``fill=tk.X``) so customtkinter cannot
        clip the text to the widget's initial measured width - the source of
        the earlier "Backend" -> "Backenc" truncation.

        Args:
            parent: The card the label is packed into.
            text: Label text.
        """
        ctk.CTkLabel(
            parent, text=text, text_color=TEXT_MUTED,
            font=FONT_SUBLABEL, anchor="w", justify="left",
        ).pack(anchor="w", padx=20, pady=(8, 4))

    # %%
    def _build_video_selection_card(self, parent: ctk.CTkBaseClass) -> None:
        """Build the Video Selection card (checkbox list + browse/refresh).

        Fills both width and height of its column. The inner scrollable list
        expands so it absorbs all extra vertical space; the browse/refresh
        button row + count label stay pinned at the bottom of the card.

        Args:
            parent: The right-column container.
        """
        card = self._make_card(parent, "Video Selection")
        card.pack(fill=tk.BOTH, expand=True, pady=(0, 16))

        # Count label first (packed bottom) so it always sits at the foot
        # regardless of how tall the scrollable list grows.
        count_label = ctk.CTkLabel(
            card, textvariable=self.selected_video_count_var,
            text_color=TEXT_MUTED, font=FONT_SMALL,
            anchor="w",
        )
        count_label.pack(side=tk.BOTTOM, anchor="w", fill="x", padx=20, pady=(6, 14))
        self._menu_widgets["video_count_label"] = count_label

        button_row = ctk.CTkFrame(card, fg_color="transparent")
        button_row.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=8)
        ctk.CTkButton(
            button_row, text="Browse...", command=self._on_browse_videos,
            fg_color=BG_HOVER, hover_color="#2A2A2A",
            text_color=TEXT_PRIMARY, font=FONT_WIDGET, width=90, height=36,
        ).pack(side=tk.LEFT)
        ctk.CTkButton(
            button_row, text="Refresh", command=self._on_refresh_clicked,
            fg_color=BG_HOVER, hover_color="#2A2A2A",
            text_color=TEXT_PRIMARY, font=FONT_WIDGET, width=90, height=36,
        ).pack(side=tk.LEFT, padx=8)

        # Scrollable list takes all remaining vertical space.
        list_container = ctk.CTkScrollableFrame(
            card, fg_color=BG_PRIMARY, corner_radius=6,
            label_text="",
            scrollbar_fg_color=BG_PRIMARY,
            scrollbar_button_color=BORDER,
        )
        list_container.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=(0, 6),
        )
        self._menu_widgets["video_list_container"] = list_container

        self._refresh_video_list()

    # %%
    def _build_action_bar(self, parent: ctk.CTkBaseClass) -> ctk.CTkFrame:
        """Build the pinned bottom action bar (loading summary + buttons).

        Args:
            parent: The content-area frame.

        Returns:
            The action bar frame, ready to be packed at the bottom.
        """
        bar = ctk.CTkFrame(parent, fg_color=BG_PRIMARY, corner_radius=0, height=84)
        bar.pack_propagate(False)
        ctk.CTkFrame(bar, height=1, fg_color=BORDER, corner_radius=0).pack(
            fill=tk.X, side=tk.TOP,
        )

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        loaded = self.models_loaded.is_set()
        left_summary = ctk.CTkFrame(inner, fg_color="transparent")
        left_summary.pack(side=tk.LEFT, fill=tk.Y)
        loading_label = ctk.CTkLabel(
            left_summary,
            text="All models ready" if loaded else self.loading_status.get(),
            text_color=ACCENT_GREEN if loaded else TEXT_MUTED,
            font=FONT_WIDGET, anchor="w",
        )
        loading_label.pack(anchor="w")
        progress = ctk.CTkProgressBar(
            left_summary, width=200, mode="indeterminate",
            progress_color=ACCENT_ORANGE,
        )
        if not loaded:
            progress.pack(anchor="w", pady=(4, 0))
            progress.start()
        self._menu_widgets["loading_summary"] = loading_label
        self._menu_widgets["loading_progress"] = progress

        quit_button = ctk.CTkButton(
            inner, text="Quit", command=self._on_close,
            fg_color="transparent", border_color="#C0392B", border_width=1,
            hover_color="#3A1410", text_color="#E27160",
            font=FONT_BUTTON, width=110, height=44,
        )
        quit_button.pack(side=tk.RIGHT)
        start_button = ctk.CTkButton(
            inner, text="Start Run", command=self._on_start_run,
            fg_color=ACCENT_ORANGE, hover_color="#FF8A33",
            text_color=TEXT_PRIMARY,
            font=FONT_BUTTON, width=180, height=44,
        )
        start_button.pack(side=tk.RIGHT, padx=(0, 8))
        bench_button = ctk.CTkButton(
            inner, text="Full Benchmark", command=self._on_full_benchmark,
            fg_color="transparent", border_color=ACCENT_TEAL, border_width=1,
            hover_color="#0F2D2A", text_color=ACCENT_TEAL,
            font=FONT_BUTTON, width=180, height=44,
        )
        bench_button.pack(side=tk.RIGHT, padx=(0, 8))

        self._menu_widgets["start_button"] = start_button
        self._menu_widgets["benchmark_button"] = bench_button
        return bar

    # %%
    def _build_conf_slider_row(
        self, parent: ctk.CTkBaseClass, spec: dict[str, Any],
    ) -> None:
        """Build one confidence-slider row inside the Settings card.

        Args:
            parent: The Settings card.
            spec: The model spec the slider controls.
        """
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X, padx=20, pady=8)
        ctk.CTkLabel(
            row, text="●", text_color=spec["color_hex"],
            font=("Segoe UI", 14),
        ).pack(side=tk.LEFT)
        ctk.CTkLabel(
            row, text=f" {spec['display']}", text_color=TEXT_PRIMARY,
            font=FONT_WIDGET, anchor="w",
        ).pack(side=tk.LEFT, padx=(2, 8))
        value_label = ctk.CTkLabel(
            row, text=f"{self.live_conf[spec['key']]:.2f}",
            text_color=TEXT_MUTED, font=FONT_WIDGET, width=48,
        )
        value_label.pack(side=tk.RIGHT)
        slider = ctk.CTkSlider(
            row, from_=_CONF_MIN, to=_CONF_MAX,
            number_of_steps=int(round((_CONF_MAX - _CONF_MIN) / _CONF_STEP)),
            progress_color=spec["color_hex"],
            button_color=spec["color_hex"],
            button_hover_color=spec["color_hex"],
            width=180,
            command=functools.partial(
                self._on_conf_slider_change, spec["key"], value_label,
            ),
        )
        slider.set(self.live_conf[spec["key"]])
        slider.pack(side=tk.RIGHT, padx=(8, 8))

    # %%
    def _on_conf_slider_change(
        self,
        model_key: str,
        value_label: ctk.CTkLabel,
        value: float,
    ) -> None:
        """ctk slider callback: snap to step, update the label + live state.

        Args:
            model_key: The model key whose conf changed.
            value_label: The label that displays the current value.
            value: Slider value supplied by customtkinter.
        """
        snapped = round(value / _CONF_STEP) * _CONF_STEP
        snapped = max(_CONF_MIN, min(_CONF_MAX, snapped))
        value_label.configure(text=f"{snapped:.2f}")
        self._on_conf_change(model_key, str(snapped))

    # %%
    def _on_conf_change(self, model_key: str, value_str: str) -> None:
        """Push a new confidence threshold to live state + the worker.

        Args:
            model_key: The model key whose conf changed.
            value_str: The new value as a string.
        """
        try:
            value = float(value_str)
        except ValueError:
            return
        self.live_conf[model_key] = value
        worker = self.workers.get(model_key)
        if worker is not None:
            worker.set_conf(value)

    # %%
    def _on_version_change(self) -> None:
        """Version-radio callback: rebuild the size-checkbox row.

        Checked sizes are preserved when the same size letter exists under
        the new version so users keep their selection across version flips.
        """
        version = self.version_var.get()
        # Remember currently checked size letters under the previous version
        # so we can carry them over.
        previously_checked: set[str] = set()
        for spec in self.model_specs:
            if self.selected_size_vars[spec["key"]].get():
                previously_checked.add(spec["size_letter"])

        # Clear all selection in size vars for the new version, then re-apply
        # the carried-over checks where the new version has matching sizes.
        for spec in self.model_specs:
            if spec.get("version") == version:
                self.selected_size_vars[spec["key"]].set(
                    spec["size_letter"] in previously_checked,
                )
            else:
                self.selected_size_vars[spec["key"]].set(False)

        # If nothing carried over, fall back to the first available size.
        active_specs = [
            spec for spec in self.model_specs if spec.get("version") == version
        ]
        if active_specs and not any(
            self.selected_size_vars[spec["key"]].get() for spec in active_specs
        ):
            self.selected_size_vars[active_specs[0]["key"]].set(True)

        self._rebuild_size_checkboxes()
        self._on_mode_change()

    # %%
    def _rebuild_size_checkboxes(self) -> None:
        """Render the size-checkbox row for the currently-selected version.

        Greyed-out and unchecked checkboxes are still shown for sizes whose
        ``.pt`` is missing on disk, so the user sees the gap rather than a
        silently shorter row.
        """
        container = self._menu_widgets.get("size_container")
        if container is None:
            return
        for child in list(container.winfo_children()):
            child.destroy()

        version = self.version_var.get()
        version_specs = [
            spec for spec in self.model_specs if spec.get("version") == version
        ]
        size_checkboxes: dict[str, ctk.CTkCheckBox] = {}
        per_row = 4
        current_row: ctk.CTkFrame | None = None
        for index, spec in enumerate(version_specs):
            if index % per_row == 0:
                current_row = ctk.CTkFrame(container, fg_color="transparent")
                current_row.pack(fill="x", pady=(0, 4))
            pt_path = self.lookup_weights(spec["key"], _BACKEND_PYTORCH)
            disabled = pt_path is None
            if disabled:
                # Greyed checkboxes cannot stay checked.
                self.selected_size_vars[spec["key"]].set(False)
            accent = spec.get("color_hex", ACCENT_ORANGE)
            checkbox = ctk.CTkCheckBox(
                current_row,
                text=spec["size_label"],
                variable=self.selected_size_vars[spec["key"]],
                onvalue=True, offvalue=False,
                command=functools.partial(self._on_size_toggle, spec["key"]),
                fg_color=accent,
                hover_color=accent,
                checkmark_color=TEXT_PRIMARY,
                font=FONT_WIDGET,
                checkbox_width=20, checkbox_height=20,
                state="disabled" if disabled else "normal",
            )
            checkbox.pack(side=tk.LEFT, padx=(0, 18))
            size_checkboxes[spec["key"]] = checkbox
        self._menu_widgets["size_checkboxes"] = size_checkboxes

    # %%
    def _on_mode_change(self) -> None:
        """Mode radio callback: rewire size checkboxes for the chosen mode.

        * Single mode: enable every size checkbox for the active version; if
          the active version has anything other than exactly one size checked,
          snap the selection back to the first available size so single-model
          runs always have a valid target.
        * Split mode: check every available size for the active version, then
          disable the checkboxes so the user cannot uncheck them mid-config.
        """
        size_checkboxes = self._menu_widgets.get("size_checkboxes", {}) or {}
        mode = self.mode_var.get()
        version = self.version_var.get()
        active_keys = [
            spec["key"] for spec in self.model_specs
            if spec.get("version") == version
            and self.lookup_weights(spec["key"], _BACKEND_PYTORCH) is not None
        ]
        if mode == "single":
            checked_now = [k for k in active_keys if self.selected_size_vars[k].get()]
            if len(checked_now) != 1 and active_keys:
                for key in active_keys:
                    self.selected_size_vars[key].set(key == active_keys[0])
            for key, checkbox in size_checkboxes.items():
                if key in active_keys:
                    checkbox.configure(state="normal")
        else:
            for key in active_keys:
                self.selected_size_vars[key].set(True)
            for key, checkbox in size_checkboxes.items():
                if key in active_keys:
                    checkbox.configure(state="disabled")
        self._refresh_backend_status()

    # %%
    def _on_size_toggle(self, model_key: str | None = None) -> None:
        """Model-size checkbox callback.

        In single mode the size checkboxes act as a radio group within the
        currently-selected YOLO version: clicking one clears the others.
        In split mode the checkboxes are disabled so this callback merely
        re-checks the box the user couldn't toggle.

        Args:
            model_key: Optional key of the checkbox that fired; required to
                implement the radio behaviour in single mode.
        """
        if self.mode_var.get() == "single" and model_key is not None:
            spec = self.model_by_key.get(model_key)
            version = spec.get("version") if spec else None
            for other_key, other_var in self.selected_size_vars.items():
                other_spec = self.model_by_key.get(other_key)
                if (
                    other_key != model_key
                    and other_spec is not None
                    and other_spec.get("version") == version
                ):
                    other_var.set(False)
            # If the user managed to uncheck the only checked size, force it
            # back on so the run is always valid.
            if not self.selected_size_vars[model_key].get():
                self.selected_size_vars[model_key].set(True)
        self._refresh_backend_status()

    # %%
    def _on_backend_change(self) -> None:
        """Backend radio callback: refresh precision support + button state."""
        self._refresh_precision_state()
        self._update_start_button_state()

    # %%
    def _on_precision_change(self) -> None:
        """Precision radio callback: refresh button state."""
        self._update_start_button_state()

    # %%
    def _on_cpu_toggle(self) -> None:
        """CPU-toggle callback: snap TensorRT off + recompute availability."""
        if self.cpu_mode_var.get() and self.backend_var.get() == _BACKEND_TENSORRT:
            self.backend_var.set(_BACKEND_PYTORCH)
        self._refresh_backend_status()

    # %%
    def _on_refresh_clicked(self) -> None:
        """Refresh button: rescan ``inputs/`` and re-discover ``models/``.

        Discovery is rerun so newly-dropped weight files are picked up
        without restarting the app. The size-checkbox row and sidebar
        status rows are rebuilt to match the updated registry.
        """
        self.refresh_discovery()
        # Rebuild the version radios + size row in case a new version landed.
        self._show_main_menu_view(self.menu_view)
        self._refresh_video_list()

    # %%
    def _refresh_video_list(self) -> None:
        """Rescan ``inputs/`` and rebuild the per-video checkbox list."""
        container = self._menu_widgets.get("video_list_container")
        if container is None:
            return
        for child in list(container.winfo_children()):
            child.destroy()
        self._video_checkbox_vars = {}

        paths = list_videos(_INPUTS_DIR)
        if not paths:
            ctk.CTkLabel(
                container, text="(no videos in inputs/)",
                text_color=TEXT_MUTED,
                font=FONT_SMALL,
            ).pack(anchor="w", padx=8, pady=6)
        for path in paths:
            var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                container, text=path.name, variable=var,
                command=self._on_video_check,
                fg_color=ACCENT_ORANGE, hover_color=ACCENT_ORANGE,
                font=FONT_WIDGET,
                checkbox_width=20, checkbox_height=20,
            ).pack(anchor="w", padx=4, pady=6)
            self._video_checkbox_vars[path.name] = var

        self.selected_video_count_var.set("0 video(s) selected")
        self._update_start_button_state()

    # %%
    def _on_browse_videos(self) -> None:
        """Open a file dialog and copy chosen videos into ``inputs/``."""
        paths = filedialog.askopenfilenames(
            title="Add videos",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
        )
        if not paths:
            return
        _INPUTS_DIR.mkdir(parents=True, exist_ok=True)
        for raw in paths:
            source = Path(raw)
            if source.suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            destination = _INPUTS_DIR / source.name
            if not destination.exists():
                try:
                    shutil.copy2(source, destination)
                except OSError as exc:
                    messagebox.showerror(
                        "Copy failed", f"Could not copy {source.name}: {exc}",
                    )
        self._refresh_video_list()

    # %%
    def _on_video_check(self) -> None:
        """Video checkbox callback: refresh selected-count + button state."""
        count = sum(1 for var in self._video_checkbox_vars.values() if var.get())
        self.selected_video_count_var.set(f"{count} video(s) selected")
        self._update_start_button_state()

    # %%
    def _selected_video_names(self) -> list[str]:
        """Return the names of currently-checked video checkboxes.

        Returns:
            Filenames in stable iteration order.
        """
        return [name for name, var in self._video_checkbox_vars.items() if var.get()]

    # %%
    def _update_start_button_state(self) -> None:
        """Enable Start Run / Full Benchmark when prerequisites are met.

        Start Run still requires the size selection from the Backend &
        Precision card; Full Benchmark only requires the loader to be done
        and at least one video selected (the strict one-video rule is
        enforced inside the click handler so the user gets a precise error).
        """
        video_count = len(self._selected_video_names())
        ready_models = self.models_loaded.is_set()
        has_size = bool(self.get_selected_size_keys())

        start_button = self._menu_widgets.get("start_button")
        if start_button is not None:
            start_button.configure(
                state="normal"
                if (ready_models and video_count >= 1)
                else "disabled",
            )
        bench_button = self._menu_widgets.get("benchmark_button")
        if bench_button is not None:
            bench_button.configure(
                state="normal"
                if (ready_models and video_count >= 1 and has_size)
                else "disabled",
            )

    # %%
    def get_selected_size_keys(self) -> list[str]:
        """Return the version-aware model keys whose checkbox is ticked.

        Keys are composed as ``f"{version}{size_letter}-seg"`` and are
        restricted to the currently-selected YOLO version so split-screen
        runs and benchmarks operate within a single version at a time.

        Returns:
            The list of selected model keys in canonical (size) order.
        """
        version = self.version_var.get()
        return [
            spec["key"]
            for spec in self.model_specs
            if spec.get("version") == version
            and self.selected_size_vars[spec["key"]].get()
        ]

    # %%
    def _refresh_backend_status(self) -> None:
        """Recompute backend availability dots + radio-button enable state."""
        if self.current_screen != "main_menu":
            return
        cpu_mode = bool(self.cpu_mode_var.get())
        selected = self.get_selected_size_keys()
        availability = backend_availability(selected, self.backend_paths, cpu_mode)

        dots = self._menu_widgets.get("backend_dots", {}) or {}
        radios = self._menu_widgets.get("backend_radios", {}) or {}
        for backend, dot in dots.items():
            dot.configure(text_color=availability.get(backend, _DOT_GREY))
        for backend, radio in radios.items():
            disable = availability.get(backend) == _DOT_RED
            radio.configure(state="disabled" if disable else "normal")

        current = self.backend_var.get()
        if availability.get(current) == _DOT_RED:
            for backend in _BACKENDS:
                if availability.get(backend) != _DOT_RED:
                    self.backend_var.set(backend)
                    break
            else:
                self.backend_var.set(_BACKEND_PYTORCH)
        self._refresh_precision_state()
        self._refresh_cpu_badge()
        self._update_start_button_state()

    # %%
    def _refresh_precision_state(self) -> None:
        """Disable precision radios that the current backend does not support.

        Also refreshes the INT8 file-availability dot and the hover tooltip
        explaining why INT8 is greyed out on a non-TensorRT backend.
        """
        backend = self.backend_var.get()
        cpu_mode = bool(self.cpu_mode_var.get())
        selected = self.get_selected_size_keys()
        radios = self._menu_widgets.get("precision_radios", {}) or {}
        dots = self._menu_widgets.get("precision_dots", {}) or {}

        availability = precision_availability(
            selected, self.backend_paths, backend, cpu_mode,
        )
        for precision, dot in dots.items():
            dot.configure(text_color=availability.get(precision, _DOT_GREY))

        for precision, radio in radios.items():
            supported = (backend, precision) in _SUPPORTED_BACKEND_PRECISIONS
            # INT8 additionally needs its engine files to exist.
            if supported and precision == _PRECISION_INT8:
                supported = availability.get(precision) != _DOT_RED
            radio.configure(state="normal" if supported else "disabled")

        tooltip = self._menu_widgets.get("int8_tooltip")
        if tooltip is not None:
            if backend != _BACKEND_TENSORRT:
                tooltip.text = _INT8_REQUIRES_TRT_TOOLTIP
            elif availability.get(_PRECISION_INT8) == _DOT_RED:
                tooltip.text = "No *_crack_int8.engine for the selected size(s)"
            else:
                tooltip.text = ""

        current_precision = self.precision_var.get()
        current_radio = radios.get(current_precision)
        current_disabled = (
            current_radio is not None and str(current_radio.cget("state")) == "disabled"
        )
        if (backend, current_precision) not in _SUPPORTED_BACKEND_PRECISIONS or current_disabled:
            for precision in _PRECISIONS:
                radio = radios.get(precision)
                enabled = radio is None or str(radio.cget("state")) != "disabled"
                if (backend, precision) in _SUPPORTED_BACKEND_PRECISIONS and enabled:
                    self.precision_var.set(precision)
                    break

    # %%
    def _refresh_cpu_badge(self) -> None:
        """Show / hide the yellow CPU-mode warning badge in the menu."""
        badge = self._menu_widgets.get("cpu_badge")
        if badge is None:
            return
        if self.cpu_mode_var.get():
            badge.configure(
                text="⚠  CPU mode active - reduced FPS",
                text_color=ACCENT_YELLOW,
                fg_color="#3D3000",
            )
        else:
            badge.configure(text="", fg_color="transparent")

    # %%
    def _on_start_run(self) -> None:
        """Validate selection and transition from the main menu to the Run Screen."""
        names = self._selected_video_names()
        if not names:
            return
        video_paths = [_INPUTS_DIR / name for name in names if (_INPUTS_DIR / name).is_file()]
        if not video_paths:
            messagebox.showerror("No videos", "The selected videos no longer exist.")
            return

        backend = self.backend_var.get()
        precision = self.precision_var.get()
        cpu_mode = bool(self.cpu_mode_var.get())

        checked_sizes = self.get_selected_size_keys()
        if not checked_sizes:
            messagebox.showerror(
                "No model", "Check at least one model size in Backend & Precision.",
            )
            return
        if self.mode_var.get() == "single":
            # Single mode forces mutually-exclusive checkboxes elsewhere, so
            # the first (and only) checked entry is the run's model.
            model_keys = [checked_sizes[0]]
        else:
            # Split mode runs every checked size for the active version.
            model_keys = list(checked_sizes)

        storage_key = backend_storage_key(backend, precision)
        loaded_for_run = [
            key for key in model_keys if (key, storage_key) in self.backends
        ]
        if not loaded_for_run:
            flavour = (
                f"{_BACKEND_DISPLAY[backend]} {precision.upper()}"
                if storage_key != backend
                else _BACKEND_DISPLAY[backend]
            )
            messagebox.showerror(
                "Backend unavailable",
                f"No {flavour} weights are loaded for the selected model "
                "size(s). Pick a different backend or precision, or drop the "
                "matching files into models/.",
            )
            return

        # Keep ``single_model_var`` in sync so existing downstream readers
        # (run-summary display name, etc.) still get a meaningful value.
        if self.mode_var.get() == "single":
            self.single_model_var.set(self.model_by_key[model_keys[0]]["display"])

        self._start_run(video_paths, model_keys, backend, precision, cpu_mode)


    # ------------------------------------------------------------------
    # Run screen
    # ------------------------------------------------------------------
    # %%
    def _start_run(
        self,
        video_paths: list[Path],
        model_keys: list[str],
        backend: str = _DEFAULT_BACKEND,
        precision: str = _DEFAULT_PRECISION,
        cpu_mode: bool = False,
    ) -> None:
        """Build the run screen and launch the video-processing thread.

        Args:
            video_paths: Ordered list of videos to play through.
            model_keys: Model keys active during this run (1 or 3).
            backend: Backend identifier the run should use.
            precision: Precision identifier the run should use.
            cpu_mode: Whether the run is forced onto the CPU.
        """
        run_id = uuid.uuid4().hex

        # Honour the CPU-only toggle: relocate every cached wrapper to the
        # requested device before we hand them to the workers. This is the
        # only place that turns the toggle into an actual model move.
        self._apply_cpu_mode_to_backends(cpu_mode)

        # Resolve the active ModelBackend per requested size + swap workers
        # over to those backends so split-screen renders the chosen flavour.
        # TensorRT INT8 lives in its own registry slot because it is a separate
        # engine file, even though the UI models it as a TensorRT precision.
        storage_key = backend_storage_key(backend, precision)
        active_backends: dict[str, ModelBackend | None] = {}
        for key in model_keys:
            wrapper = self.backends.get((key, storage_key))
            active_backends[key] = wrapper
            if wrapper is not None:
                wrapper.precision = (
                    precision
                    if (backend, precision) in _SUPPORTED_BACKEND_PRECISIONS
                    else wrapper.precision
                )
                worker = self.workers.get(key)
                if worker is not None:
                    worker.set_backend(wrapper)

        self.run_state = {
            "run_id": run_id,
            "mode": "single" if len(model_keys) == 1 else "split",
            "model_keys": model_keys,
            "backend": backend,
            "precision": precision,
            "cpu_mode": cpu_mode,
            "active_device": (
                "cpu" if cpu_mode
                else (
                    active_backends[model_keys[0]].device
                    if active_backends.get(model_keys[0]) is not None
                    else self.device
                )
            ),
            "active_backends": active_backends,
            "video_list": video_paths,
            "stop_event": threading.Event(),
            "paused": False,
            "restart_requested": False,
            "stats": {
                key: {
                    "frames": 0,
                    "total_detections": 0,
                    "frames_with_det": 0,
                    "latencies": [],
                }
                for key in model_keys
            },
            "snapshot_paths": [],
            "snapshots_saved": 0,
            "recent_frame_times": deque(maxlen=_FPS_WINDOW),
            "last_annotated": None,
            "last_meta": None,
            "start_perf": time.perf_counter(),
            "start_timestamp": datetime.now().isoformat(timespec="seconds"),
            "save_history": bool(self.save_history_var.get()),
            "save_snapshots": bool(self.save_snapshots_var.get()),
            "running": True,
            "current_video_index": 0,
        }

        # Drain any leftover display frames from a previous run.
        try:
            while True:
                self._display_queue.get_nowait()
        except queue.Empty:
            pass

        self._build_run_screen()

        self.video_thread = threading.Thread(
            target=self._video_thread_main, name="video-thread", daemon=True
        )
        self.video_thread.start()
        self.root.after(_DISPLAY_TICK_MS, self._display_tick)

    # %%
    def _build_run_screen(self) -> None:
        """Construct the Run Screen widgets (canvas + key-help toolbar)."""
        frame = tk.Frame(self.root, bg=_BG_DARK)

        canvas = tk.Canvas(
            frame,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="black",
            highlightthickness=0,
        )
        canvas.pack(padx=px(20), pady=px(20))
        self.canvas = canvas

        # Key-help toolbar below the canvas.
        help_bar = tk.Frame(frame, bg=_BG_PANEL)
        help_bar.pack(fill=tk.X, padx=px(20), pady=(0, px(10)))
        tk.Label(
            help_bar,
            text=(
                "ESC: end run    SPACE: pause/resume    S: snapshot    "
                "R: restart    + / -: confidence ±0.05"
            ),
            bg=_BG_PANEL,
            fg=_FG_GREY,
            font=("Helvetica", px(10)),
            padx=px(10),
            pady=px(6),
        ).pack(side=tk.LEFT)

        # Bind keys to root so they fire regardless of focus.
        self.root.bind("<Escape>", self._on_key_escape)
        self.root.bind("<space>", self._on_key_space)
        self.root.bind("<s>", self._on_key_snapshot)
        self.root.bind("<S>", self._on_key_snapshot)
        self.root.bind("<r>", self._on_key_restart)
        self.root.bind("<R>", self._on_key_restart)
        self.root.bind("<plus>", self._on_key_plus)
        self.root.bind("<KP_Add>", self._on_key_plus)
        self.root.bind("<equal>", self._on_key_plus)
        self.root.bind("<minus>", self._on_key_minus)
        self.root.bind("<KP_Subtract>", self._on_key_minus)
        self.root.bind("<underscore>", self._on_key_minus)

        self._swap_frame(frame, "run")

    # %%
    def _unbind_run_keys(self) -> None:
        """Remove the run-screen key bindings (called when leaving the screen)."""
        for sequence in (
            "<Escape>",
            "<space>",
            "<s>",
            "<S>",
            "<r>",
            "<R>",
            "<plus>",
            "<KP_Add>",
            "<equal>",
            "<minus>",
            "<KP_Subtract>",
            "<underscore>",
        ):
            try:
                self.root.unbind(sequence)
            except tk.TclError:
                continue

    # %%
    def _on_key_escape(self, _event: tk.Event) -> None:
        """ESC handler: signal the video thread to stop the run."""
        if self.run_state is not None:
            self.run_state["stop_event"].set()

    # %%
    def _on_key_space(self, _event: tk.Event) -> None:
        """SPACE handler: toggle pause."""
        if self.run_state is not None:
            self.run_state["paused"] = not self.run_state["paused"]

    # %%
    def _on_key_snapshot(self, _event: tk.Event) -> None:
        """S handler: save the most recently rendered annotated frame."""
        self._save_snapshot()

    # %%
    def _on_key_restart(self, _event: tk.Event) -> None:
        """R handler: ask the video thread to seek the current video to 0."""
        if self.run_state is not None:
            self.run_state["restart_requested"] = True

    # %%
    def _on_key_plus(self, _event: tk.Event) -> None:
        """+ handler: bump every active model's confidence by ``_CONF_STEP``."""
        if self.run_state is None:
            return
        for key in self.run_state["model_keys"]:
            new_conf = min(_CONF_MAX, self.live_conf[key] + _CONF_STEP)
            self.live_conf[key] = new_conf
            worker = self.workers.get(key)
            if worker is not None:
                worker.set_conf(new_conf)

    # %%
    def _on_key_minus(self, _event: tk.Event) -> None:
        """- handler: drop every active model's confidence by ``_CONF_STEP``."""
        if self.run_state is None:
            return
        for key in self.run_state["model_keys"]:
            new_conf = max(_CONF_MIN, self.live_conf[key] - _CONF_STEP)
            self.live_conf[key] = new_conf
            worker = self.workers.get(key)
            if worker is not None:
                worker.set_conf(new_conf)

    # %%
    def _save_snapshot(self) -> None:
        """Write the current annotated frame + JSON sidecar to ``outputs/``."""
        state = self.run_state
        if state is None or not state.get("save_snapshots", True):
            return
        annotated = state.get("last_annotated")
        meta = state.get("last_meta")
        if annotated is None or meta is None:
            return
        _SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        run_id = state["run_id"]
        frame_index = int(meta["frame_index"])
        jpg_path = _SNAPSHOTS_DIR / f"{run_id}_{frame_index:05d}.jpg"
        json_path = jpg_path.with_suffix(".json")
        cv2.imwrite(str(jpg_path), annotated)
        per_model: dict[str, dict[str, Any]] = meta["per_model"]
        max_conf = max(
            (float(model_meta.get("max_conf", 0.0)) for model_meta in per_model.values()),
            default=0.0,
        )
        sidecar = {
            "frame_index": frame_index,
            "video": meta.get("video"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "max_confidence": max_conf,
            "per_model": {
                name: {
                    "confidence_threshold": float(model_meta.get("conf_threshold", 0.0)),
                    "detections": int(model_meta.get("detections", 0)),
                    "inference_time_ms": float(model_meta.get("inference_ms", 0.0)),
                    "max_confidence": float(model_meta.get("max_conf", 0.0)),
                }
                for name, model_meta in per_model.items()
            },
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2)
        state["snapshot_paths"].append(str(jpg_path))
        state["snapshots_saved"] += 1

    # ------------------------------------------------------------------
    # Video thread
    # ------------------------------------------------------------------
    # %%
    def _video_thread_main(self) -> None:
        """Background thread: read frames, run inference, push to display."""
        state = self.run_state
        if state is None:
            return

        for video_index, video_path in enumerate(state["video_list"]):
            if state["stop_event"].is_set():
                break
            state["current_video_index"] = video_index
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                continue
            native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ideal_seconds_per_frame = 1.0 / max(native_fps, 1.0)

            while not state["stop_event"].is_set():
                if state["paused"]:
                    time.sleep(0.03)
                    continue
                if state["restart_requested"]:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state["restart_requested"] = False
                    continue

                frame_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

                if state["mode"] == "single":
                    self._process_single(state, frame, frame_index, total_frames, video_path)
                else:
                    self._process_split(state, frame, frame_index, total_frames, video_path)

                # Update wall-clock FPS deque for the composite display.
                wall_elapsed = time.perf_counter() - frame_start
                state["recent_frame_times"].append(wall_elapsed)

                # Pace to native FPS (sleep in small chunks so stop is responsive).
                remaining = ideal_seconds_per_frame - wall_elapsed
                if remaining > 0:
                    end_time = time.perf_counter() + remaining
                    while time.perf_counter() < end_time:
                        if state["stop_event"].is_set():
                            break
                        time.sleep(max(0.0, min(0.005, end_time - time.perf_counter())))

            capture.release()
            if state["stop_event"].is_set():
                break
            # Show transition overlay if there's another video coming up.
            if video_index < len(state["video_list"]) - 1:
                remaining_videos = len(state["video_list"]) - video_index - 1
                self.root.after(0, self._show_video_transition, remaining_videos)
                transition_start = time.perf_counter()
                while time.perf_counter() - transition_start < _VIDEO_TRANSITION_SECONDS:
                    if state["stop_event"].is_set():
                        break
                    time.sleep(0.05)
                self.root.after(0, self._hide_video_transition)

        # Run finished: build summary on main thread.
        summary = self._build_run_summary(state)
        state["running"] = False
        if state.get("save_history", True) and summary["per_model_stats"]:
            self.history.add(summary)
        self.root.after(0, self._on_run_end, summary)

    # %%
    def _process_single(
        self,
        state: dict[str, Any],
        frame: np.ndarray,
        frame_index: int,
        total_frames: int,
        video_path: Path,
    ) -> None:
        """Single-model branch of the video thread loop.

        Args:
            state: The current run state.
            frame: The raw BGR frame.
            frame_index: 1-based position of the frame in the video.
            total_frames: Total frames in the source video.
            video_path: Path to the source video (for snapshot metadata).
        """
        key = state["model_keys"][0]
        spec = self.models[key]["spec"]
        conf = float(self.live_conf[key])

        backend_id: str = state["backend"]
        precision_id: str = state["precision"]
        wrapper: ModelBackend | None = state["active_backends"].get(key)

        if wrapper is None:
            # Backend file was missing - fall back to the default PyTorch model
            # so live playback never crashes mid-run.
            model = self.models[key]["model"]
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model.predict(
                frame, conf=conf, iou=_IOU_DEFAULT, device=self.device, verbose=False
            )[0]
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - t0) * 1000.0
        else:
            result, inference_ms = wrapper.predict(frame, conf=conf)

        annotated, count, max_conf = apply_overlays(frame, result, spec)

        stats = state["stats"][key]
        stats["frames"] += 1
        stats["total_detections"] += count
        if count > 0:
            stats["frames_with_det"] += 1
        stats["latencies"].append(inference_ms)

        display = letterbox(annotated, CANVAS_W, CANVAS_H)

        rolling_fps = (
            (len(state["recent_frame_times"]) / sum(state["recent_frame_times"]))
            if state["recent_frame_times"]
            else 0.0
        )
        per_frame_fps = (1000.0 / inference_ms) if inference_ms > 0 else 0.0
        backend_color_bgr = hex_to_bgr(_BACKEND_COLOR_HEX[backend_id])

        hud_lines: list[tuple[str, tuple[int, int, int], int]] = [
            (spec["display"], hex_to_bgr(spec["color_hex"]), 2),
            (f"Model: {spec['display']}", _BGR_WHITE, 1),
            (f"Backend: {_BACKEND_DISPLAY[backend_id]}", backend_color_bgr, 2),
            (f"Precision: {precision_id.upper()}", _BGR_WHITE, 1),
            (
                f"{per_frame_fps:.1f} FPS | {inference_ms:.1f} ms",
                _BGR_WHITE,
                1,
            ),
            (
                f"Rolling: {rolling_fps:.1f} FPS",
                _BGR_WHITE,
                1,
            ),
            (f"Detections: {count}   Conf: {conf:.2f}", _BGR_WHITE, 1),
            (f"Frame: {frame_index} / {total_frames}", _BGR_WHITE, 1),
        ]
        if not self.models[key]["used_finetuned"]:
            hud_lines.append(("pretrained fallback", _BGR_RED, 2))
        draw_hud_panel(display, hud_lines, position=(10, 10))
        draw_progress_bar(
            display, frame_index, max(total_frames, 1), hex_to_bgr(spec["color_hex"])
        )

        state["last_annotated"] = display.copy()
        state["last_meta"] = {
            "frame_index": frame_index,
            "video": video_path.name,
            "per_model": {
                spec["display"]: {
                    "inference_ms": inference_ms,
                    "detections": count,
                    "max_conf": max_conf,
                    "conf_threshold": conf,
                }
            },
        }
        self._push_display(display)

    # %%
    def _process_split(
        self,
        state: dict[str, Any],
        frame: np.ndarray,
        frame_index: int,
        total_frames: int,
        video_path: Path,
    ) -> None:
        """Split-screen branch: dispatch to active workers, composite output.

        Panel layout adapts to how many model sizes were selected:

        * 1 model  -> single full-canvas panel
        * 2 models -> two panels side by side
        * 3 models -> three panels side by side
        * 4 models -> 2x2 grid (top row panels 0+1, bottom row panels 2+3)

        Args:
            state: The current run state.
            frame: The raw BGR frame.
            frame_index: 1-based position of the frame in the video.
            total_frames: Total frames in the source video.
            video_path: Path to the source video (for snapshot metadata).
        """
        active_backends: dict[str, ModelBackend | None] = state["active_backends"]
        backend_id: str = state["backend"]
        precision_id: str = state["precision"]

        # Resolve panel dimensions for the current layout.
        model_keys: list[str] = list(state["model_keys"])
        panel_w, panel_h = _split_panel_dims(len(model_keys))

        # Submit only to workers whose active backend is loaded for this run.
        for key in model_keys:
            if active_backends.get(key) is None:
                continue
            worker = self.workers[key]
            worker.set_conf(self.live_conf[key])
            worker.drain_output()
            worker.submit(frame.copy())

        panels: list[np.ndarray] = []
        meta_per_model: dict[str, dict[str, Any]] = {}
        for key in model_keys:
            spec = self.models[key]["spec"]
            color_bgr = hex_to_bgr(spec["color_hex"])
            wrapper = active_backends.get(key)

            if wrapper is None:
                panel = self._render_unavailable_panel(
                    spec, backend_id, color_bgr, panel_w, panel_h,
                )
                panels.append(panel)
                meta_per_model[spec["display"]] = {
                    "inference_ms": 0.0,
                    "detections": 0,
                    "max_conf": 0.0,
                    "conf_threshold": float(self.live_conf[key]),
                    "backend": backend_id,
                    "precision": precision_id,
                    "available": False,
                }
                continue

            worker = self.workers[key]
            item = worker.collect(_SPLIT_WORKER_TIMEOUT)
            if item is None:
                panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(
                    panel, "Timeout", (12, 40), _HUD_FONT, 0.8,
                    _BGR_RED, 2, cv2.LINE_AA,
                )
                cv2.rectangle(
                    panel, (0, 0), (panel_w - 1, panel_h - 1), color_bgr, 2
                )
                panels.append(panel)
                meta_per_model[spec["display"]] = {
                    "inference_ms": 0.0,
                    "detections": 0,
                    "max_conf": 0.0,
                    "conf_threshold": float(self.live_conf[key]),
                    "backend": backend_id,
                    "precision": precision_id,
                    "available": True,
                }
                continue

            result, inference_ms = item
            annotated, count, max_conf = apply_overlays(frame, result, spec)
            panel = letterbox(annotated, panel_w, panel_h)

            stats = state["stats"][key]
            stats["frames"] += 1
            stats["total_detections"] += count
            if count > 0:
                stats["frames_with_det"] += 1
            stats["latencies"].append(inference_ms)

            per_frame_fps = (1000.0 / inference_ms) if inference_ms > 0 else 0.0
            panel_title = (
                f"{spec.get('version_display', '')} {spec['size_label']}".strip()
                or spec["display"]
            )
            panel_lines: list[tuple[str, tuple[int, int, int], int]] = [
                (panel_title, color_bgr, 2),
                (
                    f"{per_frame_fps:.1f} FPS | {inference_ms:.1f} ms",
                    _BGR_WHITE,
                    1,
                ),
                (f"det: {count}   conf: {self.live_conf[key]:.2f}", _BGR_WHITE, 1),
            ]
            if not self.models[key]["used_finetuned"]:
                panel_lines.append(("pretrained", _BGR_RED, 1))
            draw_hud_panel(panel, panel_lines, position=(8, 8))
            cv2.rectangle(
                panel, (0, 0), (panel_w - 1, panel_h - 1), color_bgr, 2
            )
            panels.append(panel)
            meta_per_model[spec["display"]] = {
                "inference_ms": inference_ms,
                "detections": count,
                "max_conf": max_conf,
                "conf_threshold": float(self.live_conf[key]),
                "backend": backend_id,
                "precision": precision_id,
                "available": True,
            }

        composite = _composite_split_panels(panels, panel_w, panel_h)
        composite_width = composite.shape[1]

        # Composite FPS banner (bottleneck FPS) at top centre.
        rolling_fps = (
            (len(state["recent_frame_times"]) / sum(state["recent_frame_times"]))
            if state["recent_frame_times"]
            else 0.0
        )
        rolling_ms = (1000.0 / rolling_fps) if rolling_fps > 0 else 0.0
        banner_text = (
            f"{_BACKEND_DISPLAY[backend_id]} {precision_id.upper()} - "
            f"Composite {rolling_fps:.1f} FPS | {rolling_ms:.1f} ms (bottleneck)"
        )
        (text_w, text_h), _ = cv2.getTextSize(
            banner_text, _HUD_FONT, 0.7, 2
        )
        banner_x = (composite_width - text_w) // 2
        overlay = composite.copy()
        cv2.rectangle(
            overlay,
            (banner_x - 12, 2),
            (banner_x + text_w + 12, 2 + text_h + 14),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.6, composite, 0.4, 0, composite)
        cv2.putText(
            composite,
            banner_text,
            (banner_x, 2 + text_h + 8),
            _HUD_FONT,
            0.7,
            _BGR_WHITE,
            2,
            cv2.LINE_AA,
        )

        draw_progress_bar(
            composite, frame_index, max(total_frames, 1), hex_to_bgr(_ORANGE)
        )

        state["last_annotated"] = composite.copy()
        state["last_meta"] = {
            "frame_index": frame_index,
            "video": video_path.name,
            "per_model": meta_per_model,
        }
        self._push_display(composite)

    # %%
    def _render_unavailable_panel(
        self,
        spec: dict[str, Any],
        backend: str,
        color_bgr: tuple[int, int, int],
        panel_w: int = _PANEL_WIDTH,
        panel_h: int = _PANEL_HEIGHT,
    ) -> np.ndarray:
        """Render a split-screen panel for a size whose backend is unloaded.

        Args:
            spec: The model spec for this panel.
            backend: The backend identifier currently in use.
            color_bgr: Border colour for the panel.
            panel_w: Panel width in pixels for the active layout.
            panel_h: Panel height in pixels for the active layout.

        Returns:
            A pre-rendered ``panel_h x panel_w x 3`` BGR panel.
        """
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        title = (
            f"{spec.get('version_display', '')} {spec['size_label']}".strip()
            or spec["display"]
        )
        cv2.putText(
            panel, title, (12, 30), _HUD_FONT, 0.7, color_bgr, 2, cv2.LINE_AA,
        )
        cv2.putText(
            panel, "Not available", (12, 70), _HUD_FONT, 0.8,
            _BGR_WHITE, 2, cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"({_BACKEND_DISPLAY[backend]} weights missing)",
            (12, 100), _HUD_FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )
        cv2.rectangle(
            panel, (0, 0), (panel_w - 1, panel_h - 1), color_bgr, 2
        )
        return panel

    # %%
    def _push_display(self, frame_bgr: np.ndarray) -> None:
        """Replace any pending display frame with the newest one.

        Args:
            frame_bgr: The composited annotated BGR frame to display.
        """
        try:
            self._display_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._display_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    # %%
    def _display_tick(self) -> None:
        """Main-thread display tick: pull from the queue and update the canvas."""
        if self.run_state is None or not self.run_state.get("running"):
            return
        if self.current_screen != "run":
            return
        try:
            frame = self._display_queue.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.canvas.delete("display")
            canvas_width = int(self.canvas["width"])
            canvas_height = int(self.canvas["height"])
            x = max(0, (canvas_width - image.width) // 2)
            y = max(0, (canvas_height - image.height) // 2)
            self.canvas.create_image(x, y, image=photo, anchor=tk.NW, tags=("display",))
            self._canvas_photo = photo
        self.root.after(_DISPLAY_TICK_MS, self._display_tick)

    # %%
    def _show_video_transition(self, remaining: int) -> None:
        """Overlay the run screen with the transition banner.

        Args:
            remaining: Number of videos still queued after this one.
        """
        if self.current_screen != "run":
            return
        if self._transition_overlay is not None:
            self._transition_overlay.destroy()
        overlay = tk.Frame(self.current_frame, bg="black")
        overlay.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.6, relheight=0.2)
        tk.Label(
            overlay,
            text=f"Loading next video... ({remaining} remaining)",
            bg="black",
            fg="white",
            font=("Helvetica", px(16), "bold"),
        ).pack(expand=True)
        self._transition_overlay = overlay

    # %%
    def _hide_video_transition(self) -> None:
        """Tear down the transition banner once the next video starts."""
        if self._transition_overlay is not None:
            self._transition_overlay.destroy()
            self._transition_overlay = None

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------
    # %%
    def _build_run_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        """Convert the in-memory run state into a serialisable summary dict.

        Args:
            state: The current run state.

        Returns:
            A history-entry-compatible summary dict.
        """
        backend_id: str = state.get("backend", _DEFAULT_BACKEND)
        precision_id: str = state.get("precision", _DEFAULT_PRECISION)
        device_id: str = state.get("active_device", self.device)
        cpu_mode: bool = bool(state.get("cpu_mode", False))

        summary: dict[str, Any] = {
            "run_id": state["run_id"],
            "timestamp": state["start_timestamp"],
            "model_list": [
                self.models[key]["spec"]["display"] for key in state["model_keys"]
            ],
            "model_sizes": [
                self.models[key]["spec"]["size_label"] for key in state["model_keys"]
            ],
            "video_list": [path.name for path in state["video_list"]],
            "conf_thresholds": {
                self.models[key]["spec"]["display"]: float(self.live_conf[key])
                for key in state["model_keys"]
            },
            "backend": _BACKEND_DISPLAY[backend_id],
            "precision": precision_id.upper(),
            "device": device_id,
            "cpu_mode": cpu_mode,
            "per_model_stats": {},
            "snapshot_paths": list(state["snapshot_paths"]),
            "duration_seconds": time.perf_counter() - state["start_perf"],
        }
        for key in state["model_keys"]:
            stats = state["stats"][key]
            latencies = (
                np.array(stats["latencies"], dtype=np.float64)
                if stats["latencies"]
                else np.array([0.0])
            )
            frames = int(stats["frames"])
            mean_latency = float(latencies.mean()) if frames > 0 else 0.0
            mean_fps = (1000.0 / mean_latency) if mean_latency > 0 else 0.0
            summary["per_model_stats"][self.models[key]["spec"]["display"]] = {
                "frames": frames,
                "mean_fps": mean_fps,
                "mean_latency_ms": mean_latency,
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "total_detections": int(stats["total_detections"]),
                "detection_rate": (
                    (stats["frames_with_det"] / frames) if frames > 0 else 0.0
                ),
                "snapshots_saved": int(state["snapshots_saved"]),
                "model_size": self.models[key]["spec"]["size_label"],
                "backend": _BACKEND_DISPLAY[backend_id],
                "precision": precision_id.upper(),
                "device": device_id,
            }
        return summary

    # %%
    def _on_run_end(self, summary: dict[str, Any]) -> None:
        """Main-thread callback fired by the video thread when a run ends.

        Args:
            summary: The summary dict built by :meth:`_build_run_summary`.
        """
        self._unbind_run_keys()
        self.video_thread = None
        if not summary["per_model_stats"]:
            # Nothing actually ran; just bounce back to the menu.
            self.run_state = None
            self.show_main_menu()
            return
        self._show_run_summary(summary)

    # %%
    def _show_run_summary(self, summary: dict[str, Any]) -> None:
        """Display the full-screen Run Summary overlay with a countdown.

        Args:
            summary: The run summary dict.
        """
        frame = tk.Frame(self.root, bg=_BG_DARK)

        tk.Label(
            frame,
            text="Run Complete",
            bg=_BG_DARK,
            fg=_ORANGE,
            font=("Helvetica", px(28), "bold"),
        ).pack(pady=(px(20), px(10)))

        # Stats cards per model.
        cards = tk.Frame(frame, bg=_BG_DARK)
        cards.pack(fill=tk.X, padx=px(30), pady=px(10))
        for display_name, stats in summary["per_model_stats"].items():
            self._build_stats_card(cards, display_name, stats)

        # Thumbnail strip (top-8 highest-confidence snapshots).
        strip_label = tk.Label(
            frame,
            text="Top snapshots",
            bg=_BG_DARK,
            fg=_FG_GREY,
            font=("Helvetica", px(12), "italic"),
        )
        strip_label.pack(anchor=tk.W, padx=px(30), pady=(px(15), px(4)))
        strip = tk.Frame(frame, bg=_BG_DARK)
        strip.pack(fill=tk.X, padx=px(30))
        self._summary_thumb_refs = []
        ranked = gather_snapshot_confidences(summary.get("snapshot_paths", []))
        if not ranked:
            tk.Label(
                strip,
                text="(no snapshots saved)",
                bg=_BG_DARK,
                fg=_FG_GREY,
                font=("Helvetica", px(11), "italic"),
            ).pack(side=tk.LEFT)
        for path, confidence in ranked[:8]:
            self._add_summary_thumbnail(strip, path, confidence)

        # Bottom row: return button + countdown.
        bottom = tk.Frame(frame, bg=_BG_DARK)
        bottom.pack(fill=tk.X, padx=px(30), pady=px(20))
        tk.Button(
            bottom,
            text="Return to Menu",
            bg=_ORANGE,
            fg="#FFFFFF",
            activebackground="#FF8A33",
            activeforeground="#FFFFFF",
            font=("Helvetica", px(13), "bold"),
            bd=0,
            padx=px(24),
            pady=px(10),
            command=self._return_to_menu_from_summary,
        ).pack(side=tk.LEFT)
        self._summary_countdown_var.set(
            f"Auto-return in {_SUMMARY_AUTO_RETURN_SECONDS}s..."
        )
        tk.Label(
            bottom,
            textvariable=self._summary_countdown_var,
            bg=_BG_DARK,
            fg=_FG_GREY,
            font=("Helvetica", px(11), "italic"),
        ).pack(side=tk.LEFT, padx=px(20))

        self._swap_frame(frame, "summary")
        self._summary_visible = True
        self.root.after(1000, self._summary_tick, _SUMMARY_AUTO_RETURN_SECONDS - 1)

    # %%
    def _build_stats_card(
        self,
        parent: tk.Frame,
        display_name: str,
        stats: dict[str, Any],
    ) -> None:
        """Build a single per-model stats card inside the summary screen.

        Args:
            parent: Parent container frame.
            display_name: Model name shown as the card title.
            stats: The per-model stats dict.
        """
        spec = next(
            (spec for spec in self.model_specs if spec["display"] == display_name),
            None,
        )
        accent = spec["color_hex"] if spec is not None else _FG_DEFAULT
        card = tk.Frame(parent, bg=_BG_PANEL, padx=px(16), pady=px(12))
        card.pack(side=tk.LEFT, padx=px(8), fill=tk.BOTH, expand=True)
        tk.Label(
            card,
            text=display_name,
            bg=_BG_PANEL,
            fg=accent,
            font=("Helvetica", px(14), "bold"),
        ).pack(anchor=tk.W)
        mean_fps = float(stats.get("mean_fps", 0.0))
        mean_latency = float(stats.get("mean_latency_ms") or 0.0)
        if mean_latency == 0.0 and mean_fps > 0:
            mean_latency = 1000.0 / mean_fps

        rows: list[tuple[str, str]] = []
        if "backend" in stats:
            rows.append(("Backend", str(stats["backend"])))
        if "precision" in stats:
            rows.append(("Precision", str(stats["precision"])))
        if "device" in stats:
            rows.append(("Device", str(stats["device"])))
        rows.extend(
            [
                ("FPS | latency", f"{mean_fps:.1f} | {mean_latency:.1f} ms"),
                ("p50 latency", f"{float(stats.get('p50_latency_ms', 0.0)):.1f} ms"),
                ("p95 latency", f"{float(stats.get('p95_latency_ms', 0.0)):.1f} ms"),
                ("Frames", str(int(stats.get("frames", 0)))),
                ("Detections", str(int(stats.get("total_detections", 0)))),
                ("Detection rate", f"{100.0 * float(stats.get('detection_rate', 0.0)):.1f}%"),
                ("Snapshots saved", str(int(stats.get("snapshots_saved", 0)))),
            ]
        )
        for label, value in rows:
            row = tk.Frame(card, bg=_BG_PANEL)
            row.pack(fill=tk.X, pady=px(1))
            tk.Label(
                row, text=label, bg=_BG_PANEL, fg=_FG_GREY, width=18, anchor=tk.W
            ).pack(side=tk.LEFT)
            tk.Label(row, text=value, bg=_BG_PANEL, fg=_FG_DEFAULT).pack(side=tk.LEFT)

    # %%
    def _add_summary_thumbnail(
        self,
        parent: tk.Frame,
        path: Path,
        confidence: float,
    ) -> None:
        """Add a single thumbnail tile (image + caption) to the summary strip.

        Args:
            parent: The horizontal strip frame.
            path: The snapshot JPEG path.
            confidence: Max-confidence score for the caption.
        """
        try:
            with Image.open(path) as image:
                image.thumbnail((px(160), px(110)))
                photo = ImageTk.PhotoImage(image.copy())
        except (OSError, ValueError):
            return
        self._summary_thumb_refs.append(photo)
        cell = tk.Frame(parent, bg=_BG_PANEL, padx=px(4), pady=px(4))
        cell.pack(side=tk.LEFT, padx=px(4))
        tk.Label(cell, image=photo, bg=_BG_PANEL).pack()
        try:
            frame_index = int(path.stem.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            frame_index = 0
        tk.Label(
            cell,
            text=f"f{frame_index}  conf {confidence:.2f}",
            bg=_BG_PANEL,
            fg=_FG_GREY,
            font=("Helvetica", px(9)),
        ).pack()

    # %%
    def _summary_tick(self, seconds_left: int) -> None:
        """Update the summary-screen countdown and auto-return at zero.

        Args:
            seconds_left: Whole seconds remaining on the countdown.
        """
        if not self._summary_visible or self.current_screen != "summary":
            return
        if seconds_left <= 0:
            self._return_to_menu_from_summary()
            return
        self._summary_countdown_var.set(f"Auto-return in {seconds_left}s...")
        self.root.after(1000, self._summary_tick, seconds_left - 1)

    # %%
    def _return_to_menu_from_summary(self) -> None:
        """Tear down the summary and navigate back to the main menu."""
        if not self._summary_visible:
            return
        self._summary_visible = False
        self._summary_thumb_refs = []
        self.run_state = None
        self.show_main_menu()

    # ------------------------------------------------------------------
    # Benchmark mode (Full Benchmark - all discovered combinations)
    # ------------------------------------------------------------------
    # %%
    def _on_full_benchmark(self) -> None:
        """Full Benchmark click handler: validate one video, then run all combos.

        The benchmark requires *exactly one* checked video so the resulting
        ``benchmark_history.json`` entry has a single canonical ``video_key``.
        Any other selection count surfaces an error dialog and aborts.
        """
        names = self._selected_video_names()
        if len(names) != 1:
            messagebox.showerror(
                "Wrong video count",
                "Please select exactly one video for benchmarking.",
            )
            return
        video_path = _INPUTS_DIR / names[0]
        if not video_path.is_file():
            messagebox.showerror(
                "Missing video", f"{names[0]} is no longer on disk.",
            )
            return
        combos = self._enumerate_full_benchmark_combos()
        if not combos:
            messagebox.showerror(
                "Nothing to benchmark",
                "No model weights were found for any backend. Drop "
                "finetuned files into models/ and click Refresh.",
            )
            return
        self._open_benchmark_dialog(video_path, combos)

    # %%
    def _enumerate_full_benchmark_combos(
        self,
    ) -> list[tuple[str, str, str]]:
        """Return every valid ``(model_key, backend, precision)`` to benchmark.

        Rules:

        * PyTorch FP32: always run if ``.pt`` exists.
        * PyTorch FP16: run if ``.pt`` exists, CUDA is available, *and*
          the CPU-only switch is off (FP16 on CPU is not a meaningful
          configuration so we skip it).
        * ONNX FP32: run if ``.onnx`` exists.
        * ONNX FP16: skipped (not supported by the ORT path here).
        * TensorRT FP16: run if ``.engine`` exists, CUDA is available,
          *and* CPU-only is off (engines need CUDA).
        * TensorRT INT8: run if ``*_crack_int8.engine`` exists under the same
          conditions. Absent INT8 engines simply drop the combo, so the full
          benchmark still runs on a machine that has not exported them.
        * TensorRT FP32: skipped (engines are exported FP16 or INT8).

        Returns:
            Combos in (version, size, backend, precision) discovery order.
        """
        combos: list[tuple[str, str, str]] = []
        cuda_available = torch.cuda.is_available()
        cpu_mode = bool(self.cpu_mode_var.get())
        gpu_enabled = cuda_available and not cpu_mode
        for spec in self.model_specs:
            key = spec["key"]
            has_pt = self.lookup_weights(key, _BACKEND_PYTORCH) is not None
            has_onnx = self.lookup_weights(key, _BACKEND_ONNX) is not None
            has_engine = self.lookup_weights(key, _BACKEND_TENSORRT) is not None
            has_int8 = self.lookup_weights(key, _BACKEND_TENSORRT_INT8) is not None
            if has_pt:
                combos.append((key, _BACKEND_PYTORCH, _PRECISION_FP32))
                if gpu_enabled:
                    combos.append((key, _BACKEND_PYTORCH, _PRECISION_FP16))
            if has_onnx:
                combos.append((key, _BACKEND_ONNX, _PRECISION_FP32))
            if has_engine and gpu_enabled:
                combos.append((key, _BACKEND_TENSORRT, _PRECISION_FP16))
            if has_int8 and gpu_enabled:
                combos.append((key, _BACKEND_TENSORRT, _PRECISION_INT8))
        return combos

    # %%
    def _open_benchmark_dialog(
        self,
        video_path: Path,
        combos: list[tuple[str, str, str]],
    ) -> None:
        """Build the "Full Benchmark" modal and kick off the runner thread.

        Args:
            video_path: The single source video the user chose.
            combos: Pre-enumerated ``(model_key, backend, precision)`` tuples
                that should be executed in order.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Full Benchmark")
        dialog.configure(bg=_BG_DARK)
        dialog.geometry(f"{px(960)}x{px(680)}")
        dialog.transient(self.root)
        dialog.grab_set()

        cancel_event = threading.Event()

        def _on_close_attempt() -> None:
            """Intercept the OS close button - require the Cancel button."""
            cancel_event.set()

        dialog.protocol("WM_DELETE_WINDOW", _on_close_attempt)

        # Header
        header = tk.Frame(dialog, bg=_BG_DARK)
        header.pack(fill=tk.X, padx=px(20), pady=(px(15), px(2)))
        tk.Label(
            header,
            text="Full Benchmark",
            bg=_BG_DARK,
            fg=_ORANGE,
            font=FONT_MODAL_TITLE,
        ).pack(side=tk.LEFT)

        video_label = tk.Label(
            dialog,
            text=f"Video: {video_path.name}",
            bg=_BG_DARK,
            fg=ACCENT_ORANGE,
            font=FONT_WIDGET,
            anchor=tk.W,
        )
        video_label.pack(fill=tk.X, padx=px(20), pady=(0, px(6)))

        # Progress bar
        progress = ttk.Progressbar(
            dialog, mode="determinate",
            maximum=max(1, len(combos)), value=0,
        )
        progress.pack(fill=tk.X, padx=px(20), pady=(px(5), px(2)))
        progress_label = tk.Label(
            dialog, text=f"0 / {len(combos)} combinations",
            bg=_BG_DARK, fg=_FG_GREY,
            font=FONT_HUD,
            anchor=tk.W,
        )
        progress_label.pack(fill=tk.X, padx=px(20))
        combo_label = tk.Label(
            dialog, text="",
            bg=_BG_DARK, fg=ACCENT_ORANGE,
            font=FONT_HUD,
            anchor=tk.W,
        )
        combo_label.pack(fill=tk.X, padx=px(20), pady=(0, px(4)))

        # Log Text widget
        log_frame = tk.Frame(dialog, bg=_BG_DARK)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=px(20), pady=px(10))
        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        log_text = tk.Text(
            log_frame,
            bg=_BG_PANEL,
            fg=_FG_DEFAULT,
            font=FONT_HUD,
            height=12,
            yscrollcommand=log_scroll.set,
            relief=tk.FLAT,
            wrap=tk.NONE,
        )
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.config(command=log_text.yview)
        log_text.config(state=tk.DISABLED)

        # Bottom buttons (Cancel only while running; replaced on completion)
        button_row = tk.Frame(dialog, bg=_BG_DARK)
        button_row.pack(fill=tk.X, padx=px(20), pady=(0, px(15)))
        cancel_button = tk.Button(
            button_row,
            text="Cancel",
            bg=_DARK_RED,
            fg="#FFFFFF",
            activebackground="#9B3422",
            activeforeground="#FFFFFF",
            bd=0,
            padx=px(20),
            pady=px(12),
            font=FONT_BUTTON,
            command=cancel_event.set,
        )
        cancel_button.pack(side=tk.LEFT, ipady=4)

        # Capture the CPU-only switch on the main thread so the runner
        # thread doesn't have to touch the tk var.
        cpu_mode_at_start = bool(self.cpu_mode_var.get())

        # State shared between threads (mutation only on UI thread).
        dialog_state: dict[str, Any] = {
            "dialog": dialog,
            "progress": progress,
            "progress_label": progress_label,
            "combo_label": combo_label,
            "video_label": video_label,
            "log_text": log_text,
            "button_row": button_row,
            "cancel_button": cancel_button,
            "results": [],
            "combos": combos,
            "video_path": video_path,
            "video_key": video_path.name,
            "cpu_mode": cpu_mode_at_start,
        }

        # Launch the runner thread.
        runner = threading.Thread(
            target=self._benchmark_runner_main,
            args=(dialog_state, [video_path], cancel_event),
            name="benchmark-runner",
            daemon=True,
        )
        runner.start()

    # %%
    def _benchmark_runner_main(
        self,
        dialog_state: dict[str, Any],
        video_paths: list[Path],
        cancel_event: threading.Event,
    ) -> None:
        """Background thread: run every full-benchmark combo and stream output.

        Args:
            dialog_state: Shared state dict for cross-thread updates. Carries
                the ``combos`` list and the chosen ``video_path``.
            video_paths: Source videos to sample frames from (single-element
                list - kept as a list so existing helpers stay unchanged).
            cancel_event: Set by the Cancel button to stop the run early.
        """
        combos: list[tuple[str, str, str]] = dialog_state["combos"]
        cpu_mode: bool = bool(dialog_state.get("cpu_mode", False))
        results: list[BenchmarkResult] = []
        for index, (size_key, backend, precision) in enumerate(combos, start=1):
            if cancel_event.is_set():
                self.root.after(
                    0, self._benchmark_log, dialog_state, "[cancel] aborted",
                )
                break
            spec = self.model_by_key[size_key]
            # TensorRT INT8 resolves to its own engine file, not the FP16 one.
            weights_path = (
                self.lookup_weights(size_key, backend_storage_key(backend, precision))
                or _MODELS_DIR / "MISSING.pt"
            )
            subtitle = (
                f"{size_key} - {_BACKEND_DISPLAY[backend]} {precision.upper()}"
            )
            self.root.after(
                0, self._benchmark_update_combo, dialog_state, subtitle,
            )
            self.root.after(
                0,
                self._benchmark_log,
                dialog_state,
                f"=== {index}/{len(combos)}: {spec['display']} | "
                f"{_BACKEND_DISPLAY[backend]} {precision.upper()} ===",
            )

            def log_fn(message: str) -> None:
                """Thread-safe log line - bounces through the tk main loop."""
                self.root.after(0, self._benchmark_log, dialog_state, message)

            result = benchmark_one_combo(
                spec=spec,
                backend=backend,
                precision=precision,
                weights_path=weights_path,
                video_paths=video_paths,
                frame_count=self.benchmark_frame_count,
                warmup_frames=self.benchmark_warmup_frames,
                cpu_mode=cpu_mode,
                log_fn=log_fn,
                cancel_event=cancel_event,
            )
            if result is not None:
                results.append(result)
            self.root.after(0, self._benchmark_advance, dialog_state, index)

        # Persist + present results.
        try:
            append_benchmark_csv(self.benchmark_csv_path, results)
        except OSError as exc:
            self.root.after(
                0,
                self._benchmark_log,
                dialog_state,
                f"[warn] could not write CSV: {exc}",
            )
        try:
            entry = self._build_benchmark_history_entry(
                results, video_paths, combos,
            )
            self._benchmark_history_upsert(entry)
        except OSError as exc:
            self.root.after(
                0,
                self._benchmark_log,
                dialog_state,
                f"[warn] could not write benchmark_history.json: {exc}",
            )
        dialog_state["results"] = results
        dialog_state["history_entry"] = entry
        self.root.after(0, self._benchmark_show_results, dialog_state)

    # %%
    def _build_benchmark_history_entry(
        self,
        results: list[BenchmarkResult],
        video_paths: list[Path],
        combos_requested: list[tuple[str, str, str]],
    ) -> dict[str, Any]:
        """Compose one ``benchmark_history.json`` entry from a Full Benchmark run.

        Successful combos contribute real numbers; combos that were requested
        but produced no result (the weights existed at enumeration time but
        ``benchmark_one_combo`` failed or was cancelled) are still recorded
        with ``mean_fps=None`` so the chart can show them as "N/A" bars.

        Args:
            results: Per-combo benchmark rows produced by the runner.
            video_paths: Source videos sampled during the run.
            combos_requested: The ``(model_key, backend, precision)`` triples
                originally enumerated for the run.

        Returns:
            A history-entry dict that matches the documented schema.
        """
        # Index real results so the requested-but-missing pass can find them.
        results_by_combo: dict[tuple[str, str, str], BenchmarkResult] = {
            (row.model_key, row.backend, row.precision): row for row in results
        }
        combos: list[dict[str, Any]] = []
        versions_seen: set[str] = set()
        backends_seen: set[str] = set()
        precisions_seen: set[str] = set()
        for model_key, backend, precision in combos_requested:
            spec = self.model_by_key.get(model_key, {})
            version = spec.get("version", "")
            if version:
                versions_seen.add(version)
            backends_seen.add(backend)
            precisions_seen.add(precision)
            row = results_by_combo.get((model_key, backend, precision))
            if row is not None:
                combos.append({
                    "model_key": model_key,
                    "size": spec.get("size", row.size_label.lower()),
                    "backend": backend,
                    "precision": precision,
                    "device": row.device,
                    "mean_fps": float(row.mean_fps),
                    "p50_ms": float(row.p50_ms),
                    "p95_ms": float(row.p95_ms),
                    "p99_ms": float(row.p99_ms),
                    "cold_start_ms": float(row.cold_start_ms),
                    "frame_count": int(row.frames),
                    "kernel_fps": float(row.kernel_mean_fps),
                    "kernel_p50_ms": float(row.kernel_p50_ms),
                    "kernel_p95_ms": float(row.kernel_p95_ms),
                    "kernel_frame_count": int(row.kernel_frames),
                    "active_provider": row.active_provider,
                })
            else:
                combos.append({
                    "model_key": model_key,
                    "size": spec.get("size", ""),
                    "backend": backend,
                    "precision": precision,
                    "device": "",
                    "mean_fps": None,
                    "p50_ms": None,
                    "p95_ms": None,
                    "p99_ms": None,
                    "cold_start_ms": None,
                    "frame_count": 0,
                    "kernel_fps": None,
                    "kernel_p50_ms": None,
                    "kernel_p95_ms": None,
                    "kernel_frame_count": 0,
                    "active_provider": "N/A",
                })
        if not versions_seen:
            versions_seen.add(self.version_var.get())
        return {
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "yolo_versions": sorted(versions_seen),
            # ``yolo_version`` retained for backward compatibility with the
            # previous schema; consumers should prefer ``yolo_versions``.
            "yolo_version": sorted(versions_seen)[0],
            "backends": sorted(backends_seen),
            "precisions": sorted(precisions_seen),
            "combos": combos,
            "video_files": [path.name for path in video_paths],
            "video_key": video_paths[0].name if video_paths else "",
            "notes": "",
        }

    # %%
    def _benchmark_history_upsert(self, entry: dict[str, Any]) -> None:
        """Insert ``entry`` or replace any existing entry with the same video.

        Lookup key is ``video_files == [video_key]`` so re-running the
        benchmark on the same video refreshes its row in place rather than
        accumulating duplicates. Position in the list is preserved on
        replace.

        Args:
            entry: A history-entry dict produced by
                :meth:`_build_benchmark_history_entry`.
        """
        video_key = entry.get("video_key", "")
        entries = self.benchmark_history.load()
        for index, existing in enumerate(entries):
            if existing.get("video_files") == [video_key]:
                # Preserve insertion order; replace in place.
                entries[index] = entry
                self.benchmark_history.save_all(entries)
                return
        # Not found - prepend so the newest run lands at the top.
        entries.insert(0, entry)
        self.benchmark_history.save_all(entries)

    # %%
    def _benchmark_update_combo(
        self, dialog_state: dict[str, Any], subtitle: str,
    ) -> None:
        """Main-thread helper: update the modal's current-combo subtitle.

        Args:
            dialog_state: Shared state dict.
            subtitle: ``"{model_key} - {backend} {precision}"`` line.
        """
        label: tk.Label | None = dialog_state.get("combo_label")
        if label is None or not label.winfo_exists():
            return
        label.config(text=subtitle)

    # %%
    def _benchmark_log(
        self,
        dialog_state: dict[str, Any],
        message: str,
    ) -> None:
        """Append a log line to the dialog's text widget (main-thread only).

        Args:
            dialog_state: Shared state dict.
            message: The log line to append.
        """
        log_text: tk.Text = dialog_state["log_text"]
        if not log_text.winfo_exists():
            return
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, message + "\n")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)

    # %%
    def _benchmark_advance(
        self,
        dialog_state: dict[str, Any],
        completed: int,
    ) -> None:
        """Advance the progress bar to ``completed`` combinations.

        Args:
            dialog_state: Shared state dict.
            completed: Number of completed combinations.
        """
        progress: ttk.Progressbar = dialog_state["progress"]
        label: tk.Label = dialog_state["progress_label"]
        if not progress.winfo_exists():
            return
        progress["value"] = completed
        total = len(dialog_state["combos"])
        label.config(text=f"{completed} / {total} combinations")

    # %%
    def _benchmark_show_results(
        self,
        dialog_state: dict[str, Any],
    ) -> None:
        """Replace the dialog body with the chart + results table + buttons.

        Args:
            dialog_state: Shared state dict. Carries the freshly-built
                history entry under ``"history_entry"`` so the chart can
                use the same data the JSON store now holds.
        """
        dialog: tk.Toplevel = dialog_state["dialog"]
        if not dialog.winfo_exists():
            return
        dialog.title("Full Benchmark - Complete")

        # Tear down the running-state widgets so the dialog can host results.
        log_text: tk.Text = dialog_state["log_text"]
        log_text.master.destroy()
        progress: ttk.Progressbar = dialog_state["progress"]
        progress.destroy()
        progress_label: tk.Label = dialog_state["progress_label"]
        progress_label.destroy()
        combo_label: tk.Label | None = dialog_state.get("combo_label")
        if combo_label is not None and combo_label.winfo_exists():
            combo_label.destroy()
        button_row: tk.Frame = dialog_state["button_row"]
        button_row.destroy()

        results: list[BenchmarkResult] = dialog_state["results"]
        history_entry: dict[str, Any] | None = dialog_state.get("history_entry")

        # Chart at the top - dominates the result view since it summarises
        # the entire run at a glance.
        if history_entry is not None:
            chart_frame = tk.Frame(dialog, bg=_BG_DARK, height=px(360))
            chart_frame.pack(fill=tk.BOTH, expand=True, padx=px(20), pady=px(8))
            chart_frame.pack_propagate(False)
            canvas, figure = self._embed_full_benchmark_chart(
                chart_frame, [history_entry],
            )
            dialog_state["chart_canvas"] = canvas
            dialog_state["chart_figure"] = figure
            png_path = self._save_full_benchmark_chart_png(history_entry)
            if png_path is not None:
                # The log widget was just destroyed - emit to stdout instead
                # so users still get a record of where the PNG landed.
                print(f"[chart] saved {png_path}")

        if not results:
            tk.Label(
                dialog,
                text="No results produced (every combination failed or was cancelled).",
                bg=_BG_DARK,
                fg=_FG_GREY,
                font=FONT_HUD,
                pady=px(30),
            ).pack(fill=tk.X)
        else:
            self._build_benchmark_results_table(dialog, results)

        actions = tk.Frame(dialog, bg=_BG_DARK)
        actions.pack(fill=tk.X, padx=px(20), pady=(0, px(15)))
        export_button = tk.Button(
            actions,
            text="Export CSV",
            bg=_ORANGE,
            fg="#FFFFFF",
            activebackground="#FF8A33",
            activeforeground="#FFFFFF",
            bd=0,
            padx=px(18),
            pady=px(12),
            font=FONT_BUTTON,
            command=functools.partial(self._benchmark_export_csv, results),
        )
        export_button.pack(side=tk.LEFT, ipady=4)
        if not results:
            export_button.config(state=tk.DISABLED)
        tk.Button(
            actions,
            text="Close",
            bg=_BG_HOVER,
            fg=_FG_DEFAULT,
            activebackground="#444444",
            activeforeground=_FG_DEFAULT,
            bd=0,
            padx=px(18),
            pady=px(12),
            font=FONT_BUTTON,
            command=functools.partial(self._close_benchmark_dialog, dialog_state),
        ).pack(side=tk.RIGHT, ipady=4)
        dialog.protocol(
            "WM_DELETE_WINDOW",
            functools.partial(self._close_benchmark_dialog, dialog_state),
        )

    # %%
    def _close_benchmark_dialog(
        self, dialog_state: dict[str, Any],
    ) -> None:
        """Tear down the benchmark dialog and release its matplotlib figure.

        Args:
            dialog_state: Shared state dict.
        """
        figure = dialog_state.get("chart_figure")
        if figure is not None:
            try:
                plt.close(figure)
            except Exception:  # noqa: BLE001 - never block dialog close
                pass
            dialog_state["chart_figure"] = None
        canvas = dialog_state.get("chart_canvas")
        if canvas is not None:
            try:
                canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass
            dialog_state["chart_canvas"] = None
        dialog: tk.Toplevel = dialog_state["dialog"]
        try:
            dialog.destroy()
        except tk.TclError:
            pass

    # %%
    def _build_benchmark_results_table(
        self,
        parent: tk.Widget,
        results: list[BenchmarkResult],
    ) -> None:
        """Build the results Treeview inside the completed benchmark dialog.

        Args:
            parent: Container widget (the dialog).
            results: Benchmark rows to render.
        """
        wrapper = tk.Frame(parent, bg=_BG_DARK)
        wrapper.pack(fill=tk.BOTH, expand=True, padx=px(20), pady=px(10))
        scroll = tk.Scrollbar(wrapper)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        columns = (
            "model", "size", "backend", "precision", "device",
            "fps", "p50", "p95", "p99", "cold", "notes",
        )
        headings = {
            "model": "Model", "size": "Size", "backend": "Backend",
            "precision": "Precision", "device": "Device",
            "fps": "Mean FPS", "p50": "p50 ms", "p95": "p95 ms",
            "p99": "p99 ms", "cold": "Cold-start ms",
            "notes": "Notes",
        }
        widths = {
            "model": px(130), "size": px(70), "backend": px(80),
            "precision": px(70), "device": px(60),
            "fps": px(80), "p50": px(70), "p95": px(70),
            "p99": px(70), "cold": px(100),
            "notes": px(240),
        }
        tree = ttk.Treeview(
            wrapper, columns=columns, show="headings",
            height=8, yscrollcommand=scroll.set,
        )
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=tree.yview)
        # Build the per-row Notes annotation: flag PyTorch FP16 rows whose
        # mean FPS is <= the matching FP32 row, which indicates Tensor Cores
        # are underutilised at batch=1.
        notes_by_row = self._compute_benchmark_notes(results)
        for index, row in enumerate(results):
            tree.insert(
                "", tk.END,
                values=(
                    row.model_key,
                    row.size_label,
                    _BACKEND_DISPLAY[row.backend],
                    row.precision.upper(),
                    row.device,
                    f"{row.mean_fps:.1f}",
                    f"{row.p50_ms:.1f}",
                    f"{row.p95_ms:.1f}",
                    f"{row.p99_ms:.1f}",
                    f"{row.cold_start_ms:.1f}",
                    notes_by_row[index],
                ),
            )

    # %%
    def _compute_benchmark_notes(
        self,
        results: list["BenchmarkResult"],
    ) -> list[str]:
        """Return the per-row "Notes" string for the results table.

        A row is flagged ``"Batch=1 - Tensor Core underutilised"`` when it is
        a PyTorch FP16 result whose mean FPS is less than or equal to the
        matching PyTorch FP32 result for the same model. Every other row
        gets an empty string so the column stays uncluttered.

        Args:
            results: The full results list from the runner.

        Returns:
            One note string per input row, aligned to ``results`` order.
        """
        fp32_fps_by_model: dict[str, float] = {}
        for row in results:
            if (
                row.backend == _BACKEND_PYTORCH
                and row.precision == _PRECISION_FP32
            ):
                fp32_fps_by_model[row.model_key] = float(row.mean_fps)
        notes: list[str] = []
        for row in results:
            note = ""
            if (
                row.backend == _BACKEND_PYTORCH
                and row.precision == _PRECISION_FP16
            ):
                fp32_fps = fp32_fps_by_model.get(row.model_key)
                if fp32_fps is not None and float(row.mean_fps) <= fp32_fps:
                    note = "Batch=1 - Tensor Core underutilised"
            notes.append(note)
        return notes

    # %%
    def _benchmark_export_csv(
        self,
        results: list[BenchmarkResult],
    ) -> None:
        """Save the session's benchmark rows to a user-chosen CSV file.

        Args:
            results: The rows to export.
        """
        if not results:
            return
        destination = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="benchmark_session.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export benchmark results",
        )
        if not destination:
            return
        try:
            append_benchmark_csv(Path(destination), results)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Export complete", f"Wrote {len(results)} rows to:\n{destination}")

    # ------------------------------------------------------------------
    # Benchmarks screen
    # ------------------------------------------------------------------
    # %%
    def show_benchmarks_screen(self) -> None:
        """Render the Benchmark History screen with a list, chart and detail.

        Left panel (40%): a multi-select ``ttk.Treeview`` listing every saved
        benchmark run from ``benchmark_history.json``.
        Right panel (60%): a matplotlib chart (FPS per model grouped by
        backend, or compared across runs) plus a per-combo detail treeview.
        Top bar: Back / Export CSV / Compare Selected.
        """
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Benchmarks.Treeview",
            background=BG_CARD,
            foreground=TEXT_PRIMARY,
            fieldbackground=BG_CARD,
            rowheight=px(26),
            bordercolor=BORDER,
            borderwidth=0,
        )
        style.configure(
            "Benchmarks.Treeview.Heading",
            background=BG_SECONDARY,
            foreground=ACCENT_ORANGE,
            relief="flat",
        )
        style.map(
            "Benchmarks.Treeview",
            background=[("selected", ACCENT_ORANGE)],
            foreground=[("selected", BG_PRIMARY)],
        )

        frame = tk.Frame(self.root, bg=_BG_DARK)
        # ---- top bar ------------------------------------------------------
        top_bar = tk.Frame(frame, bg=_BG_DARK)
        top_bar.pack(fill=tk.X, padx=px(20), pady=(px(16), px(8)))
        tk.Button(
            top_bar, text="←  Back",
            bg=_BG_HOVER, fg=_FG_DEFAULT, activebackground="#3A3A3A",
            activeforeground=_FG_DEFAULT, bd=0,
            padx=px(14), pady=px(6),
            font=FONT_BUTTON,
            command=self.show_main_menu,
        ).pack(side=tk.LEFT)
        tk.Label(
            top_bar, text="Benchmark History",
            bg=_BG_DARK, fg=_FG_DEFAULT, font=FONT_MODAL_TITLE,
        ).pack(side=tk.LEFT, padx=px(16))
        compare_button = tk.Button(
            top_bar, text="Compare Selected",
            bg=ACCENT_TEAL, fg=BG_PRIMARY, activebackground="#1ABC9C",
            activeforeground=BG_PRIMARY, bd=0,
            padx=px(14), pady=px(6),
            font=FONT_BUTTON,
            state=tk.DISABLED,
        )
        compare_button.pack(side=tk.RIGHT)
        export_button = tk.Button(
            top_bar, text="Export CSV",
            bg=_ORANGE, fg="#FFFFFF", activebackground="#FF8A33",
            activeforeground="#FFFFFF", bd=0,
            padx=px(14), pady=px(6),
            font=FONT_BUTTON,
            state=tk.DISABLED,
        )
        export_button.pack(side=tk.RIGHT, padx=(0, px(8)))

        # Dark scrollbar style applied to both treeviews and the chart's
        # horizontal scrollbar so the whole screen reads as a single dark
        # theme rather than mixing the OS-native grey scrollbar in.
        style.configure(
            "BenchDark.Vertical.TScrollbar",
            troughcolor=BG_CARD,
            background=BORDER,
            bordercolor=BG_CARD,
            arrowcolor=TEXT_MUTED,
        )
        style.configure(
            "BenchDark.Horizontal.TScrollbar",
            troughcolor=BG_CARD,
            background=BORDER,
            bordercolor=BG_CARD,
            arrowcolor=TEXT_MUTED,
        )
        style.map(
            "BenchDark.Vertical.TScrollbar",
            background=[("active", "#3A3A3A")],
        )
        style.map(
            "BenchDark.Horizontal.TScrollbar",
            background=[("active", "#3A3A3A")],
        )

        # ---- body: 30% left column / 70% right column ----------------
        body = tk.Frame(frame, bg=_BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=px(20), pady=(0, px(16)))
        body.grid_columnconfigure(0, weight=30, uniform="bench_cols")
        body.grid_columnconfigure(1, weight=70, uniform="bench_cols")
        body.grid_rowconfigure(0, weight=1)

        # ---- left column: run list (15%) + divider + detail (85%) ----
        left_column = tk.Frame(body, bg=_BG_DARK)
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, px(8)))
        left_column.grid_columnconfigure(0, weight=1)
        left_column.grid_rowconfigure(0, weight=15, uniform="bench_rows")
        left_column.grid_rowconfigure(1, weight=0)
        left_column.grid_rowconfigure(2, weight=85, uniform="bench_rows")

        # Run list panel
        run_panel = tk.Frame(left_column, bg=_BG_DARK)
        run_panel.grid(row=0, column=0, sticky="nsew")
        run_panel.grid_columnconfigure(0, weight=1)
        run_panel.grid_rowconfigure(0, weight=1)

        columns = (
            "checked", "date", "version", "video", "combos", "best_fps",
        )
        run_tree = ttk.Treeview(
            run_panel, columns=columns, show="headings",
            style="Benchmarks.Treeview", selectmode="browse",
        )
        run_tree.heading("checked", text="✓")
        run_tree.heading("date", text="Date")
        run_tree.heading("version", text="Version(s)")
        run_tree.heading("video", text="Video")
        run_tree.heading("combos", text="Backend combos")
        run_tree.heading("best_fps", text="Best FPS")
        run_tree.column("checked", width=px(36), anchor=tk.CENTER, stretch=False)
        run_tree.column("date", width=px(130), anchor=tk.W)
        run_tree.column("version", width=px(110), anchor=tk.W)
        run_tree.column("video", width=px(170), anchor=tk.W)
        run_tree.column("combos", width=px(200), anchor=tk.W)
        run_tree.column("best_fps", width=px(80), anchor=tk.E)
        run_tree.tag_configure("selected_row", background="#1E1E1E")
        run_tree.grid(row=0, column=0, sticky="nsew")
        run_v_scroll = ttk.Scrollbar(
            run_panel, orient=tk.VERTICAL, command=run_tree.yview,
            style="BenchDark.Vertical.TScrollbar",
        )
        run_v_scroll.grid(row=0, column=1, sticky="ns")
        run_tree.configure(yscrollcommand=run_v_scroll.set)

        # 1px divider between the two left-column sections.
        divider = tk.Frame(left_column, bg="#2A2A2A", height=1)
        divider.grid(row=1, column=0, sticky="ew", pady=px(4))

        # Detail-stats panel
        detail_panel = tk.Frame(left_column, bg=_BG_DARK)
        detail_panel.grid(row=2, column=0, sticky="nsew")
        detail_panel.grid_columnconfigure(0, weight=1)
        detail_panel.grid_rowconfigure(0, weight=1)

        detail_table_columns = (
            "model_key", "size", "backend", "precision", "device",
            "provider",
            "mean_fps", "p50_ms", "p95_ms", "p99_ms",
            "cold_start_ms", "frame_count",
            "kernel_fps", "kernel_p50_ms",
        )
        detail_tree = ttk.Treeview(
            detail_panel, columns=detail_table_columns, show="headings",
            style="Benchmarks.Treeview",
        )
        for col in detail_table_columns:
            detail_tree.heading(col, text=col)
            detail_tree.column(col, width=px(88), anchor=tk.W)
        detail_tree.column("provider", width=px(150), anchor=tk.W)
        detail_tree.grid(row=0, column=0, sticky="nsew")
        detail_v_scroll = ttk.Scrollbar(
            detail_panel, orient=tk.VERTICAL, command=detail_tree.yview,
            style="BenchDark.Vertical.TScrollbar",
        )
        detail_v_scroll.grid(row=0, column=1, sticky="ns")
        detail_h_scroll = ttk.Scrollbar(
            detail_panel, orient=tk.HORIZONTAL, command=detail_tree.xview,
            style="BenchDark.Horizontal.TScrollbar",
        )
        detail_h_scroll.grid(row=1, column=0, columnspan=2, sticky="ew")
        detail_tree.configure(
            yscrollcommand=detail_v_scroll.set,
            xscrollcommand=detail_h_scroll.set,
        )

        # ---- right column: video header + toggle + chart + footnote --
        right_column = tk.Frame(body, bg=_BG_DARK)
        right_column.grid(row=0, column=1, sticky="nsew", padx=(px(8), 0))

        video_header = tk.Label(
            right_column, text="", bg=_BG_DARK, fg=ACCENT_ORANGE,
            font=FONT_WIDGET, anchor=tk.W,
        )
        video_header.pack(side=tk.TOP, fill=tk.X)

        # Pipeline / Kernel metric toggle.
        metric_var: tk.StringVar = tk.StringVar(value="pipeline")
        toggle_row = ctk.CTkFrame(right_column, fg_color="transparent")
        toggle_row.pack(side=tk.TOP, fill=tk.X, pady=(px(4), 0))
        metric_segment = ctk.CTkSegmentedButton(
            toggle_row,
            values=["Pipeline FPS", "Kernel FPS"],
            fg_color=BG_HOVER,
            selected_color=ACCENT_ORANGE,
            unselected_color=BG_SECONDARY,
            text_color=TEXT_PRIMARY,
            command=self._on_chart_metric_changed,
            font=FONT_WIDGET,
        )
        metric_segment.set("Pipeline FPS")
        metric_segment.pack(side=tk.LEFT, padx=(0, px(8)))

        # Footnote pinned to the very bottom so it never steals chart
        # vertical space. Truncate with ellipsis if too wide (no wrap).
        footnote = ctk.CTkLabel(
            right_column,
            text=(
                "Raw ONNX/TRT = engine forward.  PyTorch (Pipeline) = "
                "full Ultralytics wrapper.  Kernel toggle = engine-only "
                "for PyTorch (raw ONNX/TRT unchanged)."
            ),
            text_color=TEXT_MUTED,
            font=FONT_SMALL,
            anchor="w",
            justify="left",
        )
        footnote.pack(side=tk.BOTTOM, fill=tk.X, pady=(px(2), 0))

        # Chart container fills everything between the toggle and the
        # footnote. The scrollable canvas is built inside it by
        # ``_draw_full_benchmark_chart`` whenever the selection changes.
        chart_frame = tk.Frame(right_column, bg=_BG_DARK)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(px(4), 0))

        # ---- state -------------------------------------------------------
        # Newest-first ordering: ``benchmark_history.add`` already prepends
        # new entries; for legacy entries we re-sort by timestamp to keep
        # the screen consistent.
        entries = self.benchmark_history.load()
        entries.sort(
            key=lambda entry: str(entry.get("timestamp", "")),
            reverse=True,
        )
        bench_state: dict[str, Any] = {
            "entries": entries,
            "checked_runs": set(),
            "iid_to_run": {},
            "run_tree": run_tree,
            "detail_tree": detail_tree,
            "chart_frame": chart_frame,
            "video_header": video_header,
            "compare_button": compare_button,
            "export_button": export_button,
            "chart_canvas": None,
            "chart_figure": None,
            "metric_var": metric_var,
            "metric_segment": metric_segment,
            "current_runs": [],
        }
        # Stash on the App so ``_on_chart_metric_changed`` can find it.
        self._benchmarks_screen_state = bench_state
        self._populate_benchmarks_tree(bench_state)

        def _refresh_buttons() -> None:
            """Enable/disable Compare + Export buttons based on selection."""
            has_one = bool(run_tree.selection() or bench_state["checked_runs"])
            checked_n = len(bench_state["checked_runs"])
            export_button.configure(state=tk.NORMAL if has_one else tk.DISABLED)
            compare_button.configure(
                state=tk.NORMAL if checked_n >= 2 else tk.DISABLED,
            )

        def _apply_row_highlight() -> None:
            """Reapply the selected-row background based on current selection."""
            selected = set(run_tree.selection())
            for iid in run_tree.get_children():
                if iid in selected:
                    run_tree.item(iid, tags=("selected_row",))
                else:
                    run_tree.item(iid, tags=())

        def _on_tree_click(event: tk.Event) -> None:
            """Treeview click: toggle checkmark or update detail for the row."""
            region = run_tree.identify("region", event.x, event.y)
            column = run_tree.identify_column(event.x)
            item = run_tree.identify_row(event.y)
            if not item:
                return
            if region == "cell" and column == "#1":
                if item in bench_state["checked_runs"]:
                    bench_state["checked_runs"].discard(item)
                    run_tree.set(item, "checked", "")
                else:
                    bench_state["checked_runs"].add(item)
                    run_tree.set(item, "checked", "✓")
                _refresh_buttons()
                return
            run = bench_state["iid_to_run"].get(item)
            if run is not None:
                self._draw_full_benchmark_chart(bench_state, [run])
                self._populate_benchmark_detail(detail_tree, [run])
                _apply_row_highlight()
                _refresh_buttons()

        def _on_compare() -> None:
            """Compare button: chart every checked run side-by-side."""
            checked_runs = [
                bench_state["iid_to_run"][iid]
                for iid in bench_state["checked_runs"]
                if iid in bench_state["iid_to_run"]
            ]
            if len(checked_runs) < 2:
                return
            self._draw_full_benchmark_chart(bench_state, checked_runs)
            self._populate_benchmark_detail(detail_tree, checked_runs)

        def _on_export() -> None:
            """Export button: write the selected combos to a user-chosen CSV."""
            runs = self._benchmark_export_targets(bench_state)
            if not runs:
                return
            destination = filedialog.asksaveasfilename(
                title="Export benchmark history",
                defaultextension=".csv",
                filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
                initialdir=str(_EXPORTS_DIR),
                initialfile="benchmark_history_export.csv",
            )
            if not destination:
                return
            self._export_benchmark_history_csv(Path(destination), runs)

        def _on_tree_select(_event: tk.Event) -> None:
            """ttk select callback: highlight + refresh buttons."""
            _apply_row_highlight()
            _refresh_buttons()

        compare_button.configure(command=_on_compare)
        export_button.configure(command=_on_export)
        run_tree.bind("<ButtonRelease-1>", _on_tree_click)
        run_tree.bind("<<TreeviewSelect>>", _on_tree_select)

        # Auto-select the first run so the chart isn't empty.
        if bench_state["entries"]:
            first_iid = run_tree.get_children()[0]
            run_tree.selection_set(first_iid)
            self._draw_full_benchmark_chart(
                bench_state, [bench_state["iid_to_run"][first_iid]],
            )
            self._populate_benchmark_detail(
                detail_tree, [bench_state["iid_to_run"][first_iid]],
            )
            _apply_row_highlight()
            _refresh_buttons()
        else:
            tk.Label(
                chart_frame,
                text="No benchmarks recorded yet. Run a benchmark to see results here.",
                bg=_BG_DARK, fg=_FG_GREY, font=FONT_HUD,
            ).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self._swap_frame(frame, "benchmarks")

    # %%
    def _populate_benchmarks_tree(self, bench_state: dict[str, Any]) -> None:
        """Fill the run-list treeview from ``benchmark_history.json``.

        Columns rendered per row: Date, Version(s), Video, Backend combos
        (e.g. ``PyTorch FP32/FP16 + ONNX FP32``), Best FPS (highest mean FPS
        across all combos in the entry).

        Args:
            bench_state: The Benchmarks screen state dict.
        """
        run_tree: ttk.Treeview = bench_state["run_tree"]
        entries: list[dict[str, Any]] = bench_state["entries"]
        bench_state["iid_to_run"].clear()
        for index, entry in enumerate(entries):
            combos = entry.get("combos", []) or []
            fps_values = [
                float(combo.get("mean_fps", 0.0)) for combo in combos
                if combo.get("mean_fps") is not None
            ]
            best_fps = max(fps_values) if fps_values else 0.0
            versions_raw = entry.get("yolo_versions") or (
                [entry.get("yolo_version", "")]
                if entry.get("yolo_version") else []
            )
            version_str = ", ".join(
                version_display(v) for v in sorted(set(versions_raw)) if v
            ) or "-"
            video_key = entry.get("video_key") or (
                ", ".join(entry.get("video_files") or []) or "-"
            )
            backend_combo_summary: list[str] = []
            seen_pairs: set[tuple[str, str]] = set()
            for combo in combos:
                pair = (
                    str(combo.get("backend", "")),
                    str(combo.get("precision", "")),
                )
                if pair in seen_pairs or not pair[0]:
                    continue
                seen_pairs.add(pair)
                backend_combo_summary.append(
                    f"{_BACKEND_DISPLAY.get(pair[0], pair[0])} {pair[1].upper()}"
                )
            iid = f"run_{index}"
            run_tree.insert(
                "", tk.END, iid=iid,
                values=(
                    "",
                    str(entry.get("timestamp", ""))[:19],
                    version_str,
                    video_key,
                    ", ".join(backend_combo_summary) or "-",
                    f"{best_fps:.1f}",
                ),
            )
            bench_state["iid_to_run"][iid] = entry

    # %%
    def _on_chart_metric_changed(self, selected_label: str) -> None:
        """Segmented-button callback: redraw the chart in the chosen metric.

        Args:
            selected_label: ``"Pipeline FPS"`` or ``"Kernel FPS"`` (the
                ``CTkSegmentedButton`` passes the visible label, not the
                internal value, so we translate here).
        """
        bench_state = self._benchmarks_screen_state
        if bench_state is None:
            return
        metric_var: tk.StringVar | None = bench_state.get("metric_var")
        if metric_var is None:
            return
        metric_var.set(
            "kernel" if selected_label == "Kernel FPS" else "pipeline",
        )
        runs = bench_state.get("current_runs", []) or []
        if runs:
            self._draw_full_benchmark_chart(bench_state, runs)

    # %%
    def _draw_full_benchmark_chart(
        self,
        bench_state: dict[str, Any],
        runs: list[dict[str, Any]],
    ) -> None:
        """Render the Full Benchmark chart for ``runs`` inside the right panel.

        Previous figure (if any) is closed via ``plt.close`` before the new
        canvas is constructed to keep matplotlib's open-figure count flat.

        Args:
            bench_state: Benchmarks-screen state dict.
            runs: Entries to chart. Single entry uses the combo-coloured
                palette; two or more entries trigger Compare mode.
        """
        chart_frame: tk.Frame = bench_state["chart_frame"]
        video_header: tk.Label = bench_state["video_header"]
        # Tear down whatever the right panel currently shows.
        for child in list(chart_frame.winfo_children()):
            child.destroy()
        prev_canvas = bench_state.get("chart_canvas")
        if prev_canvas is not None:
            try:
                prev_canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass
        prev_figure = bench_state.get("chart_figure")
        if prev_figure is not None:
            try:
                plt.close(prev_figure)
            except Exception:  # noqa: BLE001 - never block UI
                pass
        bench_state["chart_canvas"] = None
        bench_state["chart_figure"] = None
        if not runs:
            video_header.config(text="")
            return
        if len(runs) == 1:
            video_key = runs[0].get("video_key") or (
                (runs[0].get("video_files") or [""])[0]
            )
            video_header.config(text=f"Video: {video_key}")
        else:
            video_header.config(
                text=f"Comparing {len(runs)} runs",
            )
        metric_var: tk.StringVar | None = bench_state.get("metric_var")
        metric = (
            metric_var.get() if metric_var is not None else "pipeline"
        )
        canvas, figure = self._embed_full_benchmark_chart(
            chart_frame, runs, metric=metric,
        )
        bench_state["chart_canvas"] = canvas
        bench_state["chart_figure"] = figure
        bench_state["current_runs"] = runs

    # %%
    def _sort_model_keys_for_chart(
        self, keys: list[str],
    ) -> list[str]:
        """Order ``keys`` by version first, then by canonical size letter.

        Args:
            keys: Model keys gathered from a history entry.

        Returns:
            ``keys`` sorted (yolo11 before yolo26; within each version: n,
            s, m, l, x). Unknown keys fall back to their literal sort order.
        """
        size_index = {letter: idx for idx, letter in enumerate(_SIZE_ORDER)}

        def sort_key(model_key: str) -> tuple[str, int, str]:
            spec = self.model_by_key.get(model_key, {})
            version = spec.get("version", model_key)
            letter = spec.get("size_letter", "z")
            return (version, size_index.get(letter, 99), model_key)

        return sorted(keys, key=sort_key)

    # %%
    def _ordered_model_keys_for_entry(
        self, entry: dict[str, Any],
    ) -> list[str]:
        """Pull and order the unique model keys present in ``entry``.

        Args:
            entry: A benchmark-history entry dict.

        Returns:
            Model keys in chart order (version then size).
        """
        keys: list[str] = []
        seen: set[str] = set()
        for combo in entry.get("combos", []) or []:
            key = str(combo.get("model_key", ""))
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
        return self._sort_model_keys_for_chart(keys)

    # %%
    def _build_full_benchmark_figure(
        self,
        entries: list[dict[str, Any]],
        title_override: str | None = None,
        metric: str = "pipeline",
    ) -> "Figure":
        """Build the grouped-bar Full Benchmark Figure for ``entries``.

        Single entry: bars coloured by backend+precision combo per the
        fixed palette. Missing combos render as hatched outlines in
        :data:`FULL_BENCH_NA_COLOR` with an "N/A" label.

        Multiple entries (Compare mode): each entry contributes a colour
        family derived from the same combo palette; one cluster per
        ``(model_key, backend+precision)`` group, one bar per entry.

        Args:
            entries: Benchmark-history entries to chart. Must be non-empty.
            title_override: Optional title to use instead of the default.
            metric: ``"pipeline"`` for full mean FPS (default) or
                ``"kernel"`` for raw forward-pass FPS.

        Returns:
            A configured :class:`matplotlib.figure.Figure` (caller embeds
            it via FigureCanvasTkAgg and is responsible for closing it).
        """
        fps_field = "kernel_fps" if metric == "kernel" else "mean_fps"
        y_label = "Kernel FPS" if metric == "kernel" else "Mean FPS"
        # Wider, slightly taller figure so 6+ model groups fit across without
        # crowding and the long title doesn't get clipped at the right edge.
        figure = Figure(figsize=(11, 5.2), dpi=100, facecolor=BG_PRIMARY)
        axes = figure.add_subplot(111, facecolor=BG_SECONDARY)
        axes.tick_params(colors=TEXT_MUTED)
        for spine in axes.spines.values():
            spine.set_color(BORDER)
        axes.set_ylabel(y_label, color=TEXT_PRIMARY)
        axes.yaxis.grid(True, color="#2A2A2A", alpha=0.5, linewidth=0.8)
        axes.set_axisbelow(True)

        # Aggregate the unique model keys across every supplied entry.
        union_keys: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            for key in self._ordered_model_keys_for_entry(entry):
                if key not in seen:
                    union_keys.append(key)
                    seen.add(key)
        union_keys = self._sort_model_keys_for_chart(union_keys)
        x_positions = np.arange(len(union_keys))

        # Hard cap the Y axis at 500 FPS - this is high enough to cover
        # every realistic combo (TRT FP16 tops out around 450 on a 4070)
        # while making sure the small PyTorch bars stay visible regardless
        # of what the tall TRT bars do. Bars above the cap are clipped to
        # 500 and labelled with a ">500" prefix instead of their raw FPS.
        ymax = 500.0
        axes.set_ylim(0, ymax)
        # Height used for hatched N/A bars - tall enough to be obvious but
        # not so tall it dwarfs the real data.
        na_bar_height = ymax * 0.5

        if len(entries) == 1:
            entry = entries[0]
            combo_count = len(FULL_BENCH_COMBO_ORDER)
            bar_width = 0.8 / max(combo_count, 1)
            combo_index_by_key: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
            for combo in entry.get("combos", []) or []:
                pair = (
                    str(combo.get("backend", "")),
                    str(combo.get("precision", "")),
                )
                combo_index_by_key.setdefault(
                    str(combo.get("model_key", "")), {},
                )[pair] = combo
            for slot, pair in enumerate(FULL_BENCH_COMBO_ORDER):
                colour = FULL_BENCH_COMBO_COLORS[pair]
                offsets = x_positions - 0.4 + (slot + 0.5) * bar_width
                for x_index, model_key in enumerate(union_keys):
                    combo = combo_index_by_key.get(model_key, {}).get(pair)
                    fps = (
                        float(combo.get(fps_field))
                        if combo is not None
                        and combo.get(fps_field) is not None
                        else None
                    )
                    if fps is None:
                        axes.bar(
                            offsets[x_index],
                            na_bar_height,
                            bar_width,
                            color="none",
                            edgecolor=FULL_BENCH_NA_COLOR,
                            hatch="////",
                            linewidth=1.2,
                            zorder=2,
                        )
                        axes.text(
                            offsets[x_index],
                            na_bar_height / 2.0,
                            "N/A",
                            ha="center", va="center",
                            color=TEXT_MUTED, fontsize=8,
                            zorder=3,
                        )
                    else:
                        # Clip the bar height at the fixed 500 cap. The
                        # label always shows the true FPS value, but adds
                        # a ">" prefix when the bar was clipped so the
                        # cap is visually obvious.
                        clipped = min(fps, ymax)
                        bar = axes.bar(
                            offsets[x_index],
                            clipped,
                            bar_width,
                            color=colour,
                            zorder=2,
                        )[0]
                        label_text = (
                            f">{int(ymax)}" if fps > ymax else f"{fps:.1f}"
                        )
                        axes.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height() + 5,
                            label_text,
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color=TEXT_PRIMARY,
                            fontweight="bold",
                            zorder=3,
                        )
            legend_handles = [
                mpatches.Patch(
                    facecolor=FULL_BENCH_COMBO_COLORS[pair],
                    edgecolor=FULL_BENCH_COMBO_COLORS[pair],
                    label=(
                        f"{_BACKEND_DISPLAY.get(pair[0], pair[0])} "
                        f"{pair[1].upper()}"
                    ),
                )
                for pair in FULL_BENCH_COMBO_ORDER
            ]
            axes.legend(
                handles=legend_handles,
                loc="upper right",
                facecolor=BG_PRIMARY,
                edgecolor=BORDER,
                labelcolor=TEXT_PRIMARY,
                fontsize=8,
            )
            if title_override is not None:
                title = title_override
            else:
                video_key = entry.get("video_key") or (
                    (entry.get("video_files") or [""])[0]
                )
                # Use just the date portion (first 10 chars of ISO timestamp)
                # so the title doesn't get clipped at the right edge on
                # narrower window widths.
                date_only = str(entry.get("timestamp", ""))[:10]
                title = f"Full Benchmark - {video_key} - {date_only}"
            axes.set_title(title, color=TEXT_PRIMARY, fontsize=11)
        else:
            # Multi-run Compare mode: one colour family per run, bars
            # clustered by (model_key, combo) across runs.
            family_palette = (
                "#E87722", "#00B4A6", "#3498DB",
                "#9B59B6", "#F1C40F", "#1ABC9C",
            )
            n_runs = len(entries)
            cluster_pairs: list[tuple[str, tuple[str, str]]] = []
            for model_key in union_keys:
                for pair in FULL_BENCH_COMBO_ORDER:
                    cluster_pairs.append((model_key, pair))
            cluster_positions = np.arange(len(cluster_pairs))
            bar_width = 0.8 / max(n_runs, 1)
            for run_index, entry in enumerate(entries):
                lookup: dict[tuple[str, str, str], float | None] = {}
                for combo in entry.get("combos", []) or []:
                    triple = (
                        str(combo.get("model_key", "")),
                        str(combo.get("backend", "")),
                        str(combo.get("precision", "")),
                    )
                    raw = combo.get(fps_field)
                    lookup[triple] = (
                        float(raw) if raw is not None else None
                    )
                heights = [
                    lookup.get((model_key, pair[0], pair[1]), None)
                    for model_key, pair in cluster_pairs
                ]
                offsets = (
                    cluster_positions - 0.4 + (run_index + 0.5) * bar_width
                )
                run_colour = family_palette[run_index % len(family_palette)]
                for cluster_index, fps in enumerate(heights):
                    if fps is None:
                        continue
                    axes.bar(
                        offsets[cluster_index],
                        fps,
                        bar_width,
                        color=run_colour,
                        zorder=2,
                    )
            axes.set_xticks(cluster_positions)
            axes.set_xticklabels(
                [
                    f"{model_key.replace('-seg', '')}\n"
                    f"{_BACKEND_DISPLAY.get(pair[0], pair[0])} {pair[1].upper()}"
                    for model_key, pair in cluster_pairs
                ],
                rotation=45 if len(cluster_pairs) > 4 else 0,
                ha="right", fontsize=7,
            )
            legend_handles = [
                mpatches.Patch(
                    facecolor=family_palette[run_index % len(family_palette)],
                    label=(
                        f"{str(entry.get('timestamp', ''))[:19]}  "
                        f"{entry.get('video_key', '')}"
                    ),
                )
                for run_index, entry in enumerate(entries)
            ]
            axes.legend(
                handles=legend_handles,
                loc="upper right",
                facecolor=BG_PRIMARY,
                edgecolor=BORDER,
                labelcolor=TEXT_PRIMARY,
                fontsize=8,
            )
            axes.set_title(
                title_override or "Full Benchmark Comparison",
                color=TEXT_PRIMARY,
            )

        # Common X-axis styling for the single-run branch (multi-run path
        # already set its own labels above).
        if len(entries) == 1:
            axes.set_xticks(x_positions)
            label_rotation = 45 if len(union_keys) > 4 else 0
            axes.set_xticklabels(
                [key.replace("-seg", "") for key in union_keys],
                rotation=label_rotation,
                ha="right" if label_rotation else "center",
                color=TEXT_MUTED,
            )

        axes.set_xlabel("Model", color=TEXT_PRIMARY)
        figure.tight_layout()
        return figure

    # %%
    def _embed_full_benchmark_chart(
        self,
        parent: tk.Widget,
        entries: list[dict[str, Any]],
        title_override: str | None = None,
        metric: str = "pipeline",
    ) -> tuple["FigureCanvasTkAgg", "Figure"]:
        """Build and embed the grouped chart with horizontal scroll.

        The figure width is computed from how many model groups + bars
        there are; when the figure is wider than the container, a
        horizontal ``ttk.Scrollbar`` activates so the chart never gets
        squashed at narrow window widths. When the figure is narrower
        than the container it stretches to fill.

        Args:
            parent: Container that will host the canvas + scrollbar.
            entries: Benchmark entries to chart. Single = single-run
                colours; two or more = Compare mode with colour families.
            title_override: Optional title override.
            metric: ``"pipeline"`` (default) or ``"kernel"`` - selects
                which FPS series is bar-charted.

        Returns:
            ``(canvas, figure)`` so the caller can dispose of them later.
        """
        # ---- Compute figure dimensions --------------------------------
        union_keys: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            for key in self._ordered_model_keys_for_entry(entry):
                if key not in seen:
                    union_keys.append(key)
                    seen.add(key)
        n_groups = max(len(union_keys), 1)
        n_bars = len(FULL_BENCH_COMBO_ORDER)
        fig_width_inches = max(12.0, n_groups * n_bars * 0.8)
        fig_height_inches = 8.0

        figure = self._build_full_benchmark_figure(
            entries, title_override, metric=metric,
        )
        figure.set_size_inches(fig_width_inches, fig_height_inches)

        # ---- Outer container with horizontal scrollbar ----------------
        outer = tk.Frame(parent, bg=BG_PRIMARY)
        outer.pack(fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(
            outer, orient=tk.HORIZONTAL,
            style="BenchDark.Horizontal.TScrollbar",
        )
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        scroll_canvas = tk.Canvas(
            outer, bg=BG_PRIMARY, highlightthickness=0,
            xscrollcommand=h_scroll.set,
        )
        scroll_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        h_scroll.configure(command=scroll_canvas.xview)

        canvas = FigureCanvasTkAgg(figure, master=scroll_canvas)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(bg=BG_PRIMARY, highlightthickness=0)
        fig_w_px_initial = int(fig_width_inches * figure.dpi)
        window_id = scroll_canvas.create_window(
            (0, 0), window=widget, anchor="nw",
            width=fig_w_px_initial,
            height=int(fig_height_inches * figure.dpi),
        )

        # Closure state for the resize handler so successive Configure
        # events don't redraw with an unchanged size.
        state: dict[str, int] = {"last_w": 0, "last_h": 0}

        def _on_scroll_canvas_configure(event: tk.Event) -> None:
            """Resize the inner widget + figure to fit the visible area.

            Args:
                event: The Configure event carrying the new size.
            """
            canvas_w = max(event.width, 1)
            canvas_h = max(event.height, 1)
            # If the figure is narrower than the container, stretch it
            # to fill; if wider, leave it at fig_w_px_initial and let
            # the horizontal scrollbar take over.
            target_w = max(fig_w_px_initial, canvas_w)
            target_h = canvas_h
            if target_w == state["last_w"] and target_h == state["last_h"]:
                return
            state["last_w"] = target_w
            state["last_h"] = target_h
            scroll_canvas.itemconfigure(
                window_id, width=target_w, height=target_h,
            )
            try:
                figure.set_size_inches(
                    target_w / figure.dpi, target_h / figure.dpi,
                )
                figure.tight_layout()
            except Exception:  # noqa: BLE001 - keep UI alive
                pass
            canvas.draw()
            scroll_canvas.configure(
                scrollregion=(0, 0, target_w, target_h),
            )

        scroll_canvas.bind("<Configure>", _on_scroll_canvas_configure)
        return canvas, figure

    # %%
    def _save_full_benchmark_chart_png(
        self,
        entry: dict[str, Any],
    ) -> Path | None:
        """Save the entry's chart to ``results/benchmark_chart_*.png``.

        The PNG goes into ``_RESULTS_DIR`` named after the video key and
        timestamp so successive runs do not clobber each other.

        Args:
            entry: The benchmark-history entry whose chart to render.

        Returns:
            The saved PNG path, or ``None`` if writing failed.
        """
        video_key = entry.get("video_key") or (
            (entry.get("video_files") or ["video"])[0]
        )
        timestamp = (
            str(entry.get("timestamp", "")).replace(":", "-").replace(" ", "_")
        )
        safe_video = video_key.replace(" ", "_").replace(".", "_")
        out_path = (
            _RESULTS_DIR
            / f"benchmark_chart_{safe_video}_{timestamp}.png"
        )
        try:
            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            figure = self._build_full_benchmark_figure([entry])
            figure.savefig(
                str(out_path), dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight",
            )
            plt.close(figure)
        except (OSError, ValueError):
            return None
        return out_path

    # %%
    def _populate_benchmark_detail(
        self,
        detail_tree: ttk.Treeview,
        runs: list[dict[str, Any]],
    ) -> None:
        """Fill the detail treeview with every combo across ``runs``.

        Args:
            detail_tree: The detail-table widget.
            runs: Selected benchmark entries.
        """
        for iid in detail_tree.get_children():
            detail_tree.delete(iid)

        def fmt(value: Any) -> str:
            """Format a numeric column - ``None`` becomes ``"N/A"``."""
            if value is None:
                return "N/A"
            try:
                return f"{float(value):.1f}"
            except (TypeError, ValueError):
                return "N/A"

        for run in runs:
            for combo in run.get("combos", []) or []:
                provider = combo.get("active_provider") or "N/A"
                detail_tree.insert(
                    "", tk.END,
                    values=(
                        str(combo.get("model_key", "")),
                        str(combo.get("size", "")),
                        str(combo.get("backend", "")),
                        str(combo.get("precision", "")),
                        str(combo.get("device", "")),
                        str(provider),
                        fmt(combo.get("mean_fps")),
                        fmt(combo.get("p50_ms")),
                        fmt(combo.get("p95_ms")),
                        fmt(combo.get("p99_ms")),
                        fmt(combo.get("cold_start_ms")),
                        int(combo.get("frame_count") or 0),
                        fmt(combo.get("kernel_fps")),
                        fmt(combo.get("kernel_p50_ms")),
                    ),
                )

    # %%
    def _benchmark_export_targets(
        self, bench_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Resolve which runs the Export CSV button should write.

        Checked runs win when present; otherwise the currently-highlighted
        row in the treeview is used. Returns an empty list when neither is
        available so the caller can no-op gracefully.

        Args:
            bench_state: The Benchmarks screen state dict.

        Returns:
            The list of run entries to export.
        """
        checked_runs = [
            bench_state["iid_to_run"][iid]
            for iid in bench_state["checked_runs"]
            if iid in bench_state["iid_to_run"]
        ]
        if checked_runs:
            return checked_runs
        run_tree: ttk.Treeview = bench_state["run_tree"]
        selected = run_tree.selection()
        return [
            bench_state["iid_to_run"][iid]
            for iid in selected if iid in bench_state["iid_to_run"]
        ]

    # %%
    def _export_benchmark_history_csv(
        self,
        destination: Path,
        runs: list[dict[str, Any]],
    ) -> None:
        """Write benchmark-history combos to ``destination`` in CSV form.

        Args:
            destination: User-chosen CSV path.
            runs: Benchmark entries whose combos should be flattened to rows.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp", "yolo_version", "run_id",
            "model_key", "size", "backend", "precision", "device",
            "mean_fps", "p50_ms", "p95_ms", "p99_ms",
            "cold_start_ms", "frame_count",
        ]
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for run in runs:
                for combo in run.get("combos", []) or []:
                    row = {
                        "timestamp": run.get("timestamp", ""),
                        "yolo_version": run.get("yolo_version", ""),
                        "run_id": run.get("run_id", ""),
                    }
                    for key in fieldnames[3:]:
                        row[key] = combo.get(key, "")
                    writer.writerow(row)

    # ------------------------------------------------------------------
    # History screen
    # ------------------------------------------------------------------
    # %%
    def show_history_screen(self) -> None:
        """Build and display the History screen."""
        # Style the Treeview to match the customtkinter dark theme. ttk lives
        # outside customtkinter's theme system so this has to be applied
        # manually each time we open the screen.
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Treeview",
            background=BG_PRIMARY, foreground=TEXT_PRIMARY,
            fieldbackground=BG_PRIMARY, bordercolor=BORDER,
            rowheight=px(22),
        )
        style.configure(
            "Treeview.Heading",
            background=BG_CARD, foreground=ACCENT_ORANGE,
            relief="flat",
        )
        style.map(
            "Treeview",
            background=[("selected", BG_HOVER)],
            foreground=[("selected", TEXT_PRIMARY)],
        )

        frame = tk.Frame(self.root, bg=_BG_DARK)

        header = tk.Frame(frame, bg=_BG_DARK)
        header.pack(fill=tk.X, padx=px(20), pady=px(15))
        tk.Button(
            header,
            text="← Back",
            bg=_BG_HOVER,
            fg=_FG_DEFAULT,
            activebackground="#333333",
            activeforeground=_FG_DEFAULT,
            bd=0,
            padx=px(14),
            pady=px(4),
            command=self.show_main_menu,
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            text="Run History",
            bg=_BG_DARK,
            fg=_ORANGE,
            font=("Helvetica", px(22), "bold"),
        ).pack(side=tk.LEFT, padx=px(15))

        table_container = tk.Frame(frame, bg=_BG_DARK)
        table_container.pack(fill=tk.X, padx=px(20))

        columns = (
            "datetime",
            "models",
            "sizes",
            "backend",
            "precision",
            "device",
            "videos",
            "fps",
            "detection_rate",
            "duration",
            "snapshots",
        )
        headings = {
            "datetime": "Date / Time",
            "models": "Models",
            "sizes": "Sizes",
            "backend": "Backend",
            "precision": "Precision",
            "device": "Device",
            "videos": "Videos",
            "fps": "Mean FPS",
            "detection_rate": "Detect %",
            "duration": "Duration (s)",
            "snapshots": "Snapshots",
        }
        widths = {
            "datetime": px(140),
            "models": px(160),
            "sizes": px(110),
            "backend": px(80),
            "precision": px(70),
            "device": px(60),
            "videos": px(180),
            "fps": px(80),
            "detection_rate": px(80),
            "duration": px(80),
            "snapshots": px(80),
        }

        tree_scroll = tk.Scrollbar(table_container)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree = ttk.Treeview(
            table_container,
            columns=columns,
            show="headings",
            height=10,
            yscrollcommand=tree_scroll.set,
        )
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor=tk.W)
        tree.pack(fill=tk.X)
        tree_scroll.config(command=tree.yview)

        entries = self.history.load()
        self._history_entries_cache = entries
        for entry in entries:
            per_model = entry.get("per_model_stats", {})
            mean_fps_values = [
                float(stat.get("mean_fps", 0.0)) for stat in per_model.values()
            ]
            avg_fps = (
                float(np.mean(mean_fps_values)) if mean_fps_values else 0.0
            )
            detect_rates = [
                float(stat.get("detection_rate", 0.0)) for stat in per_model.values()
            ]
            avg_det = float(np.mean(detect_rates)) if detect_rates else 0.0
            sizes_value = ", ".join(entry.get("model_sizes", [])) or "N/A"
            tree.insert(
                "",
                tk.END,
                iid=entry["run_id"],
                values=(
                    str(entry.get("timestamp", "")),
                    ", ".join(entry.get("model_list", [])),
                    sizes_value,
                    str(entry.get("backend", "N/A")),
                    str(entry.get("precision", "N/A")),
                    str(entry.get("device", "N/A")),
                    ", ".join(entry.get("video_list", [])),
                    f"{avg_fps:.1f}",
                    f"{100.0 * avg_det:.1f}",
                    f"{float(entry.get('duration_seconds', 0.0)):.1f}",
                    str(len(entry.get("snapshot_paths", []))),
                ),
            )
        tree.bind("<<TreeviewSelect>>", self._on_history_select)
        self._menu_widgets["history_tree"] = tree

        # Detail panel below the table.
        detail_outer = tk.Frame(frame, bg=_BG_DARK)
        detail_outer.pack(fill=tk.BOTH, expand=True, padx=px(20), pady=px(15))
        self._history_detail_frame = tk.Frame(detail_outer, bg=_BG_PANEL)
        self._history_detail_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            self._history_detail_frame,
            text="Select a run above to see its details.",
            bg=_BG_PANEL,
            fg=_FG_GREY,
            font=("Helvetica", px(11), "italic"),
        ).pack(padx=px(20), pady=px(20), anchor=tk.W)

        self._swap_frame(frame, "history")

    # %%
    def _on_history_select(self, _event: tk.Event) -> None:
        """Treeview select callback: populate the detail panel for the row."""
        tree = self._menu_widgets.get("history_tree")
        if tree is None:
            return
        selection = tree.selection()
        if not selection:
            return
        run_id = selection[0]
        entry = next(
            (item for item in self._history_entries_cache if item.get("run_id") == run_id),
            None,
        )
        if entry is None:
            return
        self._populate_history_detail(entry)

    # %%
    def _populate_history_detail(self, entry: dict[str, Any]) -> None:
        """Render the stats card, thumbnail gallery and action buttons.

        Args:
            entry: The history-entry dict to render.
        """
        for child in self._history_detail_frame.winfo_children():
            child.destroy()
        self._history_thumb_refs = []

        # Stats cards row.
        cards = tk.Frame(self._history_detail_frame, bg=_BG_PANEL)
        cards.pack(fill=tk.X, padx=px(15), pady=px(15))
        for display_name, stats in entry.get("per_model_stats", {}).items():
            self._build_stats_card(cards, display_name, stats)

        # Thumbnail gallery.
        gallery_label = tk.Label(
            self._history_detail_frame,
            text="Snapshots",
            bg=_BG_PANEL,
            fg=_FG_GREY,
            font=("Helvetica", px(11), "italic"),
        )
        gallery_label.pack(anchor=tk.W, padx=px(20))
        gallery_container = tk.Frame(self._history_detail_frame, bg=_BG_PANEL)
        gallery_container.pack(fill=tk.X, padx=px(20), pady=(px(4), px(10)))

        gallery_canvas = tk.Canvas(
            gallery_container, bg=_BG_PANEL, height=px(140), highlightthickness=0
        )
        gallery_canvas.pack(side=tk.TOP, fill=tk.X)
        h_scroll = tk.Scrollbar(
            gallery_container, orient=tk.HORIZONTAL, command=gallery_canvas.xview
        )
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        gallery_canvas.configure(xscrollcommand=h_scroll.set)
        gallery_inner = tk.Frame(gallery_canvas, bg=_BG_PANEL)
        gallery_canvas.create_window((0, 0), window=gallery_inner, anchor=tk.NW)

        ranked = gather_snapshot_confidences(entry.get("snapshot_paths", []))
        if not ranked:
            tk.Label(
                gallery_inner,
                text="(no snapshots saved)",
                bg=_BG_PANEL,
                fg=_FG_GREY,
                font=("Helvetica", px(10), "italic"),
            ).pack(side=tk.LEFT, padx=px(10), pady=px(10))
        for path, _confidence in ranked:
            try:
                with Image.open(path) as image:
                    image.thumbnail((px(160), px(110)))
                    photo = ImageTk.PhotoImage(image.copy())
            except (OSError, ValueError):
                continue
            self._history_thumb_refs.append(photo)
            cell = tk.Frame(gallery_inner, bg=_BG_PANEL, padx=px(4), pady=px(4))
            cell.pack(side=tk.LEFT, padx=px(4))
            label = tk.Label(cell, image=photo, bg=_BG_PANEL, cursor="hand2")
            label.pack()
            label.bind(
                "<Button-1>", functools.partial(self._enlarge_snapshot, path)
            )
            tk.Label(
                cell,
                text=path.name,
                bg=_BG_PANEL,
                fg=_FG_GREY,
                font=("Helvetica", px(8)),
            ).pack()

        gallery_inner.update_idletasks()
        gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all"))

        # Action buttons.
        actions = tk.Frame(self._history_detail_frame, bg=_BG_PANEL)
        actions.pack(fill=tk.X, padx=px(20), pady=(0, px(15)))
        tk.Button(
            actions,
            text="Export PDF",
            bg=_ORANGE,
            fg="#FFFFFF",
            activebackground="#FF8A33",
            activeforeground="#FFFFFF",
            bd=0,
            padx=px(14),
            pady=px(6),
            command=functools.partial(self._on_history_export, entry),
        ).pack(side=tk.LEFT)
        tk.Button(
            actions,
            text="Delete",
            bg=_DARK_RED,
            fg="#FFFFFF",
            activebackground="#9B3422",
            activeforeground="#FFFFFF",
            bd=0,
            padx=px(14),
            pady=px(6),
            command=functools.partial(self._on_history_delete, entry),
        ).pack(side=tk.LEFT, padx=px(10))

    # %%
    def _enlarge_snapshot(self, path: Path, _event: tk.Event) -> None:
        """Open a popup window showing the snapshot at a larger size.

        Args:
            path: The snapshot JPEG path.
            _event: The tkinter mouse event (unused).
        """
        try:
            image = Image.open(path)
        except (OSError, ValueError):
            return
        popup = tk.Toplevel(self.root)
        popup.title(path.name)
        popup.configure(bg=_BG_DARK)
        image.thumbnail((px(1100), px(750)))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(popup, image=photo, bg=_BG_DARK)
        label.image = photo  # keep ref alive for the popup's lifetime
        label.pack(padx=px(10), pady=px(10))
        tk.Button(
            popup,
            text="Close",
            bg=_BG_HOVER,
            fg=_FG_DEFAULT,
            bd=0,
            padx=px(14),
            pady=px(4),
            command=popup.destroy,
        ).pack(pady=(0, px(10)))

    # %%
    def _on_history_export(self, entry: dict[str, Any]) -> None:
        """Export the selected history entry as a one-page PDF.

        Args:
            entry: The history-entry dict to export.
        """
        destination = _EXPORTS_DIR / f"{entry['run_id']}.pdf"
        try:
            export_run_pdf(entry, destination)
        except Exception as exc:  # noqa: BLE001 - surface to the user
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Export complete", f"PDF written to:\n{destination}")

    # %%
    def _on_history_delete(self, entry: dict[str, Any]) -> None:
        """Remove a history entry after confirmation.

        Args:
            entry: The history-entry dict to delete.
        """
        confirmed = messagebox.askyesno(
            "Delete run",
            f"Remove this run from history?\n\nRun: {entry.get('timestamp','')}\n"
            "(Snapshot files on disk are NOT deleted.)",
        )
        if not confirmed:
            return
        self.history.delete(entry["run_id"])
        # Rebuild the history screen to reflect the deletion.
        self.show_history_screen()


# %%
def main() -> None:
    """Entry point: build the application and run the tkinter main loop."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    _INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # customtkinter requires its own root class; ctk.CTk subclasses tk.Tk so
    # every existing ttk / tk widget the app uses elsewhere still works.
    root = ctk.CTk()
    _ = App(root)
    root.mainloop()


# %%
if __name__ == "__main__":
    main()
