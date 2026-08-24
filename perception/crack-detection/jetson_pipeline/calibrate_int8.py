# %%
"""calibrate_int8.py - build TensorRT INT8 calibration caches from CrackSeg9k.

Implements NVIDIA's recommended post-training quantisation workflow: a
:class:`trt.IInt8EntropyCalibrator2` subclass streams real crack images through
the network while TensorRT builds the per-tensor dynamic-range table, then
persists that table to a ``.cache`` file. Calibration is architecture-specific,
so one cache is produced per model.

The module deliberately depends only on packages present in a stock JetPack
image (``tensorrt``, ``numpy``, ``opencv-python``, ``pyyaml``) plus one of
``pycuda`` / ``cuda-python`` / ``torch`` for the device staging buffer. Neither
Ultralytics nor PyTorch is required, so ``export_on_device.py`` can import this
file directly on the Jetson.

Usage:
    conda activate cuda_pt
    python jetson_pipeline/calibrate_int8.py            # build all missing caches
    python jetson_pipeline/calibrate_int8.py --force    # rebuild every cache
"""

from __future__ import annotations

# %% Imports
import argparse
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorrt as trt
import yaml

# %% Config
_HERE: Path = Path(__file__).parent
_CONFIG_PATH: Path = _HERE / "jetson_config.yaml"

# Letterbox fill value used by Ultralytics' preprocessing.
_LETTERBOX_FILL: int = 114

# Fixed RNG seed so the sampled calibration set is reproducible run to run.
_CALIB_SEED: int = 42

# Image extensions scanned in the calibration split.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")

# Canonical size ordering, shared with the rest of the pipeline.
_SIZE_ORDER: tuple[str, ...] = ("n", "s", "m", "l", "x")

# TensorRT logger reused by every builder in this module.
_TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.WARNING)


# %% TensorRT 10 compatibility shim
def patch_trt_nptype() -> None:
    """Restore ``trt.nptype`` on TensorRT 10+, which removed the helper.

    Ultralytics and a lot of sample code still call ``trt.nptype``. Injecting a
    compatible replacement at import time keeps both this module and any
    downstream Ultralytics engine load working on TRT 10. Mirrors the shim in
    ``crack_detection_benchmark/app.py``.
    """
    if hasattr(trt, "nptype"):
        return

    def _trt_dtype_to_np(dtype: Any) -> Any:
        """Map a TensorRT ``DataType`` to its numpy equivalent.

        Args:
            dtype: The TensorRT data type to translate.

        Returns:
            The matching numpy scalar type, defaulting to ``np.float32``.
        """
        mapping = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.INT32: np.int32,
            trt.DataType.BOOL: np.bool_,
        }
        return mapping.get(dtype, np.float32)

    trt.nptype = _trt_dtype_to_np


patch_trt_nptype()


# %%
def trt_major_version() -> int:
    """Return the installed TensorRT major version.

    Returns:
        The major version as an int, e.g. ``10`` for ``10.3.0``. Falls back to
        ``8`` when the version string cannot be parsed.
    """
    try:
        return int(str(trt.__version__).split(".")[0])
    except (AttributeError, ValueError):
        return 8


# %%
def resolve_builder_flag(name: str) -> Any:
    """Resolve a ``trt.BuilderFlag`` member across TensorRT naming conventions.

    The C++ API spells these ``kFP16`` / ``kINT8`` while the Python bindings
    expose ``FP16`` / ``INT8``; some builds carry both. Resolving by lookup
    rather than hardcoding one spelling keeps the module portable between
    TensorRT 8.x and 10.x.

    Args:
        name: Bare flag name without a prefix, e.g. ``"FP16"``.

    Returns:
        The resolved ``trt.BuilderFlag`` member.

    Raises:
        AttributeError: If neither spelling exists on this TensorRT build.
    """
    for candidate in (name, f"k{name}"):
        flag = getattr(trt.BuilderFlag, candidate, None)
        if flag is not None:
            return flag
    raise AttributeError(f"trt.BuilderFlag has no {name} / k{name} member.")


# %% Device staging buffer
class DeviceBuffer:
    """A device allocation used to feed calibration batches to TensorRT.

    TensorRT's calibrator interface hands raw device pointers back to the
    builder, so a small CUDA allocation is needed regardless of which framework
    is installed. Three allocators are tried in order - ``pycuda`` (the JetPack
    default), ``cuda-python``, then ``torch`` - so the same class works on the
    Jetson and on the training laptop.

    Attributes:
        nbytes: Size of the allocation in bytes.
        backend: Name of the allocator that satisfied the request.
    """

    def __init__(self, nbytes: int) -> None:
        """Allocate ``nbytes`` of device memory.

        Args:
            nbytes: Number of bytes to allocate.

        Raises:
            RuntimeError: If no supported CUDA allocator is importable.
        """
        self.nbytes: int = int(nbytes)
        self.backend: str = ""
        self._pointer: int = 0
        self._handle: Any = None
        self._runtime: Any = None

        if self._try_pycuda():
            return
        if self._try_cuda_python():
            return
        if self._try_torch():
            return
        raise RuntimeError(
            "INT8 calibration needs a CUDA allocator: install one of pycuda, "
            "cuda-python or torch."
        )

    def _try_pycuda(self) -> bool:
        """Attempt a pycuda allocation.

        Returns:
            ``True`` when the allocation succeeded.
        """
        try:
            import pycuda.autoinit  # noqa: F401 - the import creates the CUDA context
            import pycuda.driver as cuda
        except Exception:  # noqa: BLE001 - absence is expected; try the next one
            return False
        self._handle = cuda.mem_alloc(self.nbytes)
        self._runtime = cuda
        self._pointer = int(self._handle)
        self.backend = "pycuda"
        return True

    def _try_cuda_python(self) -> bool:
        """Attempt a cuda-python allocation.

        Returns:
            ``True`` when the allocation succeeded.
        """
        try:
            from cuda import cudart
        except Exception:  # noqa: BLE001 - absence is expected; try the next one
            return False
        status, pointer = cudart.cudaMalloc(self.nbytes)
        if int(status) != 0:
            return False
        self._runtime = cudart
        self._pointer = int(pointer)
        self.backend = "cuda-python"
        return True

    def _try_torch(self) -> bool:
        """Attempt a torch CUDA allocation.

        Returns:
            ``True`` when the allocation succeeded.
        """
        try:
            import torch
        except Exception:  # noqa: BLE001 - absence is expected on the Jetson
            return False
        if not torch.cuda.is_available():
            return False
        # A uint8 tensor gives a byte-exact allocation whose data_ptr TensorRT
        # can write into directly. Held on the instance so it stays alive.
        self._handle = torch.empty(self.nbytes, dtype=torch.uint8, device="cuda")
        self._runtime = torch
        self._pointer = int(self._handle.data_ptr())
        self.backend = "torch"
        return True

    def pointer(self) -> int:
        """Return the raw device pointer.

        Returns:
            The device address as an int, as TensorRT expects.
        """
        return self._pointer

    def copy_from_host(self, array: np.ndarray) -> None:
        """Copy a contiguous host array into the device allocation.

        Args:
            array: A C-contiguous numpy array no larger than the allocation.

        Raises:
            ValueError: If ``array`` exceeds the allocated size.
        """
        if array.nbytes > self.nbytes:
            raise ValueError(
                f"Host array is {array.nbytes} bytes, buffer is {self.nbytes}."
            )
        contiguous = np.ascontiguousarray(array)
        if self.backend == "pycuda":
            self._runtime.memcpy_htod(self._handle, contiguous)
        elif self.backend == "cuda-python":
            from cuda import cudart

            cudart.cudaMemcpy(
                self._pointer,
                contiguous.ctypes.data,
                contiguous.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        else:
            torch = self._runtime
            flat = contiguous.view(np.uint8).reshape(-1)
            self._handle[: flat.size].copy_(torch.from_numpy(flat))

    def free(self) -> None:
        """Release the device allocation. Safe to call more than once."""
        if self._pointer == 0:
            return
        if self.backend == "pycuda" and self._handle is not None:
            self._handle.free()
        elif self.backend == "cuda-python":
            from cuda import cudart

            cudart.cudaFree(self._pointer)
        # The torch backend frees its allocation when the tensor is dropped.
        self._handle = None
        self._pointer = 0


# %% Preprocessing
def letterbox(image: np.ndarray, imgsz: int) -> np.ndarray:
    """Resize an image into a square canvas preserving aspect ratio.

    Matches Ultralytics' inference-time letterbox: scale so the longest side
    equals ``imgsz``, then centre the result on a constant-114 canvas.

    Args:
        image: Source BGR image as ``HxWx3`` ``uint8``.
        imgsz: Target square side length.

    Returns:
        An ``imgsz x imgsz x 3`` BGR ``uint8`` canvas.
    """
    height, width = image.shape[:2]
    scale = min(imgsz / float(height), imgsz / float(width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((imgsz, imgsz, 3), _LETTERBOX_FILL, dtype=np.uint8)
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


# %%
def preprocess_image(path: Path, imgsz: int) -> np.ndarray | None:
    """Load and preprocess one calibration image exactly as YOLO would.

    Letterbox to ``imgsz``, BGR to RGB, HWC to CHW, scale to ``[0, 1]``,
    cast to float32 and make contiguous.

    Args:
        path: Path to the image file.
        imgsz: Target square side length.

    Returns:
        A contiguous ``(3, imgsz, imgsz)`` float32 array, or ``None`` when the
        file could not be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    boxed = letterbox(image, imgsz)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw, dtype=np.float32)


# %%
def collect_calibration_images(
    split_dir: Path,
    count: int,
    seed: int = _CALIB_SEED,
) -> list[Path]:
    """Sample calibration images from a dataset split directory.

    Args:
        split_dir: Directory holding the split's images.
        count: Maximum number of images to sample.
        seed: RNG seed - fixed so the sample is reproducible.

    Returns:
        Sampled image paths in sorted order. Fewer than ``count`` entries when
        the split is smaller; empty when the directory is absent.
    """
    if not split_dir.is_dir():
        return []
    candidates = sorted(
        path
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not candidates:
        return []
    if len(candidates) <= count:
        return candidates
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, count))


# %% Calibrator
class CrackCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibration using CrackSeg9k test split images.

    TensorRT calls :meth:`get_batch` repeatedly while profiling the network's
    activations, then hands the resulting dynamic-range table to
    :meth:`write_calibration_cache`. On a later build the cache is returned
    straight from :meth:`read_calibration_cache` and calibration is skipped.

    Attributes:
        cache_file: Path the calibration table is persisted to.
        imgsz: Square input resolution the images are letterboxed to.
        batch_size: Images per calibration batch.
    """

    def __init__(
        self,
        image_paths: list[Path],
        cache_file: Path,
        imgsz: int = 640,
        batch_size: int = 1,
    ) -> None:
        """Prepare the calibrator and its device staging buffer.

        Args:
            image_paths: Calibration images, already sampled.
            cache_file: Where to read/write the calibration table.
            imgsz: Square input resolution.
            batch_size: Images per calibration batch.

        Raises:
            ValueError: If ``image_paths`` is empty and no cache exists.
        """
        super().__init__()
        self.cache_file: Path = Path(cache_file)
        self.imgsz: int = int(imgsz)
        self.batch_size: int = max(1, int(batch_size))
        self._image_paths: list[Path] = list(image_paths)
        self._cursor: int = 0

        if not self._image_paths and not self.cache_file.is_file():
            raise ValueError(
                "No calibration images were found and no cache exists at "
                f"{self.cache_file}."
            )

        nbytes = self.batch_size * 3 * self.imgsz * self.imgsz * np.float32().itemsize
        # Skip the allocation entirely when a cache already covers this model:
        # TensorRT never calls get_batch in that case.
        self._buffer: DeviceBuffer | None = (
            None if self.cache_file.is_file() else DeviceBuffer(nbytes)
        )

    def get_batch_size(self) -> int:
        """Return the calibration batch size.

        Returns:
            Images per calibration batch.
        """
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        """Stage the next calibration batch on the device.

        Args:
            names: Input tensor names TensorRT wants filled, in binding order.

        Returns:
            A single-element list holding the device pointer, or ``None`` once
            the calibration set is exhausted.
        """
        if self._buffer is None or self._cursor >= len(self._image_paths):
            return None

        batch: list[np.ndarray] = []
        while len(batch) < self.batch_size and self._cursor < len(self._image_paths):
            path = self._image_paths[self._cursor]
            self._cursor += 1
            tensor = preprocess_image(path, self.imgsz)
            if tensor is not None:
                batch.append(tensor)

        if not batch:
            return None
        # Repeat the last image if the split ran short, so every batch is full
        # and TensorRT never profiles a partially-initialised buffer.
        while len(batch) < self.batch_size:
            batch.append(batch[-1])

        stacked = np.ascontiguousarray(np.stack(batch, axis=0), dtype=np.float32)
        self._buffer.copy_from_host(stacked)
        return [self._buffer.pointer()]

    def read_calibration_cache(self) -> bytes | None:
        """Return a previously written calibration table.

        Returns:
            The cache bytes when the file exists, otherwise ``None`` so
            TensorRT falls back to running calibration.
        """
        if self.cache_file.is_file():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Persist the calibration table TensorRT produced.

        Args:
            cache: The serialised dynamic-range table.
        """
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(bytes(cache))

    def close(self) -> None:
        """Release the device staging buffer. Safe to call more than once."""
        if self._buffer is not None:
            self._buffer.free()
            self._buffer = None


# %% Engine building
def build_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path | None,
    workspace_mib: int,
    fp16: bool,
    int8: bool,
    calibrator: CrackCalibrator | None,
    log: Any,
) -> bytes | None:
    """Build a TensorRT engine from an ONNX graph.

    Handles both the TensorRT 8.x and 10.x builder APIs: the ``EXPLICIT_BATCH``
    network flag and the ``max_workspace_size`` attribute are applied only when
    the installed build exposes them, and precision flags are resolved by name
    rather than hardcoded to one spelling.

    Args:
        onnx_path: Source ONNX graph.
        engine_path: Where to write the serialised engine. ``None`` builds the
            engine without persisting it, which is how a calibration-only pass
            produces its cache.
        workspace_mib: Builder workspace limit in MiB.
        fp16: Enable FP16 kernels.
        int8: Enable INT8 kernels (requires ``calibrator``).
        calibrator: The INT8 calibrator, or ``None`` for FP16/FP32 builds.
        log: Callable accepting one log line.

    Returns:
        The serialised engine bytes, or ``None`` when the build failed.
    """
    builder = trt.Builder(_TRT_LOGGER)

    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if trt_major_version() < 10 and explicit_batch is not None:
        network = builder.create_network(1 << int(explicit_batch))
    else:
        # TensorRT 10 networks are always explicit-batch; passing the flag is
        # deprecated and emits a warning.
        network = builder.create_network(0)

    parser = trt.OnnxParser(network, _TRT_LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        for index in range(parser.num_errors):
            log(f"[trt-parse] {parser.get_error(index)}")
        return None

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_mib) * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # TensorRT 8.3 and older
        config.max_workspace_size = workspace_bytes

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(resolve_builder_flag("FP16"))
    if int8:
        if not builder.platform_has_fast_int8:
            log("[trt-build] this platform reports no fast INT8 support")
        config.set_flag(resolve_builder_flag("INT8"))
        if calibrator is not None:
            config.int8_calibrator = calibrator

    log(f"[trt-build] building from {onnx_path.name} (fp16={fp16}, int8={int8})")
    if hasattr(builder, "build_serialized_network"):
        serialised = builder.build_serialized_network(network, config)
    else:  # TensorRT 8.0 and older
        engine = builder.build_engine(network, config)
        serialised = engine.serialize() if engine is not None else None

    if serialised is None:
        log(f"[trt-build] builder returned no engine for {onnx_path.name}")
        return None

    payload = bytes(serialised)
    if engine_path is not None:
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(payload)
        log(f"[trt-build] wrote {engine_path.name} ({len(payload) / 1e6:.1f} MB)")
    return payload


# %% Cache orchestration
def cache_path_for(model_name: str, cache_dir: Path, cache_suffix: str) -> Path:
    """Return the calibration cache path for one model.

    Args:
        model_name: Model key, e.g. ``yolo11n-seg``.
        cache_dir: Directory holding every calibration cache.
        cache_suffix: Stem suffix from the config, e.g. ``_int8``.

    Returns:
        ``{cache_dir}/{model_name}{cache_suffix}.cache``.
    """
    return cache_dir / f"{model_name}{cache_suffix}.cache"


# %%
def ensure_calibration_cache(
    model_name: str,
    onnx_path: Path,
    cache_file: Path,
    image_paths: list[Path],
    imgsz: int,
    workspace_mib: int,
    log: Any,
    force: bool = False,
) -> bool:
    """Build the calibration cache for one model unless it already exists.

    The cache is a by-product of an INT8 engine build, so a throwaway engine is
    built purely to drive calibration. That engine is discarded - the engines
    that ship are produced by ``export_all.py`` (laptop) and
    ``export_on_device.py`` (Jetson), both of which reuse this cache.

    Args:
        model_name: Model key, e.g. ``yolo11n-seg``.
        onnx_path: The model's ONNX graph.
        cache_file: Where the calibration table lives.
        image_paths: Sampled calibration images.
        imgsz: Square input resolution.
        workspace_mib: Builder workspace limit in MiB.
        log: Callable accepting one log line.
        force: Rebuild even when the cache already exists.

    Returns:
        ``True`` when a usable cache exists after the call.
    """
    if cache_file.is_file() and not force:
        log(f"[calib] {model_name}: cache present ({cache_file.name}) - reusing")
        return True
    if force and cache_file.is_file():
        cache_file.unlink()
        log(f"[calib] {model_name}: --force removed {cache_file.name}")
    if not onnx_path.is_file():
        log(f"[calib] {model_name}: no ONNX at {onnx_path.name} - skipped")
        return False
    if not image_paths:
        log(f"[calib] {model_name}: no calibration images available - skipped")
        return False

    log(f"[calib] {model_name}: calibrating on {len(image_paths)} images")
    calibrator = CrackCalibrator(image_paths, cache_file, imgsz=imgsz, batch_size=1)
    try:
        # FP16 is enabled alongside INT8 so TensorRT may fall back per layer,
        # and so this build matches the precision set the real export uses.
        # Note it does NOT rescue the YOLO-seg mask-prototype branch: on
        # TensorRT 10.7 / SM 8.9 the fused Conv+SiLU at model.23/proto/cv3 has
        # no conforming INT8 *or* FP16 tactic and aborts the build. That is
        # harmless here - TensorRT writes the calibration table before tactic
        # selection, so the cache this function exists to produce is complete
        # and reusable even when the throwaway engine fails to build.
        build_engine_from_onnx(
            onnx_path=onnx_path,
            engine_path=None,
            workspace_mib=workspace_mib,
            fp16=True,
            int8=True,
            calibrator=calibrator,
            log=log,
        )
    finally:
        calibrator.close()

    if cache_file.is_file():
        log(f"[calib] {model_name}: wrote {cache_file.name}")
        return True
    log(f"[calib] {model_name}: calibration produced no cache")
    return False


# %% Shared helpers
def load_config(config_path: Path = _CONFIG_PATH) -> dict[str, Any]:
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


# %%
def parse_onnx_stem(stem: str, finetuned_suffix: str) -> tuple[str, str] | None:
    """Parse an exported ONNX stem into ``(version, size_letter)``.

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
def print_line(message: str) -> None:
    """Print one log line to stdout.

    Used as the default ``log`` callable so the module works without ``rich``.

    Args:
        message: The line to print.
    """
    print(message)


# %% Main function
def main(force: bool) -> int:
    """Build every missing calibration cache for the exported ONNX models.

    Args:
        force: Rebuild caches that already exist.

    Returns:
        Process exit code - ``0`` when every discovered model has a cache.
    """
    config = load_config()
    models_dir = resolve_path(config["models_dir"])
    data_dir = resolve_path(config["data_dir"])
    export_cfg = config["export"]
    naming = config["naming"]

    cache_dir = models_dir / export_cfg["int8_calib_cache"]
    split = export_cfg["int8_calib_split"]
    split_dir = data_dir / "images" / split
    imgsz = int(export_cfg["imgsz"])
    # Standalone runs happen on the laptop; the Jetson drives calibration
    # through export_on_device.py, which passes jetson.workspace_mib instead.
    workspace_mib = int(export_cfg["workspace_mib"])

    print_line(f"[calib] TensorRT {trt.__version__} (major {trt_major_version()})")
    print_line(f"[calib] models:  {models_dir}")
    print_line(f"[calib] split:   {split_dir}")

    images = collect_calibration_images(split_dir, int(export_cfg["int8_calib_images"]))
    if not images:
        print_line(f"[calib] no images under {split_dir} - nothing to calibrate")
        return 1
    print_line(f"[calib] sampled {len(images)} image(s) with seed {_CALIB_SEED}")

    finetuned_suffix = naming["finetuned_suffix"]
    onnx_files = sorted(models_dir.glob(f"*{finetuned_suffix}.onnx"))
    if not onnx_files:
        print_line(
            f"[calib] no *{finetuned_suffix}.onnx under {models_dir} - "
            "run export_all.py first"
        )
        return 1

    built = 0
    failed = 0
    for onnx_path in onnx_files:
        parsed = parse_onnx_stem(onnx_path.stem, finetuned_suffix)
        if parsed is None:
            continue
        version, size_letter = parsed
        model_name = f"{version}{size_letter}-seg"
        cache_file = cache_path_for(model_name, cache_dir, naming["cache_suffix"])
        try:
            ok = ensure_calibration_cache(
                model_name=model_name,
                onnx_path=onnx_path,
                cache_file=cache_file,
                image_paths=images,
                imgsz=imgsz,
                workspace_mib=workspace_mib,
                log=print_line,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001 - one bad model must not stop the rest
            print_line(f"[calib] {model_name}: failed - {exc}")
            ok = False
        if ok:
            built += 1
        else:
            failed += 1

    print_line(f"[calib] caches ready: {built}, failed/skipped: {failed}")
    print_line(f"[calib] cache directory: {cache_dir}")
    return 0 if failed == 0 else 1


# %% Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build TensorRT INT8 calibration caches for every exported ONNX model.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild caches that already exist.",
    )
    args = parser.parse_args()
    sys.exit(main(force=bool(args.force)))
