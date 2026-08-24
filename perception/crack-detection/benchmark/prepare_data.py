# %%
"""CrackSeg9k -> YOLO-format dataset preparation.

Converts the CrackSeg9k dataset (RGB images + binary PNG crack masks) into the
Ultralytics YOLO **segmentation** label format and writes a train/val/test
split under ``data/processed/``.

Assumed CrackSeg9k folder layout under ``data/raw/``
---------------------------------------------------
CrackSeg9k ships RGB images and masks in *sibling* directories::

    data/raw/crackseg9k/images/<name>.jpg         # the RGB image
    data/raw/crackseg9k/masks/Masks/<name>.png    # binary ground-truth mask

Only ``masks/Masks/`` is used - these are pure black/white PNGs where white
pixels mark cracks. The sibling ``masks/Heads/`` folder holds grayscale soft
probability maps which are **not** ground truth and are ignored entirely.

A mask is matched to its image purely by **shared filename stem**: images may
be ``.jpg`` while masks are always ``.png``, so the extension is stripped
before comparing. The split is stratified by the source subfolder beneath
``data/raw/`` so every category is proportionally represented.

Run with::

    python prepare_data.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

# Directory / file locations, all relative to this file.
_BASE_DIR: Path = Path(__file__).parent
_TRAIN_CONFIG_PATH: Path = _BASE_DIR / "train_config.yaml"
_RAW_DIR: Path = _BASE_DIR / "data" / "raw"
_PROCESSED_DIR: Path = _BASE_DIR / "data" / "processed"
_DATASET_YAML_PATH: Path = _PROCESSED_DIR / "dataset.yaml"

# Recognised RGB image extensions.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Name of the top-level mask directory; only its ``Masks/`` subfolder (binary
# ground truth) is used - the ``Heads/`` subfolder is ignored.
_MASK_DIR_NAME: str = "masks"
_GROUND_TRUTH_SUBDIR: str = "Masks"

# Contours with an area smaller than this (in pixels) are treated as noise.
_MIN_CONTOUR_AREA: float = 10.0

# Pixel intensity above which a mask pixel counts as "crack".
_MASK_THRESHOLD: int = 127

# Fixed RNG seed so the split is reproducible across runs.
_SPLIT_SEED: int = 42

# The three dataset splits, in fixed order.
_SPLITS: tuple[str, ...] = ("train", "val", "test")


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
def find_mask_for_image(image_path: Path) -> Path | None:
    """Locate the binary ground-truth mask that corresponds to an RGB image.

    The ground-truth mask is a ``.png`` in the ``masks/Masks/`` directory, a
    sibling of the ``images/`` directory, matched by shared filename stem. The
    grayscale probability maps in ``masks/Heads/`` are never consulted.

    Args:
        image_path: Path to the RGB image.

    Returns:
        The matching mask path, or ``None`` if no mask file is found.
    """
    # images/ and masks/ are siblings under the crackseg9k root, so the mask
    # directory is one level up from the image's own directory.
    masks_dir = image_path.parent.parent / _MASK_DIR_NAME / _GROUND_TRUTH_SUBDIR
    candidate = masks_dir / f"{image_path.stem}.png"
    return candidate if candidate.is_file() else None


# %%
def find_image_mask_pairs(
    raw_dir: Path,
    console: Console,
) -> tuple[list[tuple[Path, Path, str]], int]:
    """Recursively discover every (image, mask) pair under ``raw_dir``.

    Images located inside the ``masks/`` directory (both the ``Masks/`` and
    ``Heads/`` subfolders) are ignored so masks are never mistaken for input
    images.

    Args:
        raw_dir: Root directory holding the extracted CrackSeg9k dataset.
        console: ``rich`` console used to warn about images without a mask.

    Returns:
        A tuple of ``(pairs, missing_mask_count)`` where ``pairs`` is a list of
        ``(image_path, mask_path, subfolder)`` tuples and ``missing_mask_count``
        is the number of images that had no corresponding mask.
    """
    pairs: list[tuple[Path, Path, str]] = []
    missing_mask_count = 0

    for image_path in sorted(raw_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        # Skip files that live inside the mask directory - they are not images.
        if _MASK_DIR_NAME in image_path.parts:
            continue

        mask_path = find_mask_for_image(image_path)
        if mask_path is None:
            missing_mask_count += 1
            console.print(
                f"[yellow]warning:[/yellow] no mask found for {image_path} - skipping."
            )
            continue

        # Stratification key: the first path component below data/raw/.
        relative_parts = image_path.relative_to(raw_dir).parts
        subfolder = relative_parts[0] if len(relative_parts) > 1 else "root"
        pairs.append((image_path, mask_path, subfolder))

    return pairs, missing_mask_count


# %%
def is_blank_mask(mask: np.ndarray) -> bool:
    """Return whether a mask contains no crack pixels at all.

    Args:
        mask: Single-channel mask image as an ``np.ndarray``.

    Returns:
        ``True`` if every pixel is below the crack threshold (all background).
    """
    return bool(mask.max() <= _MASK_THRESHOLD)


# %%
def mask_to_polygons(
    mask: np.ndarray,
    min_area: float,
) -> tuple[list[list[float]], int]:
    """Convert a binary crack mask into normalised YOLO segmentation polygons.

    Each disconnected white region becomes its own polygon (one label row),
    so masks with multiple separate cracks yield multiple polygons. Contour
    coordinates are normalised to ``[0, 1]`` by dividing by image width/height.

    Args:
        mask: Single-channel mask image; white pixels mark cracks.
        min_area: Minimum contour area in pixels; smaller contours are dropped.

    Returns:
        A tuple of ``(polygons, small_contour_count)`` where ``polygons`` is a
        list of flat ``[x1, y1, x2, y2, ...]`` normalised coordinate lists and
        ``small_contour_count`` is how many contours were dropped as too small.
    """
    height, width = mask.shape[:2]
    # CrackSeg9k Masks/ are already clean binary PNGs (0 / 255), so the mask is
    # passed straight to findContours with no extra thresholding.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[list[float]] = []
    small_contour_count = 0

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            small_contour_count += 1
            continue
        # A valid polygon needs at least three distinct vertices.
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            small_contour_count += 1
            continue

        flat: list[float] = []
        for x, y in points:
            flat.append(round(float(np.clip(x / width, 0.0, 1.0)), 6))
            flat.append(round(float(np.clip(y / height, 0.0, 1.0)), 6))
        polygons.append(flat)

    return polygons, small_contour_count


# %%
def write_label_file(label_path: Path, polygons: list[list[float]]) -> None:
    """Write a YOLO segmentation label file (class index 0 for ``crack``).

    Args:
        label_path: Destination ``.txt`` path (parent dirs created if missing).
        polygons: List of flat normalised polygon coordinate lists.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "0 " + " ".join(f"{value:.6f}" for value in polygon)
        for polygon in polygons
    ]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# %%
def stratified_split(
    pairs: list[tuple[Path, Path, str]],
    ratios: dict[str, float],
) -> dict[str, list[tuple[Path, Path, str]]]:
    """Split image/mask pairs into train/val/test, stratified by subfolder.

    Each source subfolder is split independently using the given ratios, so
    every category keeps its proportional representation in all three splits.

    Args:
        pairs: List of ``(image_path, mask_path, subfolder)`` tuples.
        ratios: Mapping with ``train``, ``val`` and ``test`` ratios summing ~1.

    Returns:
        A dict mapping each split name to its list of pairs.
    """
    rng = random.Random(_SPLIT_SEED)
    result: dict[str, list[tuple[Path, Path, str]]] = {split: [] for split in _SPLITS}

    # Group pairs by their stratification subfolder.
    by_subfolder: dict[str, list[tuple[Path, Path, str]]] = {}
    for pair in pairs:
        by_subfolder.setdefault(pair[2], []).append(pair)

    for subfolder_pairs in by_subfolder.values():
        shuffled = list(subfolder_pairs)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_end = int(total * ratios["train"])
        val_end = train_end + int(total * ratios["val"])
        result["train"].extend(shuffled[:train_end])
        result["val"].extend(shuffled[train_end:val_end])
        result["test"].extend(shuffled[val_end:])

    return result


# %%
def write_dataset_yaml(processed_dir: Path, dataset_yaml_path: Path) -> None:
    """Write the Ultralytics ``dataset.yaml`` describing the prepared dataset.

    Args:
        processed_dir: Root directory holding ``images/`` and ``labels/``.
        dataset_yaml_path: Destination path for the dataset YAML file.
    """
    content: dict[str, Any] = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["crack"],
    }
    with dataset_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


# %%
def reset_processed_dirs(processed_dir: Path) -> None:
    """Empty and recreate the processed images/labels split directories.

    Args:
        processed_dir: Root directory holding ``images/`` and ``labels/``.
    """
    for kind in ("images", "labels"):
        for split in _SPLITS:
            target = processed_dir / kind / split
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


# %%
def print_summary(
    console: Console,
    total_images: int,
    missing_mask_count: int,
    blank_mask_count: int,
    tiny_only_count: int,
    small_contour_count: int,
    split_image_counts: dict[str, int],
    split_contour_counts: dict[str, int],
) -> None:
    """Print the dataset-preparation summary table using ``rich``.

    Args:
        console: ``rich`` console to print to.
        total_images: Total RGB images discovered with a matching mask.
        missing_mask_count: Images skipped because no mask file was found.
        blank_mask_count: Images skipped because the mask had no crack pixels.
        tiny_only_count: Images skipped because every contour was too small.
        small_contour_count: Individual contours dropped as below the area floor.
        split_image_counts: Mapping of split name to number of images written.
        split_contour_counts: Mapping of split name to number of polygons written.
    """
    table = Table(title="CrackSeg9k -> YOLO Preparation Summary", title_style="bold white")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    kept = sum(split_image_counts.values())
    table.add_row("Images with matching mask", str(total_images))
    table.add_row("Images kept (crack present)", str(kept))
    table.add_row("Skipped - no mask file", str(missing_mask_count))
    table.add_row("Skipped - blank mask", str(blank_mask_count))
    table.add_row("Skipped - only tiny contours", str(tiny_only_count))
    table.add_row("Tiny contours dropped (<10px area)", str(small_contour_count))
    for split in _SPLITS:
        table.add_row(
            f"  {split} images / polygons",
            f"{split_image_counts[split]} / {split_contour_counts[split]}",
        )
    console.print(table)


# %%
def prepare_dataset(console: Console) -> None:
    """Run the full CrackSeg9k -> YOLO conversion and split pipeline.

    Args:
        console: ``rich`` console used for all progress output.
    """
    config = load_train_config(_TRAIN_CONFIG_PATH)
    ratios: dict[str, float] = config["split"]

    if not _RAW_DIR.is_dir() or not any(_RAW_DIR.iterdir()):
        console.print(
            f"[red]No data found in {_RAW_DIR}.[/red] "
            "Download CrackSeg9k and extract it there first (see README)."
        )
        return

    console.print(f"[cyan]Scanning {_RAW_DIR} for image/mask pairs...[/cyan]")
    pairs, missing_mask_count = find_image_mask_pairs(_RAW_DIR, console)
    if not pairs:
        console.print("[red]No image/mask pairs discovered - nothing to do.[/red]")
        return
    console.print(f"[cyan]Found {len(pairs)} image/mask pair(s).[/cyan]")

    reset_processed_dirs(_PROCESSED_DIR)

    # First pass: convert every mask and keep only images with real cracks.
    blank_mask_count = 0
    tiny_only_count = 0
    small_contour_count = 0
    kept_pairs: list[tuple[Path, Path, str, list[list[float]]]] = []

    for image_path, mask_path, subfolder in pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            console.print(f"[yellow]warning:[/yellow] unreadable mask {mask_path} - skipping.")
            missing_mask_count += 1
            continue

        if is_blank_mask(mask):
            blank_mask_count += 1
            continue

        polygons, tiny = mask_to_polygons(mask, _MIN_CONTOUR_AREA)
        small_contour_count += tiny
        if not polygons:
            # The mask had white pixels but every region was below the floor.
            tiny_only_count += 1
            continue

        kept_pairs.append((image_path, mask_path, subfolder, polygons))

    if not kept_pairs:
        console.print("[red]Every mask was blank or noise - no usable data.[/red]")
        return

    # Second pass: split, then copy images and write label files.
    split_pairs = stratified_split(
        [(img, msk, sub) for img, msk, sub, _ in kept_pairs], ratios
    )
    polygons_by_image: dict[Path, list[list[float]]] = {
        img: polys for img, _, _, polys in kept_pairs
    }

    split_image_counts: dict[str, int] = {split: 0 for split in _SPLITS}
    split_contour_counts: dict[str, int] = {split: 0 for split in _SPLITS}

    for split, split_items in split_pairs.items():
        for image_path, _mask_path, _subfolder in split_items:
            polygons = polygons_by_image[image_path]
            destination_image = _PROCESSED_DIR / "images" / split / image_path.name
            destination_label = (
                _PROCESSED_DIR / "labels" / split / f"{image_path.stem}.txt"
            )
            shutil.copy2(image_path, destination_image)
            write_label_file(destination_label, polygons)
            split_image_counts[split] += 1
            split_contour_counts[split] += len(polygons)

    write_dataset_yaml(_PROCESSED_DIR, _DATASET_YAML_PATH)
    console.print(f"[green]dataset.yaml written to[/green] {_DATASET_YAML_PATH}")

    print_summary(
        console,
        total_images=len(pairs),
        missing_mask_count=missing_mask_count,
        blank_mask_count=blank_mask_count,
        tiny_only_count=tiny_only_count,
        small_contour_count=small_contour_count,
        split_image_counts=split_image_counts,
        split_contour_counts=split_contour_counts,
    )


# %%
def main() -> None:
    """Entry point: prepare the CrackSeg9k dataset for YOLO fine-tuning."""
    console = Console()
    console.print("[bold]CrackSeg9k -> YOLO Dataset Preparation[/bold]")
    prepare_dataset(console)
    console.print("[bold green]Data preparation complete.[/bold green]")


# %%
if __name__ == "__main__":
    main()
