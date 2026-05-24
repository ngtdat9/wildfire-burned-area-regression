"""
organize_environment_terrain.py
================================
After Phase 2 (environment & terrain extraction), download the exported
GeoTIFFs from Google Drive into:

    data/raw/environment_terrain/

Files often arrive in one flat folder, for example:

    data/raw/environment_terrain/
    ├── 20778153__terrain__static.tif
    ├── 20778153__weather__stacked.tif
    ├── 20778186__terrain__static.tif
    └── 20778186__weather__stacked.tif

This script groups them by fire ID:

    data/raw/environment_terrain/
    ├── 20778153/
    │   ├── 20778153__terrain__static.tif
    │   └── 20778153__weather__stacked.tif
    └── 20778186/
        ├── 20778186__terrain__static.tif
        └── 20778186__weather__stacked.tif

The fire ID is taken from the part of the filename before the first
double underscore (``__``), e.g.
``20778153__terrain__static.tif`` → ``20778153``.

Only files sitting **directly** in the input folder are moved. Files
already inside subfolders are left alone. If a destination file already
exists, the file is skipped (not overwritten).

Usage (from project root):

    python scripts/organize_environment_terrain.py

    python scripts/organize_environment_terrain.py \\
        --input-dir data/raw/environment_terrain
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = Path("data/raw/environment_terrain")
VALID_SUFFIXES = {".tif", ".tiff"}


def fire_id_from_filename(filename: str) -> str | None:
    """Return the fire ID (text before the first ``__``) or None if invalid."""
    stem = Path(filename).stem
    if "__" not in stem:
        return None
    fire_id = stem.split("__", 1)[0]
    return fire_id if fire_id else None


def organize(input_dir: Path) -> int:
    """
    Move flat GeoTIFFs in ``input_dir`` into per-fire subfolders.

    Returns 0 on success, non-zero if the input directory is missing.
    """
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1

    # Only files directly in the input folder (not in subfolders).
    candidates = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    ]

    files_found = len(candidates)
    files_moved = 0
    files_skipped = 0
    folders_created = 0
    created_folder_paths: set[Path] = set()

    print(f"\nInput directory: {input_dir.resolve()}")
    print(f"GeoTIFF files found (flat): {files_found}\n")

    for src in sorted(candidates):
        fire_id = fire_id_from_filename(src.name)
        if fire_id is None:
            print(f"[WARN] Skipping (no '__' in name): {src.name}")
            files_skipped += 1
            continue

        dest_dir = input_dir / fire_id
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            if dest_dir not in created_folder_paths:
                created_folder_paths.add(dest_dir)
                folders_created += 1

        dest = dest_dir / src.name
        if dest.exists():
            print(f"[WARN] Destination already exists, skipping: {dest}")
            files_skipped += 1
            continue

        shutil.move(str(src), str(dest))
        print(f"  Moved: {src.name}  ->  {fire_id}/")
        files_moved += 1

    print(f"\n{'─' * 50}")
    print("Summary")
    print(f"{'─' * 50}")
    print(f"  Files found (flat):     {files_found}")
    print(f"  Files moved:            {files_moved}")
    print(f"  Files skipped:          {files_skipped}")
    print(f"  Fire folders created:   {folders_created}")
    print(f"{'─' * 50}\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Organize flat Phase 2 environment/terrain GeoTIFFs into one "
            "subfolder per fire ID under data/raw/environment_terrain/."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing flat GeoTIFF downloads (default: {DEFAULT_INPUT_DIR}).",
    )
    args = parser.parse_args()
    sys.exit(organize(args.input_dir))


if __name__ == "__main__":
    main()
