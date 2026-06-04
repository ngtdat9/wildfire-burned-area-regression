"""
organize_fire_mask_stacks.py
============================
After downloading stacked fire-mask GeoTIFFs from Google Drive into:

    data/raw/fire_mask_stacked/

Files often arrive flat, for example:

    data/raw/fire_mask_stacked/
    ├── 20778153__fire_mask__stacked.tif
    └── 20778186__fire_mask__stacked.tif

This script groups them by fire ID (text before the first ``__``):

    data/raw/fire_mask_stacked/
    ├── 20778153/
    │   └── 20778153__fire_mask__stacked.tif
    └── 20778186/
        └── 20778186__fire_mask__stacked.tif

Usage (from project root):

    python scripts/task1/phase_01_burn_labels/organize_fire_mask_stacks.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = Path("data/raw/fire_mask_stacked")
VALID_SUFFIXES = {".tif", ".tiff"}


def fire_id_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem
    if "__" not in stem:
        return None
    fire_id = stem.split("__", 1)[0]
    return fire_id if fire_id else None


def organize(input_dir: Path) -> int:
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1

    candidates = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    ]

    files_found = len(candidates)
    files_moved = 0
    files_skipped = 0
    folders_created = 0
    created: set[Path] = set()

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
            if dest_dir not in created:
                created.add(dest_dir)
                folders_created += 1

        dest = dest_dir / src.name
        if dest.exists():
            print(f"[WARN] Destination exists, skipping: {dest}")
            files_skipped += 1
            continue

        shutil.move(str(src), str(dest))
        print(f"  Moved: {src.name}  ->  {fire_id}/")
        files_moved += 1

    print(f"\n{'─' * 50}")
    print("Summary")
    print(f"  Files found (flat):   {files_found}")
    print(f"  Files moved:          {files_moved}")
    print(f"  Files skipped:        {files_skipped}")
    print(f"  Fire folders created: {folders_created}")
    print(f"{'─' * 50}\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize flat fire-mask stack GeoTIFFs by fire ID."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Download folder (default: {DEFAULT_INPUT_DIR}).",
    )
    args = parser.parse_args()
    sys.exit(organize(args.input_dir))


if __name__ == "__main__":
    main()
