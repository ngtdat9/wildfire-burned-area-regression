"""
check_phase1_phase2_inventory.py
==================================
Before Phase 3 geospatial alignment, verify which fire events have all
required Phase 1 and Phase 2 outputs on disk.

Expected organized layout:

    data/raw/satellite_labels/<fire_id>/
        <fire_id>_BurnedArea.tif
        <fire_id>_Severity.tif

    data/raw/environment_terrain/<fire_id>/
        <fire_id>__terrain__static.tif
        <fire_id>__weather__stacked.tif

The script scans subfolder names under both roots, checks the four
required GeoTIFFs for each fire ID, writes a CSV report, and prints a
terminal summary.

Usage (from project root):

    python scripts/check_phase1_phase2_inventory.py

    python scripts/check_phase1_phase2_inventory.py \\
        --labels-dir data/raw/satellite_labels \\
        --environment-dir data/raw/environment_terrain \\
        --output data/raw/csv/phase1_phase2_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_LABELS_DIR = Path("data/raw/satellite_labels")
DEFAULT_ENV_DIR = Path("data/raw/environment_terrain")
DEFAULT_OUTPUT = Path("data/raw/csv/phase1_phase2_inventory.csv")

CSV_COLUMNS = [
    "fire_id",
    "burned_area_exists",
    "severity_exists",
    "terrain_exists",
    "weather_stacked_exists",
    "phase1_complete",
    "phase2_complete",
    "ready_for_phase3",
    "missing_files",
]


def collect_fire_ids(labels_dir: Path, env_dir: Path) -> list[str]:
    """Return sorted union of fire IDs from subfolder names in both roots."""
    ids: set[str] = set()

    for root in (labels_dir, env_dir):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir():
                ids.add(child.name)

    return sorted(ids)


def required_files(fire_id: str) -> dict[str, Path]:
    """Map logical file keys to relative filenames (within each fire folder)."""
    return {
        "burned_area": Path(f"{fire_id}_BurnedArea.tif"),
        "severity": Path(f"{fire_id}_Severity.tif"),
        "terrain": Path(f"{fire_id}__terrain__static.tif"),
        "weather_stacked": Path(f"{fire_id}__weather__stacked.tif"),
    }


def check_fire(
    fire_id: str,
    labels_dir: Path,
    env_dir: Path,
) -> dict[str, object]:
    """Check one fire ID and return one CSV row as a dict."""
    paths = required_files(fire_id)

    burned_area_path = labels_dir / fire_id / paths["burned_area"]
    severity_path = labels_dir / fire_id / paths["severity"]
    terrain_path = env_dir / fire_id / paths["terrain"]
    weather_path = env_dir / fire_id / paths["weather_stacked"]

    burned_area_exists = burned_area_path.is_file()
    severity_exists = severity_path.is_file()
    terrain_exists = terrain_path.is_file()
    weather_stacked_exists = weather_path.is_file()

    phase1_complete = burned_area_exists and severity_exists
    phase2_complete = terrain_exists and weather_stacked_exists
    ready_for_phase3 = phase1_complete and phase2_complete

    missing: list[str] = []
    if not burned_area_exists:
        missing.append(paths["burned_area"].name)
    if not severity_exists:
        missing.append(paths["severity"].name)
    if not terrain_exists:
        missing.append(paths["terrain"].name)
    if not weather_stacked_exists:
        missing.append(paths["weather_stacked"].name)

    return {
        "fire_id": fire_id,
        "burned_area_exists": burned_area_exists,
        "severity_exists": severity_exists,
        "terrain_exists": terrain_exists,
        "weather_stacked_exists": weather_stacked_exists,
        "phase1_complete": phase1_complete,
        "phase2_complete": phase2_complete,
        "ready_for_phase3": ready_for_phase3,
        "missing_files": ";".join(missing),
    }


def run_inventory(
    labels_dir: Path,
    env_dir: Path,
    output_path: Path,
) -> int:
    """Build the inventory CSV and print a summary. Returns exit code."""
    if not labels_dir.is_dir():
        print(f"[WARN] Labels directory not found: {labels_dir}")
    if not env_dir.is_dir():
        print(f"[WARN] Environment directory not found: {env_dir}")

    fire_ids = collect_fire_ids(labels_dir, env_dir)
    rows = [check_fire(fid, labels_dir, env_dir) for fid in fire_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    phase1_count = sum(1 for r in rows if r["phase1_complete"])
    phase2_count = sum(1 for r in rows if r["phase2_complete"])
    ready_count = sum(1 for r in rows if r["ready_for_phase3"])
    not_ready_count = len(rows) - ready_count
    not_ready = [r for r in rows if not r["ready_for_phase3"]]

    print(f"\n{'=' * 60}")
    print("Phase 1 + Phase 2 Inventory Check")
    print(f"{'=' * 60}")
    print(f"Labels dir       : {labels_dir.resolve()}")
    print(f"Environment dir  : {env_dir.resolve()}")
    print(f"Output CSV       : {output_path.resolve()}")
    print(f"{'─' * 60}")
    print(f"Total fire IDs found     : {len(rows)}")
    print(f"Phase 1 complete         : {phase1_count}")
    print(f"Phase 2 complete         : {phase2_count}")
    print(f"Ready for Phase 3        : {ready_count}")
    print(f"Not ready for Phase 3    : {not_ready_count}")
    print(f"{'─' * 60}")

    if not_ready:
        print("First not-ready fire IDs (up to 10):")
        for row in not_ready[:10]:
            missing = row["missing_files"] or "(none listed)"
            print(f"  {row['fire_id']}: missing {missing}")
    else:
        print("All fire IDs are ready for Phase 3.")

    print(f"{'=' * 60}\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check which fire events have all Phase 1 and Phase 2 GeoTIFFs "
            "before starting Phase 3 alignment."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help=f"Phase 1 root folder (default: {DEFAULT_LABELS_DIR}).",
    )
    parser.add_argument(
        "--environment-dir",
        type=Path,
        default=DEFAULT_ENV_DIR,
        help=f"Phase 2 root folder (default: {DEFAULT_ENV_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()
    sys.exit(run_inventory(args.labels_dir, args.environment_dir, args.output))


if __name__ == "__main__":
    main()
