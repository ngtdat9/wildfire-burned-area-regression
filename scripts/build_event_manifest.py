"""
build_event_manifest.py
=======================
Starter script that reads a fire CSV and writes a JSON manifest of all
fire events to:
    data/raw/csv/event_manifest.json

For each event the manifest stores:
    - fire_id
    - lat, lon
    - area (or null if missing)
    - start_date, end_date
    - status flags for the burn-label and environment-terrain outputs

The status flags are placeholders for now; they will be filled in by
future phases (e.g. once a GeoTIFF has been downloaded into
`data/raw/satellite_labels/` we can flip `burn_labels.exported` to True).

Usage (from project root):
    python scripts/build_event_manifest.py data/raw/csv/fire_all.csv

NOTE: This script does NOT perform any geospatial alignment — that is
the responsibility of Phase 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["Id", "lat", "lon", "start_date", "end_date"]
DEFAULT_OUTPUT = Path("data/raw/csv/event_manifest.json")


def _safe_value(row: dict, key: str):
    """Return row[key] or None if missing / NaN."""
    if key not in row:
        return None
    value = row[key]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_manifest(csv_path: Path, output_path: Path) -> int:
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] CSV is missing required columns: {missing}")
        print(f"        Found: {list(df.columns)}")
        return 1

    events = []
    for _, row in df.iterrows():
        row = row.to_dict()
        events.append({
            "fire_id": str(_safe_value(row, "Id")),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "area": _safe_value(row, "area"),
            "start_date": str(_safe_value(row, "start_date")),
            "end_date": str(_safe_value(row, "end_date")),
            "burn_labels": {
                "exported": False,
                "downloaded": False,
                "path": None,
            },
            "environment_terrain": {
                "weather_exported": False,
                "terrain_exported": False,
                "downloaded": False,
                "path": None,
            },
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"source_csv": str(csv_path), "events": events}, f, indent=2)

    print(f"[OK] Wrote {len(events)} events -> {output_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a JSON event manifest from the fire CSV."
    )
    parser.add_argument("csv", help="Path to fire CSV (e.g. data/raw/csv/fire_all.csv).")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})."
    )
    args = parser.parse_args()

    sys.exit(build_manifest(Path(args.csv), Path(args.output)))


if __name__ == "__main__":
    main()
