"""
align_fire_event_layers.py
==========================
Phase 3 — Geospatial alignment.

For each fire event that is ready in the Phase 1 + Phase 2 inventory,
reproject every layer onto the **Landsat severity** reference grid
(30 m). This makes CRS, resolution, transform, width, height, and
bounds identical so layers can be stacked for machine learning later.

Input (per fire):
    data/raw/satellite_labels/<fire_id>/<fire_id>_Severity.tif      (reference)
    data/raw/satellite_labels/<fire_id>/<fire_id>_BurnedArea.tif
    data/raw/environment_terrain/<fire_id>/<fire_id>__terrain__static.tif
    data/raw/environment_terrain/<fire_id>/<fire_id>__weather__stacked.tif

Output (per fire):
    data/aligned/<fire_id>/labels/burned_area_30m_aligned.tif
    data/aligned/<fire_id>/labels/severity_30m_aligned.tif
    data/aligned/<fire_id>/terrain/terrain_30m_aligned.tif
    data/aligned/<fire_id>/weather/weather_stacked_30m_aligned.tif
    data/aligned/<fire_id>/metadata.json

Resampling:
    - Burned area, severity: nearest neighbour (class / mask data)
    - Terrain, weather: bilinear (continuous variables)

Usage (from project root):

    python scripts/task1/phase_03_alignment/align_fire_event_layers.py --limit 1
    python scripts/task1/phase_03_alignment/align_fire_event_layers.py --fire-id 20562846
    python scripts/task1/phase_03_alignment/align_fire_event_layers.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


DEFAULT_INVENTORY = Path("data/raw/csv/phase1_phase2_inventory.csv")
DEFAULT_LABELS_DIR = Path("data/raw/satellite_labels")
DEFAULT_ENV_DIR = Path("data/raw/environment_terrain")
DEFAULT_OUTPUT_DIR = Path("data/aligned")
SUMMARY_CSV_NAME = "alignment_summary.csv"

RESAMPLING = {
    "severity": Resampling.nearest,
    "burned_area": Resampling.nearest,
    "terrain": Resampling.bilinear,
    "weather_stacked": Resampling.bilinear,
}


def parse_bool(value: object) -> bool:
    """Parse True/False values from the inventory CSV."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_ready_fire_ids(
    inventory_path: Path,
    fire_id: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Return fire IDs from the inventory where ready_for_phase3 is True."""
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")

    ready: list[str] = []
    with inventory_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row["fire_id"]).strip()
            if not parse_bool(row.get("ready_for_phase3", False)):
                continue
            if fire_id is not None and fid != fire_id:
                continue
            ready.append(fid)

    if fire_id is not None and not ready:
        raise ValueError(
            f"Fire ID {fire_id!r} is not ready for Phase 3 "
            f"(missing files or not in inventory)."
        )

    if limit is not None:
        ready = ready[:limit]

    return ready


def input_paths(
    fire_id: str,
    labels_dir: Path,
    env_dir: Path,
) -> dict[str, Path]:
    """Build expected input paths for one fire."""
    return {
        "severity": labels_dir / fire_id / f"{fire_id}_Severity.tif",
        "burned_area": labels_dir / fire_id / f"{fire_id}_BurnedArea.tif",
        "terrain": env_dir / fire_id / f"{fire_id}__terrain__static.tif",
        "weather_stacked": env_dir / fire_id / f"{fire_id}__weather__stacked.tif",
    }


def output_paths(fire_id: str, output_dir: Path) -> dict[str, Path]:
    """Build output paths for one aligned fire."""
    fire_root = output_dir / fire_id
    return {
        "severity": fire_root / "labels" / "severity_30m_aligned.tif",
        "burned_area": fire_root / "labels" / "burned_area_30m_aligned.tif",
        "terrain": fire_root / "terrain" / "terrain_30m_aligned.tif",
        "weather_stacked": fire_root / "weather" / "weather_stacked_30m_aligned.tif",
        "metadata": fire_root / "metadata.json",
    }


def reference_profile_from_severity(severity_path: Path) -> dict:
    """Read the severity raster and build the target grid profile."""
    with rasterio.open(severity_path) as src:
        profile = src.profile.copy()
        profile.update(
            compress="deflate",
            tiled=True,
        )
        ref = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "count": src.count,
            "dtype": src.dtypes[0],
            "profile": profile,
            "descriptions": list(src.descriptions) if src.descriptions else None,
            "resolution": src.res,
            "shape": (src.height, src.width),
        }
    return ref


def write_reference_copy(
    src_path: Path,
    dst_path: Path,
    ref_profile: dict,
) -> None:
    """
    Write the severity layer to the aligned folder.

    Because severity *is* the reference grid, we copy its pixels onto
    the same grid with compression enabled.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        data = src.read()
        profile = ref_profile["profile"].copy()
        profile.update(
            count=src.count,
            dtype=src.dtypes[0],
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)
            if src.descriptions:
                dst.descriptions = src.descriptions


def reproject_to_reference(
    src_path: Path,
    dst_path: Path,
    ref: dict,
    resampling: Resampling,
) -> None:
    """Reproject all bands of ``src_path`` onto the reference grid."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        dst_count = src.count
        dst_dtype = src.dtypes[0]
        dst_data = np.zeros(
            (dst_count, ref["height"], ref["width"]),
            dtype=dst_dtype,
        )

        for band_idx in range(1, dst_count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst_data[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref["transform"],
                dst_crs=ref["crs"],
                resampling=resampling,
            )

        profile = ref["profile"].copy()
        profile.update(count=dst_count, dtype=dst_dtype)

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(dst_data)
            if src.descriptions:
                dst.descriptions = src.descriptions


def build_metadata(
    fire_id: str,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    ref: dict,
) -> dict:
    """Build the per-fire metadata.json content."""
    transform = ref["transform"]
    bounds = ref["bounds"]

    return {
        "fire_id": fire_id,
        "reference_file": str(inputs["severity"]),
        "target_crs": ref["crs"].to_string() if ref["crs"] else None,
        "target_transform": list(transform)[:6],
        "target_width": ref["width"],
        "target_height": ref["height"],
        "target_bounds": {
            "left": bounds.left,
            "bottom": bounds.bottom,
            "right": bounds.right,
            "top": bounds.top,
        },
        "target_resolution": ref["resolution"],
        "input_files": {key: str(path) for key, path in inputs.items()},
        "output_files": {
            key: str(path)
            for key, path in outputs.items()
            if key != "metadata"
        },
        "resampling": {
            "severity": "nearest",
            "burned_area": "nearest",
            "terrain": "bilinear",
            "weather_stacked": "bilinear",
        },
    }


def align_one_fire(
    fire_id: str,
    labels_dir: Path,
    env_dir: Path,
    output_dir: Path,
) -> dict:
    """
    Align all layers for one fire event.

    Returns a summary row dict for alignment_summary.csv.
    """
    inputs = input_paths(fire_id, labels_dir, env_dir)
    outputs = output_paths(fire_id, output_dir)

    row = {
        "fire_id": fire_id,
        "aligned_success": False,
        "error": "",
        "severity_output": str(outputs["severity"]),
        "burned_area_output": str(outputs["burned_area"]),
        "terrain_output": str(outputs["terrain"]),
        "weather_output": str(outputs["weather_stacked"]),
    }

    print(f"\n{'─' * 60}")
    print(f"Fire ID: {fire_id}")
    print("Input files:")
    for name, path in inputs.items():
        status = "OK" if path.is_file() else "MISSING"
        print(f"  [{status}] {name}: {path}")

    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")

    ref = reference_profile_from_severity(inputs["severity"])
    print("Reference grid (severity):")
    print(f"  CRS        : {ref['crs']}")
    print(f"  Resolution : {ref['resolution']}")
    print(f"  Shape      : {ref['shape']} (height, width)")

    # 1) Severity — reference copy
    write_reference_copy(inputs["severity"], outputs["severity"], ref)
    print(f"  Wrote severity -> {outputs['severity']}")

    # 2) Burned area — nearest neighbour
    reproject_to_reference(
        inputs["burned_area"],
        outputs["burned_area"],
        ref,
        RESAMPLING["burned_area"],
    )
    print(f"  Wrote burned area -> {outputs['burned_area']}")

    # 3) Terrain — bilinear
    reproject_to_reference(
        inputs["terrain"],
        outputs["terrain"],
        ref,
        RESAMPLING["terrain"],
    )
    print(f"  Wrote terrain -> {outputs['terrain']}")

    # 4) Weather stacked — bilinear (all bands)
    reproject_to_reference(
        inputs["weather_stacked"],
        outputs["weather_stacked"],
        ref,
        RESAMPLING["weather_stacked"],
    )
    print(f"  Wrote weather -> {outputs['weather_stacked']}")

    metadata = build_metadata(fire_id, inputs, outputs, ref)
    outputs["metadata"].parent.mkdir(parents=True, exist_ok=True)
    with outputs["metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote metadata -> {outputs['metadata']}")

    row["aligned_success"] = True
    return row


def write_summary_csv(summary_path: Path, rows: list[dict]) -> None:
    """Write the run summary CSV."""
    fieldnames = [
        "fire_id",
        "aligned_success",
        "error",
        "severity_output",
        "burned_area_output",
        "terrain_output",
        "weather_output",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_alignment(
    inventory_path: Path,
    labels_dir: Path,
    env_dir: Path,
    output_dir: Path,
    fire_id: str | None,
    limit: int | None,
) -> int:
    """Main alignment loop."""
    all_ready = load_ready_fire_ids(inventory_path, fire_id=fire_id, limit=None)
    fire_ids = load_ready_fire_ids(inventory_path, fire_id=fire_id, limit=limit)

    print(f"\n{'=' * 60}")
    print("Phase 3 — Geospatial Alignment")
    print(f"{'=' * 60}")
    print(f"Inventory      : {inventory_path.resolve()}")
    print(f"Labels dir     : {labels_dir.resolve()}")
    print(f"Environment dir: {env_dir.resolve()}")
    print(f"Output dir     : {output_dir.resolve()}")
    print(f"Ready fires    : {len(all_ready)}")
    print(f"To process     : {len(fire_ids)}")
    print(f"{'=' * 60}")

    summary_rows: list[dict] = []
    processed = 0
    failed = 0

    for fid in fire_ids:
        try:
            row = align_one_fire(fid, labels_dir, env_dir, output_dir)
            summary_rows.append(row)
            processed += 1
        except Exception as exc:
            failed += 1
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] Fire {fid} failed: {err_msg}")
            traceback.print_exc()
            summary_rows.append({
                "fire_id": fid,
                "aligned_success": False,
                "error": err_msg,
                "severity_output": "",
                "burned_area_output": "",
                "terrain_output": "",
                "weather_output": "",
            })

    summary_path = output_dir / SUMMARY_CSV_NAME
    write_summary_csv(summary_path, summary_rows)

    print(f"\n{'=' * 60}")
    print("Alignment summary")
    print(f"{'=' * 60}")
    print(f"Total ready fires (inventory): {len(all_ready)}")
    print(f"Processed this run            : {len(fire_ids)}")
    print(f"Succeeded                     : {processed}")
    print(f"Failed                        : {failed}")
    print(f"Summary CSV                   : {summary_path.resolve()}")
    print(f"{'=' * 60}\n")

    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3: align Phase 1 and Phase 2 GeoTIFFs onto the "
            "Landsat severity 30 m reference grid per fire event."
        )
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help=f"Phase 1+2 inventory CSV (default: {DEFAULT_INVENTORY}).",
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Aligned output root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N ready fire IDs.",
    )
    parser.add_argument(
        "--fire-id",
        type=str,
        default=None,
        help="Process only this fire ID (must be ready in inventory).",
    )
    args = parser.parse_args()

    try:
        code = run_alignment(
            inventory_path=args.inventory,
            labels_dir=args.labels_dir,
            env_dir=args.environment_dir,
            output_dir=args.output_dir,
            fire_id=args.fire_id,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    sys.exit(code)


if __name__ == "__main__":
    main()
