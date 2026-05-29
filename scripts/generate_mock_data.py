"""
generate_mock_data.py
=====================
Phase 4 development utility — create tiny synthetic GeoTIFF fire events
under data/aligned_compressed/ so that build_tensors.py and
build_splits.py can be tested without downloading real data.

Creates N mock fire events (MOCK001, MOCK002, …) with small rasters
and a realistic band layout matching the Phase 3 output format.
Mock entries are appended to compression_report.csv as fire_complete=True.

Usage (from project root):

    python scripts/generate_mock_data.py              # 5 fires, 50×50 px
    python scripts/generate_mock_data.py --n-fires 3 --size 100
    python scripts/generate_mock_data.py --clean      # remove all mock events
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

DEFAULT_OUTPUT_DIR = Path("data/aligned_compressed")
REPORT_CSV = DEFAULT_OUTPUT_DIR / "compression_report.csv"
REPORT_FIELDS = [
    "fire_id", "source_file", "output_file",
    "original_size_mb", "compressed_size_mb", "reduction_percent",
    "verified", "fire_complete", "original_deleted", "error",
]

N_WEATHER_VARS = 6
WEATHER_VAR_NAMES = [
    "temperature_2m",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "surface_pressure",
    "total_precipitation",
]
MOCK_PREFIX = "MOCK"


def _make_transform(size: int) -> rasterio.Affine:
    return from_bounds(-120.0, 37.0, -119.5, 37.5, size, size)


def _write_tif(
    path: Path,
    data: np.ndarray,
    crs: CRS,
    transform: rasterio.Affine,
    dtype: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis]  # (1, H, W)
    count, height, width = data.shape
    with rasterio.open(
        path, "w",
        driver="GTiff",
        count=count,
        height=height,
        width=width,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data.astype(dtype))


def _create_mock_fire(
    fire_id: str,
    size: int,
    n_weather_days: int,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict:
    fire_dir = out_dir / fire_id
    crs = CRS.from_epsg(32610)  # UTM zone 10N — representative for US West
    transform = _make_transform(size)
    n_weather_bands = n_weather_days * N_WEATHER_VARS

    # Labels
    burned = (rng.random((size, size)) > 0.7).astype("uint8")
    severity = (rng.integers(0, 5, (size, size)) * burned).astype("uint8")
    _write_tif(fire_dir / "labels/burned_area_30m_aligned.tif", burned, crs, transform, "uint8")
    _write_tif(fire_dir / "labels/severity_30m_aligned.tif", severity, crs, transform, "uint8")

    # Terrain: 3 bands — elevation (m), slope (deg), aspect (deg)
    terrain = rng.random((3, size, size)).astype("float32")
    terrain[0] = terrain[0] * 2000 + 500   # elevation 500–2500 m
    terrain[1] = terrain[1] * 45           # slope 0–45 deg
    terrain[2] = terrain[2] * 360          # aspect 0–360 deg
    _write_tif(fire_dir / "terrain/terrain_30m_aligned.tif", terrain, crs, transform, "float32")

    # Weather: 6 variables × n_weather_days bands
    # Band ordering: [var0_day1, var1_day1, …, var5_day1, var0_day2, …]
    weather = rng.random((n_weather_bands, size, size)).astype("float32")
    for day in range(n_weather_days):
        b = day * N_WEATHER_VARS
        weather[b + 0] = weather[b + 0] * 30 + 5       # temperature_2m: 5–35 °C
        weather[b + 1] = weather[b + 1] * 80 + 10      # relative_humidity_pct: 10–90 %
        weather[b + 2] = weather[b + 2] * 15            # wind_speed_ms: 0–15 m/s
        weather[b + 3] = weather[b + 3] * 360           # wind_dir_deg: 0–360 °
        weather[b + 4] = weather[b + 4] * 5000 + 95000 # surface_pressure: 95–100 kPa
        weather[b + 5] = weather[b + 5] * 10            # total_precipitation: 0–10 mm
    _write_tif(fire_dir / "weather/weather_stacked_30m_aligned.tif", weather, crs, transform, "float32")

    # metadata.json  (mirrors Phase 3 format)
    meta = {
        "fire_id": fire_id,
        "reference_layer": "severity_30m_aligned.tif",
        "crs": crs.to_string(),
        "transform": list(transform)[:6],
        "width": size,
        "height": size,
        "weather_bands": n_weather_bands,
        "weather_days": n_weather_days,
        "weather_vars_per_day": N_WEATHER_VARS,
        "terrain_bands": 3,
        "mock": True,
    }
    (fire_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def _existing_fire_ids_in_report() -> set[str]:
    if not REPORT_CSV.exists():
        return set()
    with open(REPORT_CSV, newline="") as f:
        return {row["fire_id"] for row in csv.DictReader(f)}


def _append_to_report(fire_id: str, fire_dir: Path) -> None:
    if fire_id in _existing_fire_ids_in_report():
        return
    layers = [
        "labels/burned_area_30m_aligned.tif",
        "labels/severity_30m_aligned.tif",
        "terrain/terrain_30m_aligned.tif",
        "weather/weather_stacked_30m_aligned.tif",
    ]
    write_header = not REPORT_CSV.exists()
    with open(REPORT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        if write_header:
            writer.writeheader()
        for layer in layers:
            p = fire_dir / layer
            size_mb = round(p.stat().st_size / (1024 * 1024), 4) if p.exists() else 0.0
            writer.writerow({
                "fire_id": fire_id,
                "source_file": f"data/aligned/{fire_id}/{layer}",
                "output_file": str(fire_dir / layer),
                "original_size_mb": size_mb,
                "compressed_size_mb": size_mb,
                "reduction_percent": 0.0,
                "verified": True,
                "fire_complete": True,
                "original_deleted": False,
                "error": "",
            })


def _clean_mock_fires(out_dir: Path) -> None:
    existing = _existing_fire_ids_in_report()
    removed = []
    for fire_id in existing:
        if not fire_id.startswith(MOCK_PREFIX):
            continue
        d = out_dir / fire_id
        if d.exists():
            shutil.rmtree(d)
            removed.append(fire_id)

    if removed and REPORT_CSV.exists():
        rows = []
        with open(REPORT_CSV, newline="") as f:
            rows = [r for r in csv.DictReader(f) if not r["fire_id"].startswith(MOCK_PREFIX)]
        with open(REPORT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    if removed:
        print(f"Removed {len(removed)} mock fire(s): {removed}")
    else:
        print("No mock fires found to remove.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mock Phase 3 data for testing Phase 4 scripts."
    )
    parser.add_argument("--n-fires", type=int, default=5,
                        help="Number of mock fire events to create. (default: 5)")
    parser.add_argument("--size", type=int, default=50,
                        help="Pixel dimensions of each raster (NxN). (default: 50)")
    parser.add_argument("--weather-days", type=int, default=10,
                        help="Base number of weather days per fire. (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. (default: 42)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: data/aligned_compressed)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all MOCK* fire events and exit.")
    args = parser.parse_args()

    if args.clean:
        _clean_mock_fires(args.out_dir)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    created = 0
    for i in range(1, args.n_fires + 1):
        fire_id = f"{MOCK_PREFIX}{i:03d}"
        if (args.out_dir / fire_id).exists():
            print(f"  {fire_id}  already exists — skipping")
            continue
        n_days = max(5, args.weather_days + int(rng.integers(-2, 3)))
        _create_mock_fire(fire_id, args.size, n_days, rng, args.out_dir)
        _append_to_report(fire_id, args.out_dir / fire_id)
        print(f"  {fire_id}  created  ({args.size}×{args.size} px, {n_days} weather days, {n_days * N_WEATHER_VARS} bands)")
        created += 1

    if created == 0:
        print("Nothing to create — all mock fires already exist.")
        return

    print(f"\nMock data written to {args.out_dir}/")
    print("\nNext steps:")
    print("  python scripts/build_tensors.py --limit 5")
    print("  python scripts/build_splits.py")


if __name__ == "__main__":
    main()
