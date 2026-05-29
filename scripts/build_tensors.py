"""
build_tensors.py
================
Phase 4 — Tensor structuring & ML export.

For each complete fire event in data/aligned_compressed/<fire_id>/,
stack the aligned GeoTIFF layers into a single compressed .npz file
ready for machine learning.

Feature channels (15 total):
  Indices 0–2   terrain  — elevation_m, slope_deg, aspect_deg  (static)
  Indices 3–14  weather  — 6 ERA5 variables × 2 aggregations (mean, max)
                           aggregated across all weather days of the fire
                           Variables: temperature_2m, relative_humidity_pct,
                           wind_speed_ms, wind_dir_deg, surface_pressure,
                           total_precipitation

Labels (saved separately, not in features):
  burned_area  (H, W) uint8  — 0 = unburned, 1 = burned  (MODIS)
  severity     (H, W) uint8  — 0–4 dNBR class            (Landsat)

Output per fire:
  data/tensors/<fire_id>/tensor.npz     — compressed numpy arrays
  data/tensors/<fire_id>/band_info.json — channel names and descriptions

Global output:
  data/tensors/dataset_manifest.json   — inventory of all processed fires

Loading a tensor later:
  data = np.load("data/tensors/<fire_id>/tensor.npz")
  X         = data["features"]     # (H, W, 15) float32
  y_burned  = data["burned_area"]  # (H, W)     uint8
  y_sev     = data["severity"]     # (H, W)     uint8

Usage (from project root):

    # Process all fires with fire_complete=True in compression_report.csv
    python scripts/build_tensors.py

    # Single fire (useful for testing with mock data)
    python scripts/build_tensors.py --fire-id MOCK001

    # First N fires only
    python scripts/build_tensors.py --limit 5

    # Custom directories
    python scripts/build_tensors.py \\
        --aligned-dir data/aligned_compressed \\
        --tensor-dir  data/tensors
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rasterio

DEFAULT_ALIGNED_DIR = Path("data/aligned_compressed")
DEFAULT_TENSOR_DIR = Path("data/tensors")
REPORT_CSV = DEFAULT_ALIGNED_DIR / "compression_report.csv"
MANIFEST_NAME = "dataset_manifest.json"

N_WEATHER_VARS = 6
WEATHER_VAR_NAMES = [
    "temperature_2m",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "surface_pressure",
    "total_precipitation",
]
TERRAIN_BAND_NAMES = ["elevation_m", "slope_deg", "aspect_deg"]


class FireResult(NamedTuple):
    fire_id: str
    success: bool
    height: int
    width: int
    n_feature_bands: int
    n_weather_days: int
    error: str


def complete_fire_ids(report_csv: Path) -> list[str]:
    """Return fire IDs where every row in the report has fire_complete=True."""
    if not report_csv.exists():
        return []
    seen: dict[str, bool] = {}
    with open(report_csv, newline="") as f:
        for row in csv.DictReader(f):
            fid = row["fire_id"]
            ok = row.get("fire_complete", "").strip().lower() == "true"
            seen[fid] = seen.get(fid, True) and ok
    return [fid for fid, ok in seen.items() if ok]


def read_all_bands(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()  # (C, H, W)


def aggregate_weather(weather: np.ndarray) -> np.ndarray:
    """
    Reduce variable-length daily weather stack to a fixed-size summary.

    Band ordering from Phase 2 export (6 vars per day):
      [temp_d1, rh_d1, ws_d1, wd_d1, sp_d1, tp_d1,
       temp_d2, rh_d2, …, tp_dN]

    Aggregations: mean and max across all N days, for each variable.

    Input:  (N*6, H, W)
    Output: (12,  H, W)  — [var0_mean, var0_max, var1_mean, var1_max, …]
    """
    n_bands = weather.shape[0]
    if n_bands % N_WEATHER_VARS != 0:
        raise ValueError(
            f"Weather band count {n_bands} not divisible by {N_WEATHER_VARS}. "
            "Expected 6 variables per day."
        )
    agg: list[np.ndarray] = []
    for v in range(N_WEATHER_VARS):
        var_days = weather[v::N_WEATHER_VARS]          # (N_days, H, W)
        agg.append(var_days.mean(axis=0, keepdims=True))
        agg.append(var_days.max(axis=0, keepdims=True))
    return np.concatenate(agg, axis=0)                 # (12, H, W)


def build_band_info() -> list[dict]:
    bands: list[dict] = []
    for name in TERRAIN_BAND_NAMES:
        bands.append({"name": name, "source": "terrain", "aggregation": "static"})
    for var in WEATHER_VAR_NAMES:
        bands.append({"name": f"{var}_mean", "source": "weather", "aggregation": "mean_over_fire_duration"})
        bands.append({"name": f"{var}_max",  "source": "weather", "aggregation": "max_over_fire_duration"})
    return bands


def process_fire(fire_id: str, aligned_dir: Path, tensor_dir: Path) -> FireResult:
    fire_in  = aligned_dir / fire_id
    fire_out = tensor_dir  / fire_id

    burned_tif   = fire_in / "labels"  / "burned_area_30m_aligned.tif"
    severity_tif = fire_in / "labels"  / "severity_30m_aligned.tif"
    terrain_tif  = fire_in / "terrain" / "terrain_30m_aligned.tif"
    weather_tif  = fire_in / "weather" / "weather_stacked_30m_aligned.tif"

    for p in [burned_tif, severity_tif, terrain_tif, weather_tif]:
        if not p.exists():
            return FireResult(fire_id, False, 0, 0, 0, 0, f"missing: {p.name}")

    try:
        burned   = read_all_bands(burned_tif)[0].astype("uint8")    # (H, W)
        severity = read_all_bands(severity_tif)[0].astype("uint8")  # (H, W)
        terrain  = read_all_bands(terrain_tif).astype("float32")    # (3, H, W)
        weather  = read_all_bands(weather_tif).astype("float32")    # (N*6, H, W)

        n_weather_bands = weather.shape[0]
        n_days = n_weather_bands // N_WEATHER_VARS

        weather_agg = aggregate_weather(weather)                    # (12, H, W)
        features_chw = np.concatenate([terrain, weather_agg], axis=0)  # (15, H, W)
        features = np.moveaxis(features_chw, 0, -1)                # (H, W, 15)

        fire_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fire_out / "tensor.npz",
            features=features,
            burned_area=burned,
            severity=severity,
        )

        band_info = {
            "fire_id": fire_id,
            "n_feature_bands": features.shape[-1],
            "n_weather_days": n_days,
            "bands": build_band_info(),
        }
        (fire_out / "band_info.json").write_text(json.dumps(band_info, indent=2))

        h, w = burned.shape
        return FireResult(fire_id, True, h, w, features.shape[-1], n_days, "")

    except Exception as exc:
        traceback.print_exc()
        return FireResult(fire_id, False, 0, 0, 0, 0, str(exc))


def write_manifest(results: list[FireResult], tensor_dir: Path) -> None:
    fires = [
        {
            "fire_id":        r.fire_id,
            "success":        r.success,
            "height":         r.height,
            "width":          r.width,
            "n_feature_bands": r.n_feature_bands,
            "n_weather_days": r.n_weather_days,
            "error":          r.error,
        }
        for r in results
    ]
    manifest = {
        "n_fires_processed": len(results),
        "n_fires_success":   sum(r.success for r in results),
        "n_feature_bands":   len(build_band_info()),
        "feature_bands":     build_band_info(),
        "label_bands":       ["burned_area", "severity"],
        "tensor_dir":        str(tensor_dir),
        "fires":             fires,
    }
    (tensor_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: stack aligned GeoTIFFs into ML-ready tensors."
    )
    parser.add_argument("--fire-id",     help="Process a single fire event only.")
    parser.add_argument("--limit",       type=int, help="Cap on number of fires to process.")
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR,
                        help=f"Phase 3 output directory. (default: {DEFAULT_ALIGNED_DIR})")
    parser.add_argument("--tensor-dir",  type=Path, default=DEFAULT_TENSOR_DIR,
                        help=f"Tensor output directory. (default: {DEFAULT_TENSOR_DIR})")
    args = parser.parse_args()

    args.tensor_dir.mkdir(parents=True, exist_ok=True)

    if args.fire_id:
        fire_ids = [args.fire_id]
    else:
        fire_ids = complete_fire_ids(REPORT_CSV)
        if not fire_ids:
            print(
                f"No fire_complete=True entries found in {REPORT_CSV}.\n"
                "Generate mock data first:  python scripts/generate_mock_data.py",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.limit:
            fire_ids = fire_ids[: args.limit]

    print(f"Processing {len(fire_ids)} fire(s)  →  {args.tensor_dir}/")

    results: list[FireResult] = []
    for fire_id in fire_ids:
        r = process_fire(fire_id, args.aligned_dir, args.tensor_dir)
        status = "OK  " if r.success else "FAIL"
        detail = (
            f"{r.height}×{r.width} px  |  {r.n_feature_bands} feature bands  |  {r.n_weather_days} weather days"
            if r.success else r.error
        )
        print(f"  [{status}]  {fire_id}  {detail}")
        results.append(r)

    write_manifest(results, args.tensor_dir)

    n_ok   = sum(r.success for r in results)
    n_fail = len(results) - n_ok
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    print(f"Manifest written to {args.tensor_dir / MANIFEST_NAME}")
    if n_ok > 0:
        print("\nNext step:  python scripts/build_splits.py")


if __name__ == "__main__":
    main()
