"""
validate_aligned_fire_data.py
=============================
Validate aligned wildfire GeoTIFF folders under ``data/aligned_compressed/``.

Checks file presence, readability, spatial alignment, and pixel-value
sanity for burned area, severity, stacked fire mask, terrain, and weather.

Memory-safe: reads one band at a time; does not load full multi-band stacks.

Usage (from project root):

    python scripts/validate_aligned_fire_data.py --limit 5
    python scripts/validate_aligned_fire_data.py --fire-id 20778153
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

DEFAULT_INPUT_DIR = Path("data/aligned_compressed")
DEFAULT_OUTPUT = Path("data/aligned_compressed/aligned_data_validation_report.csv")

EXPECTED_FILES = {
    "burned_area": Path("labels/burned_area_30m_aligned.tif"),
    "severity": Path("labels/severity_30m_aligned.tif"),
    "fire_mask": Path("labels/fire_mask_stacked_30m_aligned.tif"),
    "terrain": Path("terrain/terrain_30m_aligned.tif"),
    "weather": Path("weather/weather_stacked_30m_aligned.tif"),
}

SEVERITY_MAX = 4
FIRE_MASK_FULL_BAND_CHECK_LIMIT = 100
TERRAIN_SAMPLE_BANDS = 3
WEATHER_SAMPLE_BANDS = 3

CSV_COLUMNS = [
    "fire_id",
    "all_files_exist",
    "all_files_readable",
    "same_crs",
    "same_transform",
    "same_shape",
    "same_bounds",
    "burned_area_valid",
    "burned_area_unique_values",
    "burned_area_burned_pixels",
    "severity_valid",
    "severity_unique_values",
    "severity_nonzero_pixels",
    "fire_mask_valid",
    "fire_mask_band_count",
    "fire_mask_first_band_pixels",
    "fire_mask_middle_band_pixels",
    "fire_mask_last_band_pixels",
    "fire_mask_non_decreasing_sample",
    "terrain_valid",
    "terrain_band_count",
    "weather_valid",
    "weather_band_count",
    "overall_valid",
    "error",
]


def list_fire_ids(
    input_dir: Path,
    fire_id: str | None,
    limit: int | None,
) -> list[str]:
    if fire_id:
        folder = input_dir / fire_id
        if not folder.is_dir():
            raise FileNotFoundError(f"Fire folder not found: {folder}")
        return [fire_id]

    ids = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
    if limit is not None:
        ids = ids[:limit]
    return ids


def format_unique(values: np.ndarray, max_items: int = 20) -> str:
    uniq = np.unique(values)
    if len(uniq) > max_items:
        head = ", ".join(str(int(v)) if np.issubdtype(uniq.dtype, np.integer) else str(v)
                         for v in uniq[:max_items])
        return f"{head}, ... ({len(uniq)} total)"
    return ", ".join(str(int(v)) if np.issubdtype(uniq.dtype, np.integer) else str(v)
                     for v in uniq)


def valid_mask(data: np.ndarray, nodata) -> np.ndarray:
    """True where pixel is valid data (excludes NaN/Inf and nodata)."""
    mask = np.isfinite(data)
    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            return mask
        mask &= data != nodata
    return mask


def read_band(path: Path, band_idx: int) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as src:
        data = src.read(band_idx)
        return data, src.nodata


def open_meta(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "count": src.count,
            "nodata": src.nodata,
        }


def compare_spatial(metas: list[dict]) -> dict[str, bool]:
    ref = metas[0]
    same_crs = all(m["crs"] == ref["crs"] for m in metas)
    same_transform = all(m["transform"] == ref["transform"] for m in metas)
    same_shape = all(
        m["width"] == ref["width"] and m["height"] == ref["height"] for m in metas
    )
    same_bounds = all(m["bounds"] == ref["bounds"] for m in metas)
    return {
        "same_crs": same_crs,
        "same_transform": same_transform,
        "same_shape": same_shape,
        "same_bounds": same_bounds,
    }


def check_burned_area(path: Path) -> tuple[bool, str, int, str]:
    data, nodata = read_band(path, 1)
    mask = valid_mask(data, nodata)
    if not mask.any():
        return False, "", 0, "no valid pixels"

    valid = data[mask]
    uniq = np.unique(valid)
    if not np.all(np.isin(uniq, [0, 1])):
        return False, format_unique(uniq), int(np.sum(valid == 1)), "values not in {0,1}"

    burned = int(np.sum(valid == 1))
    if burned == 0 and int(np.sum(valid == 0)) == 0:
        return False, format_unique(uniq), burned, "no valid 0/1 pixels"

    return True, format_unique(uniq), burned, ""


def check_severity(path: Path) -> tuple[bool, str, int, str]:
    data, nodata = read_band(path, 1)
    mask = valid_mask(data, nodata)
    if not mask.any():
        return False, "", 0, "no valid pixels"

    valid = data[mask]
    uniq = np.unique(valid)
    if valid.min() < 0 or valid.max() > SEVERITY_MAX:
        return False, format_unique(uniq), int(np.sum(valid != 0)), "values outside 0-4"

    nonzero = int(np.sum(valid != 0))
    return True, format_unique(uniq), nonzero, ""


def check_fire_mask(path: Path) -> tuple[bool, int, int, int, int, bool, str]:
    with rasterio.open(path) as src:
        band_count = src.count
        if band_count <= 1:
            return False, band_count, 0, 0, 0, False, "band count must be > 1"

        def burned_in_band(bidx: int) -> tuple[int, str]:
            data = src.read(bidx)
            nodata = src.nodata
            mask = valid_mask(data, nodata)
            if not mask.any():
                return 0, "no valid pixels"
            valid = data[mask]
            uniq = np.unique(valid)
            if not np.all(np.isin(uniq, [0, 1])):
                return int(np.sum(valid == 1)), f"band {bidx} values not in {{0,1}}"
            return int(np.sum(valid == 1)), ""

        first_idx = 1
        middle_idx = (band_count + 1) // 2
        last_idx = band_count

        first_px, err1 = burned_in_band(first_idx)
        if err1:
            return False, band_count, first_px, 0, 0, False, err1

        middle_px, err2 = burned_in_band(middle_idx)
        if err2:
            return False, band_count, first_px, middle_px, 0, False, err2

        last_px, err3 = burned_in_band(last_idx)
        if err3:
            return False, band_count, first_px, middle_px, last_px, False, err3

        if first_px == 0 and middle_px == 0 and last_px == 0:
            return False, band_count, first_px, middle_px, last_px, False, "no burned pixels in sample bands"

        if band_count <= FIRE_MASK_FULL_BAND_CHECK_LIMIT:
            counts = []
            for bidx in range(1, band_count + 1):
                px, err = burned_in_band(bidx)
                if err:
                    return False, band_count, first_px, middle_px, last_px, False, err
                counts.append(px)
            non_decreasing = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
        else:
            non_decreasing = first_px <= middle_px <= last_px

        if last_px < first_px:
            return False, band_count, first_px, middle_px, last_px, non_decreasing, "last band < first band burned count"

        if not non_decreasing:
            return False, band_count, first_px, middle_px, last_px, non_decreasing, "non-decreasing check failed"

        return True, band_count, first_px, middle_px, last_px, non_decreasing, ""


def check_multiband_numeric(
    path: Path,
    min_bands: int,
    sample_bands: int,
) -> tuple[bool, int, str]:
    with rasterio.open(path) as src:
        band_count = src.count
        if band_count < min_bands:
            return False, band_count, f"expected >= {min_bands} bands"

        n_sample = min(sample_bands, band_count)
        for bidx in range(1, n_sample + 1):
            data = src.read(bidx)
            nodata = src.nodata
            mask = valid_mask(data, nodata)
            if not mask.any():
                return False, band_count, f"band {bidx} has no valid pixels"
            valid = data[mask].astype(np.float64)
            if not np.isfinite(valid).all():
                return False, band_count, f"band {bidx} has non-finite values"

        return True, band_count, ""


def validate_fire(fire_id: str, input_dir: Path) -> dict:
    row = {col: "" for col in CSV_COLUMNS}
    row["fire_id"] = fire_id
    row["all_files_exist"] = False
    row["all_files_readable"] = False
    row["same_crs"] = False
    row["same_transform"] = False
    row["same_shape"] = False
    row["same_bounds"] = False
    row["burned_area_valid"] = False
    row["severity_valid"] = False
    row["fire_mask_valid"] = False
    row["fire_mask_non_decreasing_sample"] = False
    row["terrain_valid"] = False
    row["weather_valid"] = False
    row["overall_valid"] = False

    fire_dir = input_dir / fire_id
    paths = {key: fire_dir / rel for key, rel in EXPECTED_FILES.items()}

    missing = [key for key, path in paths.items() if not path.is_file()]
    row["all_files_exist"] = len(missing) == 0
    if missing:
        row["error"] = f"missing: {', '.join(missing)}"
        return row

    metas: dict[str, dict] = {}
    errors: list[str] = []

    for key, path in paths.items():
        try:
            metas[key] = open_meta(path)
        except Exception as exc:
            errors.append(f"{key} unreadable: {exc}")

    row["all_files_readable"] = len(metas) == len(paths)
    if not row["all_files_readable"]:
        row["error"] = "; ".join(errors) if errors else "one or more files failed to open"
        return row

    spatial = compare_spatial(list(metas.values()))
    row["same_crs"] = spatial["same_crs"]
    row["same_transform"] = spatial["same_transform"]
    row["same_shape"] = spatial["same_shape"]
    row["same_bounds"] = spatial["same_bounds"]

    if not all(spatial.values()):
        errors.append("spatial mismatch between layers")

    try:
        ok, uniq, burned, err = check_burned_area(paths["burned_area"])
        row["burned_area_valid"] = ok
        row["burned_area_unique_values"] = uniq
        row["burned_area_burned_pixels"] = burned
        if err:
            errors.append(f"burned_area: {err}")
    except Exception as exc:
        errors.append(f"burned_area: {exc}")

    try:
        ok, uniq, nonzero, err = check_severity(paths["severity"])
        row["severity_valid"] = ok
        row["severity_unique_values"] = uniq
        row["severity_nonzero_pixels"] = nonzero
        if err:
            errors.append(f"severity: {err}")
    except Exception as exc:
        errors.append(f"severity: {exc}")

    try:
        ok, bands, first_px, mid_px, last_px, non_dec, err = check_fire_mask(
            paths["fire_mask"]
        )
        row["fire_mask_valid"] = ok
        row["fire_mask_band_count"] = bands
        row["fire_mask_first_band_pixels"] = first_px
        row["fire_mask_middle_band_pixels"] = mid_px
        row["fire_mask_last_band_pixels"] = last_px
        row["fire_mask_non_decreasing_sample"] = non_dec
        if err:
            errors.append(f"fire_mask: {err}")
    except Exception as exc:
        errors.append(f"fire_mask: {exc}")

    try:
        ok, bands, err = check_multiband_numeric(
            paths["terrain"], min_bands=3, sample_bands=TERRAIN_SAMPLE_BANDS
        )
        row["terrain_valid"] = ok
        row["terrain_band_count"] = bands
        if err:
            errors.append(f"terrain: {err}")
    except Exception as exc:
        errors.append(f"terrain: {exc}")

    try:
        ok, bands, err = check_multiband_numeric(
            paths["weather"], min_bands=2, sample_bands=WEATHER_SAMPLE_BANDS
        )
        row["weather_valid"] = ok
        row["weather_band_count"] = bands
        if err:
            errors.append(f"weather: {err}")
    except Exception as exc:
        errors.append(f"weather: {exc}")

    row["overall_valid"] = (
        row["all_files_exist"]
        and row["all_files_readable"]
        and row["same_crs"]
        and row["same_transform"]
        and row["same_shape"]
        and row["same_bounds"]
        and row["burned_area_valid"]
        and row["severity_valid"]
        and row["fire_mask_valid"]
        and row["terrain_valid"]
        and row["weather_valid"]
    )
    row["error"] = "; ".join(errors)
    return row


def write_report(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(output_path, index=False)


def print_summary(rows: list[dict]) -> None:
    total = len(rows)
    valid = sum(1 for r in rows if r["overall_valid"])
    invalid = total - valid

    print(f"\n{'=' * 60}")
    print("Aligned fire data validation summary")
    print(f"{'=' * 60}")
    print(f"Total fires checked : {total}")
    print(f"Overall valid       : {valid}")
    print(f"Invalid             : {invalid}")

    invalid_rows = [r for r in rows if not r["overall_valid"]]
    if invalid_rows:
        print("\nFirst invalid fires (up to 10):")
        for r in invalid_rows[:10]:
            reason = r["error"] or "check flags in CSV"
            print(f"  * {r['fire_id']}: {reason}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate aligned wildfire GeoTIFF folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Aligned dataset root (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument("--fire-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV report path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        sys.exit(1)

    try:
        fire_ids = list_fire_ids(args.input_dir, args.fire_id, args.limit)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not fire_ids:
        print(f"[ERROR] No fire folders found under {args.input_dir}")
        sys.exit(1)

    print(f"\nValidating {len(fire_ids)} fire(s) under {args.input_dir.resolve()}")

    rows: list[dict] = []
    for fid in fire_ids:
        print(f"  Checking {fid}...", flush=True)
        try:
            rows.append(validate_fire(fid, args.input_dir))
        except Exception as exc:
            row = {col: "" for col in CSV_COLUMNS}
            row["fire_id"] = fid
            row["overall_valid"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            for flag in (
                "all_files_exist",
                "all_files_readable",
                "same_crs",
                "same_transform",
                "same_shape",
                "same_bounds",
                "burned_area_valid",
                "severity_valid",
                "fire_mask_valid",
                "fire_mask_non_decreasing_sample",
                "terrain_valid",
                "weather_valid",
            ):
                row[flag] = False
            rows.append(row)
            traceback.print_exc()

    write_report(rows, args.output)
    print_summary(rows)
    print(f"Report written: {args.output.resolve()}")

    sys.exit(0 if all(r["overall_valid"] for r in rows) else 1)


if __name__ == "__main__":
    main()
