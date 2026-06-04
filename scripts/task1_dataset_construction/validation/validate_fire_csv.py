"""
validate_fire_csv.py
====================
Small validation helper for the fire-event CSV that feeds Phases 1 and 2.

Checks that the CSV contains the required columns:
    Id, lat, lon, start_date, end_date

Warns (does not fail) when the optional `area` column is missing, and
when `area` is present it prints a small summary so you can sanity-check
that the values are in square metres (m²), as Phases 1 and 2 expect.

Usage (from project root):
    python scripts/task1/validation/validate_fire_csv.py data/raw/csv/fire_all.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["Id", "lat", "lon", "start_date", "end_date"]
OPTIONAL_COLUMNS = ["area"]


def _summarise_area(series: pd.Series) -> None:
    """Print a small report on the `area` column.

    Phases 1 and 2 assume `area` is in SQUARE METRES (m²). This helper
    prints the median in both m² and km² and flags any values that
    look implausible (negative, zero, or wildly large).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()

    print("\n`area` column summary (assumed unit: m²):")
    if valid.empty:
        print("  [WARN] No numeric values could be parsed from `area`.")
        return

    median_m2 = float(valid.median())
    min_m2 = float(valid.min())
    max_m2 = float(valid.max())
    n_missing = int(numeric.isna().sum())
    n_nonpos = int((valid <= 0).sum())

    print(f"  count    : {len(valid)} valid ({n_missing} missing/non-numeric)")
    print(f"  min      : {min_m2:,.0f} m²  ({min_m2/1e6:.4f} km²)")
    print(f"  median   : {median_m2:,.0f} m²  ({median_m2/1e6:.4f} km²)")
    print(f"  max      : {max_m2:,.0f} m²  ({max_m2/1e6:.2f} km²)")

    if n_nonpos:
        print(f"  [WARN] {n_nonpos} row(s) have area <= 0.")

    if 1e5 <= median_m2 <= 1e10:
        print("  [OK] median magnitude is consistent with m² (a typical "
              "fire ~0.1–10,000 km²).")
    else:
        print("  [WARN] median magnitude does NOT look like m². Re-check "
              "the unit of the `area` column.")


def validate(csv_path: Path) -> int:
    """Return 0 on success, non-zero on failure (suitable for sys.exit)."""
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        return 1

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return 1

    df.columns = [c.strip() for c in df.columns]

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]

    print(f"\nFile      : {csv_path}")
    print(f"Records   : {len(df)}")
    print(f"Columns   : {list(df.columns)}")

    if missing_required:
        print(f"\n[ERROR] Missing required columns: {missing_required}")
        return 1
    print("\n[OK] All required columns present.")

    for col in missing_optional:
        print(f"[WARN] Optional column missing: '{col}'")

    if "area" in df.columns:
        _summarise_area(df["area"])

    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the structure of a fire-event CSV before running Phase 1 or Phase 2."
    )
    parser.add_argument("csv", help="Path to the fire CSV (e.g. data/raw/csv/fire_all.csv).")
    args = parser.parse_args()

    sys.exit(validate(Path(args.csv)))


if __name__ == "__main__":
    main()
