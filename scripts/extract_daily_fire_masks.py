"""
extract_daily_fire_masks.py
===========================
Incremental dataset update — daily stacked fire masks from MODIS MCD64A1.

For each fire in the input CSV, export **one multi-band GeoTIFF** to
Google Drive where each band is the **cumulative burned mask** through
that calendar day (1 = burned by that date, 0 = not burned yet).

Band names:
    fire_mask_day001_YYYYMMDD
    fire_mask_day002_YYYYMMDD
    ...

Uses MODIS/061/MCD64A1 ``BurnDate`` (day-of-year when the pixel burned).

Mask rule for calendar day *D*:
    BurnDate > 0  AND  BurnDate <= day_of_year(D)

Same ROI logic as Phase 1 (``area`` in m², ``--buffer`` floor).

Usage (from project root):

    python scripts/extract_daily_fire_masks.py data/raw/csv/fire_all.csv --limit 1
    python scripts/extract_daily_fire_masks.py data/raw/csv/fire_all.csv --fire-id 20778153
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import ee
import pandas as pd

DEFAULT_CSV = Path("data/raw/csv/fire_all.csv")
DEFAULT_FOLDER = "Wildfire_FireMask_Stacked"
DEFAULT_BUFFER_M = 20_000
DEFAULT_PROJECT = "wildfire-area-regression"
MODIS_COLLECTION = "MODIS/061/MCD64A1"
EXPORT_SCALE_M = 500


def parse_date(date_str: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str!r}")


def to_ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def build_dates(start_date: str, end_date: str) -> list[str]:
    """Inclusive list of YYYY-MM-DD strings from start to end."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    dates: list[str] = []
    cur = start
    while cur <= end:
        dates.append(to_ymd(cur))
        cur += timedelta(days=1)
    return dates


def roi_from_row(row: dict, buffer_m: int) -> tuple[ee.Geometry, int]:
    lat = float(row["lat"])
    lon = float(row["lon"])
    try:
        area_m2 = float(row["area"])
        radius_m = max(buffer_m, int(math.sqrt(area_m2 / math.pi) * 1.5))
    except (TypeError, ValueError, KeyError):
        radius_m = buffer_m
    region = ee.Geometry.Point([lon, lat]).buffer(radius_m).bounds()
    return region, radius_m


def _band_suffix(day_index: int, date_str: str) -> str:
    compact = date_str.replace("-", "")
    return f"fire_mask_day{day_index:03d}_{compact}"


def cumulative_mask_for_date(
    modis_collection: ee.ImageCollection,
    region: ee.Geometry,
    date_str: str,
    day_index: int,
) -> ee.Image:
    """
    Cumulative burned mask through ``date_str``.

    Uses max BurnDate for the calendar year of ``date_str``, then
    BurnDate > 0 AND BurnDate <= day_of_year(date).
    """
    dt = parse_date(date_str)
    year = dt.year
    doy = dt.timetuple().tm_yday

    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    burn_date = (
        modis_collection
        .filterDate(year_start, year_end)
        .select("BurnDate")
        .max()
    )

    mask = (
        burn_date.gt(0)
        .And(burn_date.lte(doy))
        .rename(_band_suffix(day_index, date_str))
        .toUint8()
        .clip(region)
    )
    return mask


def build_stacked_fire_mask(
    dates: list[str],
    region: ee.Geometry,
    start_date: str,
    end_date: str,
) -> ee.Image:
    """Stack daily cumulative masks with ``ImageCollection.toBands()``."""
    t_start = (parse_date(start_date) - timedelta(days=15)).strftime("%Y-%m-%d")
    t_end = (parse_date(end_date) + timedelta(days=15)).strftime("%Y-%m-%d")

    modis = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterBounds(region)
        .filterDate(t_start, t_end)
    )

    daily_images = [
        cumulative_mask_for_date(modis, region, d, day_index)
        for day_index, d in enumerate(dates, start=1)
    ]

    stacked = ee.ImageCollection.fromImages(daily_images).toBands()
    stacked = stacked.toUint8().clip(region).set({
        "source": MODIS_COLLECTION,
        "role": "fire_mask_stacked",
        "weather_start_date": dates[0],
        "weather_end_date": dates[-1],
        "day_count": len(dates),
    })
    return stacked


def export_fire_mask(
    row: dict,
    buffer_m: int,
    drive_folder: str,
    dry_run: bool,
) -> dict:
    """Submit one GEE export task for a stacked fire-mask GeoTIFF."""
    fire_id = str(row["Id"])
    start_date = to_ymd(parse_date(row["start_date"]))
    end_date = to_ymd(parse_date(row["end_date"]))
    dates = build_dates(start_date, end_date)

    region, radius_m = roi_from_row(row, buffer_m)

    print(f"\n{'─' * 60}")
    print(f"Fire: {fire_id}  |  {row['lat']}, {row['lon']}")
    print(f"  Dates : {dates[0]} -> {dates[-1]} ({len(dates)} days)")
    print(f"  ROI   : radius ~ {radius_m / 1000:.1f} km")
    print(f"  Bands : ~ {len(dates)} (one cumulative mask per day)")

    desc = f"{fire_id}__fire_mask__stacked"
    result = {
        "fire_id": fire_id,
        "task_id": None,
        "days": len(dates),
        "drive_folder": drive_folder,
        "file_prefix": desc,
        "note": "",
    }

    if dry_run:
        print(f"  [DRY RUN] Would export: {desc}.tif")
        result["note"] = "dry-run"
        return result

    stacked = build_stacked_fire_mask(dates, region, start_date, end_date)
    task = ee.batch.Export.image.toDrive(
        image=stacked,
        description=desc,
        fileNamePrefix=desc,
        folder=drive_folder,
        region=region,
        scale=EXPORT_SCALE_M,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"  Export task: {task.id}")
    result["task_id"] = task.id
    return result


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    required = {"Id", "lat", "lon", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export stacked daily cumulative MODIS fire-mask GeoTIFFs "
            "(one band per day) to Google Drive."
        )
    )
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--buffer", type=int, default=DEFAULT_BUFFER_M)
    parser.add_argument("--fire-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    if not args.dry_run:
        try:
            ee.Initialize(project=args.project)
        except Exception as exc:
            print(f"[ERROR] GEE init failed: {exc}")
            print("Run: earthengine authenticate")
            sys.exit(1)

    df = load_dataframe(csv_path)
    if args.fire_id:
        df = df[df["Id"].astype(str) == str(args.fire_id)]
    if args.limit is not None:
        df = df.head(args.limit)

    if df.empty:
        print("[ERROR] No fire records to process.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Daily stacked fire-mask extraction (MODIS MCD64A1)")
    print(f"{'=' * 60}")
    print(f"CSV        : {csv_path}")
    print(f"Drive folder: {args.folder}")
    print(f"Records    : {len(df)}")
    print(f"Dry run    : {args.dry_run}")
    print(f"{'=' * 60}")

    tasks: list[dict] = []
    errors: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        try:
            info = export_fire_mask(
                row.to_dict(),
                buffer_m=args.buffer,
                drive_folder=args.folder,
                dry_run=args.dry_run,
            )
            tasks.append(info)
            if not args.dry_run:
                time.sleep(args.sleep)
        except Exception as exc:
            fid = str(row.get("Id", "?"))
            print(f"  [SKIP] {fid}: {exc}")
            errors.append((fid, str(exc)))

    summary_path = csv_path.parent / "fire_mask_stacked_export_summary.json"
    summary = {
        "phase": "fire_mask_stacked_extraction",
        "project": args.project,
        "input_csv": str(csv_path),
        "drive_folder": args.folder,
        "dry_run": args.dry_run,
        "submitted": len(tasks),
        "failed": errors,
        "tasks": tasks,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Submitted (or built) {len(tasks)}/{len(df)} records.")
    if errors:
        print("Failed:")
        for fid, msg in errors:
            print(f"  * {fid}: {msg}")
    print(f"Summary: {summary_path}")
    print("Monitor: https://code.earthengine.google.com/tasks")
    print(f"{'=' * 60}\n")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
