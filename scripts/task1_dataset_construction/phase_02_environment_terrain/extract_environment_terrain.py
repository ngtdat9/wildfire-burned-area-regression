"""
Phase 2 — Environment & Terrain Extraction (Google Earth Engine)
================================================================
Project : wildfire-burned-area-regression
Script  : scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py
Role    : Student 2 (Weather & Terrain Extraction)
Status  : First version completed.

Purpose
-------
For every fire event in the input CSV this script exports:

  1. A static terrain GeoTIFF (NASA SRTM-derived elevation, slope, aspect)
     at 30 m resolution.
  2. One **stacked** ERA5-Land weather GeoTIFF per fire (default): all
     days from (start_date - lookback_days) through end_date are merged
     into a single multi-band image using ``ImageCollection.toBands()``.
     Optional legacy mode: one GeoTIFF per day (--daily-weather-exports).

Input CSV columns expected:
    Id, area, end_date, lat, lon, start_date   (extra columns are ignored)

Typical usage (from project root):
    earthengine authenticate                          # first-time only

    # Dry run (builds export manifest, no GEE calls):
    python scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py \
        data/raw/csv/fire_all.csv --limit 1 --dry-run

    # Live run (stacked weather — default, 2 GEE tasks per fire):
    python scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py \
        data/raw/csv/fire_all.csv \
        --project wildfire-area-regression \
        --folder Wildfire_Environment_Exports \
        --lookback-days 3 \
        --min-buffer 20000 \
        --weather-scale 9000 \
        --terrain-scale 30 \
        --limit 1

    # Legacy: one weather GeoTIFF per day (many GEE tasks):
    python scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py \
        data/raw/csv/fire_all.csv \
        --daily-weather-exports \
        ...

Dependencies:
    pip install -r requirements.txt

Outputs land in Google Drive under the requested folder. Once exported,
they are typically synced to:
    data/raw/environment_terrain/

------------------------------------------------------------------------------
ROI rule (current, after the 2026-05-23 fix):
  The `area` column in the CSV has been CONFIRMED to be in SQUARE
  METRES (m²). The ROI radius is derived as:

      radius_m = max(min_buffer_m, sqrt(area_m2 / pi) * 1.5)

  with --min-buffer (default 20 km) acting as a floor so small fires
  still get enough spatial context. If `area` is missing or
  unparsable, the script falls back to --min-buffer.

  This replaces an earlier formula that wrongly assumed `area` was in
  km² and produced absurd >10,000 km buffers (same bug as Phase 1).

TODO (future improvements — DO NOT implement here yet):
  1. ERA5-Land default scale is 9000 m and is NOT aligned to the
     Landsat 30 m grid yet. Alignment is a separate phase (Role 3).
  2. Terrain is exported at 30 m but still must be reprojected to the
     common reference grid in Phase 3.
  3. Stacked weather uses ``--weather-scale 9000`` (ERA5 native). Do not
     use 30 m for weather until Phase 3 alignment.
  4. Alignment / harmonisation across layers will be handled by Role 3.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import ee
import pandas as pd


ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"

# Two lists kept in 1:1 order:
#   ERA5_BANDS_SRC -> the raw band names as exposed by the ECMWF
#                      ERA5_LAND/DAILY_AGGR collection. (As of the
#                      current dataset revision, precipitation is named
#                      `total_precipitation_sum`, NOT `total_precipitation`.)
#   ERA5_BANDS_OUT -> the renamed band names we want in the output
#                      GeoTIFF. Keeping these stable (no `_sum` suffix)
#                      means downstream code, notebooks, and docs do not
#                      need to change when the upstream collection
#                      renames bands.
# `.select(ERA5_BANDS_SRC, ERA5_BANDS_OUT)` does the rename in one step.
ERA5_BANDS_SRC = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_pressure",
    "total_precipitation_sum",
]
ERA5_BANDS_OUT = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_pressure",
    "total_precipitation",
]
SRTM_COLLECTION = "USGS/SRTMGL1_003"

DEFAULT_LOOKBACK_DAYS = 3
DEFAULT_MIN_BUFFER_M = 20_000       # fallback minimum ROI radius (20 km)
DEFAULT_WEATHER_SCALE_M = 9000      # ERA5 native-ish; set to 30 for 30 m export
DEFAULT_TERRAIN_SCALE_M = 30        # SRTM native
DEFAULT_SLEEP_S = 1.0

# Weather variables written into each daily slice (before day/date suffix).
WEATHER_EXPORT_BANDS = [
    "temperature_2m",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "surface_pressure",
    "total_precipitation",
]


# ─────────────────────────────────────────────
# Date parsing
# ─────────────────────────────────────────────
def parse_date(date_str: str) -> datetime:
    """Accept YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY, or DD-MM-YYYY."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: '{date_str}'")


def to_ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def build_dates(start_date: str, end_date: str, lookback_days: int) -> list[str]:
    """Return the inclusive list of dates from (start - lookback) to end."""
    start_dt = parse_date(start_date) - timedelta(days=lookback_days)
    end_dt = parse_date(end_date)
    return [to_ymd(d) for d in pd.date_range(start_dt, end_dt, freq="D")]


# ─────────────────────────────────────────────
# ROI + CRS helpers
# ─────────────────────────────────────────────
def roi_from_point_area(lat: float, lon: float, area_m2, min_buffer_m: int) -> tuple[ee.Geometry, int]:
    """
    Build an ROI from (lat, lon) with a circular buffer whose radius is
    derived from the reported fire area.

    `area_m2` is the value of the CSV `area` column, which has been
    confirmed to be in SQUARE METRES (m²). The radius is the
    equivalent-circle radius * 1.5, with `min_buffer_m` acting as a
    floor.
    """
    radius_m = min_buffer_m
    if area_m2 is not None:
        try:
            area_m2 = float(area_m2)
            radius_m = max(min_buffer_m, int(math.sqrt(area_m2 / math.pi) * 1.5))
        except Exception:
            radius_m = min_buffer_m

    region = ee.Geometry.Point([lon, lat]).buffer(radius_m).bounds()
    return region, radius_m


def pick_utm_zone_from_lon(lon: float, lat: float) -> str:
    """Generic UTM zone selection from (lon, lat); northern/southern EPSG."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return f"EPSG:{epsg}"


# ─────────────────────────────────────────────
# Weather derivations
# ─────────────────────────────────────────────
def compute_wind_speed_direction(img: ee.Image) -> ee.Image:
    u = img.select("u_component_of_wind_10m")
    v = img.select("v_component_of_wind_10m")
    wind_speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed_ms")
    wind_dir = (
        u.multiply(-1)
         .atan2(v.multiply(-1))
         .multiply(180.0 / 3.141592653589793)
         .mod(360)
         .rename("wind_dir_deg")
    )
    return img.addBands([wind_speed, wind_dir])


def compute_relative_humidity(img: ee.Image) -> ee.Image:
    K_TO_C = 273.15
    temp_c = img.select("temperature_2m").subtract(K_TO_C)
    dewp_c = img.select("dewpoint_temperature_2m").subtract(K_TO_C)

    A, B = 17.625, 243.04
    gamma_temp = temp_c.multiply(A).divide(temp_c.add(B))
    gamma_dewp = dewp_c.multiply(A).divide(dewp_c.add(B))

    rh = gamma_dewp.exp().divide(gamma_temp.exp()).multiply(100).rename("relative_humidity_pct")
    return img.addBands(rh)


def extract_daily_weather(date_str: str, region: ee.Geometry, utm_crs: str) -> ee.Image:
    """Return an ee.Image for one calendar day, clipped to the ROI."""
    next_day = to_ymd(parse_date(date_str) + timedelta(days=1))

    col = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(date_str, next_day)
        .filterBounds(region)
        # Select the raw bands by their collection names, then rename
        # them in the same step so downstream logic (the `desired` list
        # below, output filenames, notebooks, docs) keeps using the
        # cleaner names like `total_precipitation` instead of
        # `total_precipitation_sum`.
        .select(ERA5_BANDS_SRC, ERA5_BANDS_OUT)
    )
    img = col.median()
    img = compute_wind_speed_direction(img)
    img = compute_relative_humidity(img)

    desired = ee.List(WEATHER_EXPORT_BANDS)

    def _accumulate(b, acc):
        acc = ee.List(acc)
        return ee.Algorithms.If(img.bandNames().contains(b), acc.add(b), acc)

    available = ee.List(desired.iterate(_accumulate, ee.List([])))

    out = (
        img.select(available)
        # Cast to Float32 so every band has the SAME dtype before export.
        # Earth Engine requires all bands in a single GeoTIFF export to
        # share a dtype; ERA5 bands plus the derived RH/wind bands can
        # otherwise mix dtypes and trigger:
        #   "Exported bands must have compatible data types"
        .toFloat()
        .clip(region)
        .set({
            "system:time_start": ee.Date(date_str).millis(),
            "date": date_str,
            "source": "ERA5-Land Daily AGGR",
            "crs_target": utm_crs,
            "role": "environment_weather",
        })
    )
    return out


def _rename_daily_weather_bands(img: ee.Image, day_index: int, date_str: str) -> ee.Image:
    """Append ``_dayNNN_YYYYMMDD`` to each band name (server-side)."""
    suffix = f"_day{day_index:03d}_{date_str.replace('-', '')}"
    new_names = img.bandNames().map(
        lambda band: ee.String(band).cat(suffix)
    )
    return img.rename(new_names)


def build_stacked_weather_image(
    dates: list[str],
    region: ee.Geometry,
    utm_crs: str,
) -> ee.Image:
    """
    Stack daily ERA5 weather into one multi-band ``ee.Image``.

    For each date in ``dates``:
      1. Call ``extract_daily_weather``.
      2. Rename bands to include day index + compact date
         (e.g. ``temperature_2m_day001_20170707``).
      3. Merge all days with ``ImageCollection.toBands()``.

    GEE ``toBands()`` may prefix band names with the image index (``0_``,
    ``1_``, …). The day index and date are still embedded in each name.
    """
    if not dates:
        raise ValueError("Cannot stack weather: date list is empty.")

    daily_images: list[ee.Image] = []
    for day_index, date_str in enumerate(dates, start=1):
        day_img = extract_daily_weather(date_str, region, utm_crs)
        day_img = _rename_daily_weather_bands(day_img, day_index, date_str)
        daily_images.append(day_img)

    stacked = ee.ImageCollection.fromImages(daily_images).toBands()
    stacked = stacked.toFloat().clip(region).set({
        "source": "ERA5-Land Daily AGGR (stacked)",
        "crs_target": utm_crs,
        "role": "environment_weather_stacked",
        "weather_start_date": dates[0],
        "weather_end_date": dates[-1],
        "weather_day_count": len(dates),
        "weather_vars_per_day": len(WEATHER_EXPORT_BANDS),
    })
    return stacked


def extract_terrain(region: ee.Geometry, utm_crs: str) -> ee.Image:
    """Static SRTM-derived terrain: elevation, slope, aspect (30 m)."""
    srtm = ee.Image(SRTM_COLLECTION)
    terrain = ee.Terrain.products(srtm)
    return (
        terrain.select(["elevation", "slope", "aspect"])
        # Cast to Float32 so elevation (Int16 from SRTM) and the
        # computed slope/aspect bands share a single dtype. Without
        # this, the export task fails with:
        #   "Exported bands must have compatible data types; found
        #    inconsistent types: Int16 and Byte"
        .toFloat()
        .clip(region)
        .set({
            "source": "NASA SRTM 30m (USGS/SRTMGL1_003)",
            "crs_target": utm_crs,
            "role": "environment_terrain",
        })
    )


# ─────────────────────────────────────────────
# Export helper (reliable region coords)
# ─────────────────────────────────────────────
def export_to_drive(image: ee.Image, description: str, prefix: str,
                    folder: str, region: ee.Geometry, scale: int, crs: str) -> ee.batch.Task:
    region_coords = region.coordinates().getInfo()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=prefix,
        region=region_coords,
        scale=scale,
        crs=crs,
        fileFormat="GeoTIFF",
        maxPixels=1e13,
    )
    task.start()
    return task


def export_fire(row: dict, folder: str, lookback_days: int, min_buffer_m: int,
                weather_scale_m: int, terrain_scale_m: int,
                sleep_s: float, dry_run: bool,
                daily_weather_exports: bool) -> tuple[list[dict], dict]:
    """
    Run the per-event Phase 2 pipeline.

    Returns
    -------
    tasks : flat task list (for debugging / legacy summary field)
    event : structured record for this fire (terrain + weather metadata)
    """
    fire_id = str(row["Id"])
    lat = float(row["lat"])
    lon = float(row["lon"])
    start_date = str(row["start_date"])
    end_date = str(row["end_date"])
    area_m2 = row.get("area", None)

    weather_mode = "daily" if daily_weather_exports else "stacked"
    event: dict = {
        "fire_id": fire_id,
        "weather_mode": weather_mode,
        "terrain_task_id": None,
        "weather_task_id": None,
        "weather_days": None,
        "weather_start_date": None,
        "weather_end_date": None,
        "weather_band_count_expected": None,
    }
    tasks: list[dict] = []

    # Dry-run: avoid any ee.* calls but use the SAME ROI rule as the live path.
    if dry_run:
        radius_m = min_buffer_m
        if area_m2 is not None:
            try:
                area_m2 = float(area_m2)
                radius_m = max(min_buffer_m, int(math.sqrt(area_m2 / math.pi) * 1.5))
            except Exception:
                radius_m = min_buffer_m

        utm_crs = pick_utm_zone_from_lon(lon, lat)
        dates = build_dates(start_date, end_date, lookback_days=lookback_days)
        area_str = (
            f"area = {area_m2:,.0f} m² ≈ {area_m2/1e6:.2f} km²"
            if isinstance(area_m2, float)
            else "no `area` reported"
        )
        print(f"\n{'─'*60}")
        print(f"Fire: {fire_id}  |  {lat:.4f}, {lon:.4f}")
        print(f"  Dates  : {dates[0]} -> {dates[-1]} ({len(dates)} days incl. lookback)")
        print(f"  ROI    : radius ~ {radius_m/1000:.1f} km  ({area_str})")
        print(f"  CRS    : {utm_crs}")
        print(f"  Folder : {folder}")
        print(f"  Weather: {weather_mode}")
        print(f"  DryRun : {dry_run}")

        print("  [DRY RUN] Terrain manifest created.")
        tasks.append({
            "Id": fire_id, "layer": "terrain", "date": "static",
            "task_id": None, "note": "dry-run",
        })
        event["terrain_task_id"] = None
        event["weather_days"] = len(dates)
        event["weather_start_date"] = dates[0]
        event["weather_end_date"] = dates[-1]
        event["weather_band_count_expected"] = len(dates) * len(WEATHER_EXPORT_BANDS)

        if daily_weather_exports:
            for d in dates:
                print(f"  [DRY RUN] Daily weather manifest: {d}")
                tasks.append({
                    "Id": fire_id, "layer": "weather", "date": d,
                    "task_id": None, "note": "dry-run",
                })
        else:
            print(f"  [DRY RUN] Stacked weather manifest: {fire_id}__weather__stacked")
            tasks.append({
                "Id": fire_id, "layer": "weather_stacked", "date": "stacked",
                "task_id": None, "note": "dry-run",
            })
        return tasks, event

    # Live path: build ee geometries and perform exports.
    region, radius_m = roi_from_point_area(lat, lon, area_m2, min_buffer_m=min_buffer_m)
    utm_crs = pick_utm_zone_from_lon(lon, lat)
    dates = build_dates(start_date, end_date, lookback_days=lookback_days)

    try:
        area_val = float(area_m2) if area_m2 is not None else None
    except (TypeError, ValueError):
        area_val = None
    area_str = (
        f"area = {area_val:,.0f} m² ≈ {area_val/1e6:.2f} km²"
        if area_val is not None
        else "no `area` reported"
    )

    print(f"\n{'─'*60}")
    print(f"Fire: {fire_id}  |  {lat:.4f}, {lon:.4f}")
    print(f"  Dates  : {dates[0]} -> {dates[-1]} ({len(dates)} days incl. lookback)")
    print(f"  ROI    : radius ~ {radius_m/1000:.1f} km  ({area_str})")
    print(f"  CRS    : {utm_crs}")
    print(f"  Folder : {folder}")
    print(f"  Weather: {weather_mode}")
    print(f"  DryRun : {dry_run}")

    event["weather_days"] = len(dates)
    event["weather_start_date"] = dates[0]
    event["weather_end_date"] = dates[-1]
    event["weather_band_count_expected"] = len(dates) * len(WEATHER_EXPORT_BANDS)

    # Terrain (once per fire).
    terrain_img = extract_terrain(region, utm_crs)
    desc = f"{fire_id}__terrain__static"
    t = export_to_drive(terrain_img, desc, desc, folder, region, terrain_scale_m, utm_crs)
    print(f"   Terrain task: {t.id}")
    tasks.append({"Id": fire_id, "layer": "terrain", "date": "static", "task_id": t.id})
    event["terrain_task_id"] = t.id

    if daily_weather_exports:
        # Legacy: one export task per calendar day.
        for d in dates:
            try:
                wimg = extract_daily_weather(d, region, utm_crs)
                desc = f"{fire_id}__weather__{d}"
                t = export_to_drive(wimg, desc, desc, folder, region, weather_scale_m, utm_crs)
                print(f"  Weather task: {t.id} ({d})")
                tasks.append({"Id": fire_id, "layer": "weather", "date": d, "task_id": t.id})
                time.sleep(sleep_s)
            except Exception as e:
                print(f"  [SKIP] Weather failed {d}: {e}")
                tasks.append({
                    "Id": fire_id, "layer": "weather", "date": d,
                    "task_id": None, "error": str(e),
                })
    else:
        # Default: one stacked multi-band weather GeoTIFF per fire.
        try:
            stacked = build_stacked_weather_image(dates, region, utm_crs)
            desc = f"{fire_id}__weather__stacked"
            t = export_to_drive(
                stacked, desc, desc, folder, region, weather_scale_m, utm_crs,
            )
            print(f"  Stacked weather task: {t.id} "
                  f"({len(dates)} days x {len(WEATHER_EXPORT_BANDS)} vars "
                  f"≈ {event['weather_band_count_expected']} bands)")
            tasks.append({
                "Id": fire_id, "layer": "weather_stacked", "date": "stacked",
                "task_id": t.id,
            })
            event["weather_task_id"] = t.id
        except Exception as e:
            print(f"  [SKIP] Stacked weather failed: {e}")
            tasks.append({
                "Id": fire_id, "layer": "weather_stacked", "date": "stacked",
                "task_id": None, "error": str(e),
            })

    return tasks, event


def main(csv_path: str, project: str, folder: str, lookback_days: int,
         min_buffer_m: int, weather_scale_m: int, terrain_scale_m: int,
         sleep_s: float, dry_run: bool, limit: int | None,
         daily_weather_exports: bool) -> None:
    """Phase 2 entrypoint."""
    csv_path = str(Path(csv_path))

    # Init EE (skip for dry-run so unauthenticated checks still work).
    if not dry_run:
        try:
            ee.Initialize(project=project)
        except Exception as e:
            print(f"[ERROR] GEE init failed: {e}")
            print("Run `earthengine authenticate` first and ensure you have access to the project.")
            sys.exit(1)
    else:
        print("[DRY RUN] Skipping Earth Engine initialization.")

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    required = {"Id", "lat", "lon", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] CSV missing columns: {missing}")
        print(f"Found: {list(df.columns)}")
        sys.exit(1)

    if limit is not None:
        df = df.head(limit)

    weather_mode = "daily" if daily_weather_exports else "stacked"
    print(f"\n{'='*60}")
    print("Phase 2 - Environment & Terrain Extraction (GEE)")
    print(f"{'='*60}")
    print(f"Project      : {project}")
    print(f"Input        : {csv_path}")
    print(f"Folder       : {folder}")
    print(f"Records      : {len(df)}")
    print(f"Weather mode : {weather_mode}")
    print(f"Dry run      : {dry_run}")
    print(f"{'='*60}\n")

    all_tasks = []
    all_events = []
    errors = []

    for _, row in df.iterrows():
        try:
            tasks, event = export_fire(
                row.to_dict(),
                folder=folder,
                lookback_days=lookback_days,
                min_buffer_m=min_buffer_m,
                weather_scale_m=weather_scale_m,
                terrain_scale_m=terrain_scale_m,
                sleep_s=sleep_s,
                dry_run=dry_run,
                daily_weather_exports=daily_weather_exports,
            )
            all_tasks.extend(tasks)
            all_events.append(event)
        except Exception as e:
            fid = row.get("Id", "?")
            print(f"[SKIP] Fire {fid} failed: {e}")
            errors.append((fid, str(e)))

    summary = {
        "phase": "02_environment_terrain_extraction",
        "project": project,
        "input_csv": csv_path,
        "drive_folder": folder,
        "dry_run": dry_run,
        "weather_mode": weather_mode,
        "lookback_days": lookback_days,
        "min_buffer_m": min_buffer_m,
        "weather_scale_m": weather_scale_m,
        "terrain_scale_m": terrain_scale_m,
        "events": all_events,
        "tasks": all_tasks,
        "failed": errors,
    }

    out_summary = csv_path.replace(".csv", "_environment_export_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Submitted (or built) {len(df) - len(errors)}/{len(df)} records.")
    if errors:
        print("Failed:")
        for fid, msg in errors:
            print(f"  * {fid}: {msg}")
    print(f"Summary saved: {out_summary}")
    print("Monitor tasks: https://code.earthengine.google.com/tasks")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: Export ERA5 weather + SRTM terrain per fire Id. "
            "Default: one stacked weather GeoTIFF per fire. "
            "Use --daily-weather-exports for legacy one-file-per-day mode."
        )
    )
    parser.add_argument("csv", help="Path to input CSV (e.g. data/raw/csv/fire_all.csv).")
    parser.add_argument("--project", default="wildfire-area-regression", help="GEE/Cloud project id.")
    parser.add_argument("--folder", default="Wildfire_Environment_Exports", help="Google Drive folder for exports.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Days before start_date to include.")
    parser.add_argument("--min-buffer", type=int, default=DEFAULT_MIN_BUFFER_M, help="Minimum ROI radius in metres.")
    parser.add_argument("--weather-scale", type=int, default=DEFAULT_WEATHER_SCALE_M,
                        help="ERA5 export scale in metres (default 9000; do not use 30).")
    parser.add_argument("--terrain-scale", type=int, default=DEFAULT_TERRAIN_SCALE_M, help="SRTM export scale (m).")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S,
                        help="Seconds to sleep between daily weather task submissions (legacy mode only).")
    parser.add_argument(
        "--daily-weather-exports", action="store_true",
        help="Legacy mode: export one weather GeoTIFF per day (many GEE tasks). "
             "Default is one stacked multi-band weather GeoTIFF per fire.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build manifest but do not call GEE.")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N rows (recommended for testing).")

    args = parser.parse_args()

    main(
        csv_path=args.csv,
        project=args.project,
        folder=args.folder,
        lookback_days=args.lookback_days,
        min_buffer_m=args.min_buffer,
        weather_scale_m=args.weather_scale,
        terrain_scale_m=args.terrain_scale,
        sleep_s=args.sleep,
        dry_run=args.dry_run,
        limit=args.limit,
        daily_weather_exports=args.daily_weather_exports,
    )
