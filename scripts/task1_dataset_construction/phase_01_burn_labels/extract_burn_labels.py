"""
Phase 1 — Burn Label Extraction (Google Earth Engine)
=====================================================
Project : wildfire-burned-area-regression
Script  : scripts/task1/phase_01_burn_labels/extract_burn_labels.py
Role    : Student 1 (Satellite Label Extraction)
Status  : First version completed.

Purpose
-------
For every fire event in the input CSV this script exports two label rasters
(per-event GeoTIFFs) to Google Drive:

  1. Burned Area Mask — MODIS MCD64A1 (binary mask, 500 m).
  2. 4-Class Severity Map — Landsat 8/9 dNBR (30 m), masked to burned area.

Severity classes (USGS standard):
    1 = Low            (dNBR  0.10 - 0.27)
    2 = Moderate-Low   (dNBR  0.27 - 0.44)
    3 = Moderate-High  (dNBR  0.44 - 0.66)
    4 = High           (dNBR > 0.66)
    0 = Unburned / outside mask

Input CSV columns expected:
    Id, area, end_date, lat, lon, start_date   (extra columns are ignored)

Typical usage (from project root):
    earthengine authenticate                          # first-time only
    python scripts/task1/phase_01_burn_labels/extract_burn_labels.py \
        data/raw/csv/fire_all.csv \
        --folder Wildfire_Burn_Labels \
        --buffer 20000

Dependencies:
    pip install -r requirements.txt

Outputs land in Google Drive under the requested folder. Once exported,
they are typically synced to:
    data/raw/satellite_labels/

------------------------------------------------------------------------------
ROI rule (current, after the 2026-05-23 fix):
  The `area` column in the CSV has been CONFIRMED to be in SQUARE
  METRES (m²). The ROI radius is now derived as:

      radius_m = max(buffer_m, sqrt(area_m2 / pi) * 1.5)

  i.e. the equivalent-circle radius times 1.5, with --buffer (default
  20 km) acting as a floor so small fires still get enough spatial
  context. If `area` is missing or unparsable, the script falls back
  to the fixed --buffer value.

  This replaces an earlier broken formula that wrongly assumed `area`
  was in km² and produced absurd >10,000 km buffers. See the matching
  note in
  docs/task_01_dataset_construction/phase_01_burn_label_extraction.md
  ("Important ROI Fix").

TODO (future improvements — DO NOT implement here yet):
  1. Prefer using the actual TS-SatFire bounding box (if/when available)
     instead of a point + circular buffer.
  2. Decide whether masked "unburned" pixels should be turned into a real
     class 0 instead of nodata for downstream ML.
  3. Export per-event metadata (sensor list, scene counts, date windows,
     CRS, ROI bounds) as a sidecar JSON.
  4. Alignment to a common 30 m reference grid will be handled by Role 3
     in a later phase (see docs/current_status.md).
------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import ee
import pandas as pd


# ─────────────────────────────────────────────
# 0. Initialise GEE
# ─────────────────────────────────────────────
# NOTE: Replace the project_id below with your own GEE Cloud project,
# or move it to configs/gee_config.yaml in a future refactor.
try:
    ee.Initialize(project="wildfire-area-regression")
except Exception as e:
    print(f"[ERROR] GEE init failed: {e}")
    print("Run `earthengine authenticate` first, then set your project ID above.")
    sys.exit(1)


# ─────────────────────────────────────────────
# 1. Helper: parse flexible date strings
# ─────────────────────────────────────────────
def parse_date(date_str: str) -> datetime:
    """Accept YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY, or DD-MM-YYYY."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: '{date_str}'")


def date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# 2. Cloud / shadow masking for Landsat 8/9
# ─────────────────────────────────────────────
def mask_landsat_clouds(image: ee.Image) -> ee.Image:
    """Use QA_PIXEL band to mask clouds and cloud shadows."""
    qa = image.select("QA_PIXEL")
    cloud        = qa.bitwiseAnd(1 << 3).eq(0)
    cloud_shadow = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud.And(cloud_shadow))


def scale_landsat(image: ee.Image) -> ee.Image:
    """Apply official USGS scale + offset for Collection-2 SR bands."""
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True)


# ─────────────────────────────────────────────
# 3. NBR calculation
# ─────────────────────────────────────────────
def compute_nbr(image: ee.Image, sensor: str = "L8") -> ee.Image:
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Landsat 8/9 SR:  NIR = SR_B5,  SWIR2 = SR_B7
    """
    nir   = image.select("SR_B5")
    swir2 = image.select("SR_B7")
    nbr   = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")
    return image.addBands(nbr)


# ─────────────────────────────────────────────
# 4. Landsat composite around a date window
# ─────────────────────────────────────────────
def get_landsat_composite(
    region: ee.Geometry,
    start: str,
    end: str,
    window_days: int = 60,
) -> ee.Image:
    """
    Return a median Landsat 8/9 SR composite for the given date range.
    Falls back to a +/- window_days buffer if the date range is short.
    """
    t_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=window_days)).strftime("%Y-%m-%d")
    t_end   = (datetime.strptime(end,   "%Y-%m-%d") + timedelta(days=window_days)).strftime("%Y-%m-%d")

    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(region)
          .filterDate(t_start, t_end)
          .map(mask_landsat_clouds)
          .map(scale_landsat))

    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(region)
          .filterDate(t_start, t_end)
          .map(mask_landsat_clouds)
          .map(scale_landsat))

    combined = l8.merge(l9)
    count = combined.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No Landsat scenes found between {t_start} and {t_end}.")
    print(f"    -> {count} Landsat scenes in composite window ({t_start} - {t_end})")
    return combined.median().clip(region)


# ─────────────────────────────────────────────
# 5. MODIS burned-area mask (MCD64A1)
# ─────────────────────────────────────────────
def get_modis_burned_mask(region: ee.Geometry, start: str, end: str) -> ee.Image:
    """
    MCD64A1 'BurnDate' band > 0  -> pixel was burned in that month.
    Returns a binary mask clipped to region.
    """
    t_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=15)).strftime("%Y-%m-%d")
    t_end   = (datetime.strptime(end,   "%Y-%m-%d") + timedelta(days=15)).strftime("%Y-%m-%d")

    modis = (ee.ImageCollection("MODIS/061/MCD64A1")
             .filterBounds(region)
             .filterDate(t_start, t_end)
             .select("BurnDate"))

    burned_mask = modis.max().gt(0).rename("BurnedArea").clip(region)
    return burned_mask


# ─────────────────────────────────────────────
# 6. dNBR severity classification
# ─────────────────────────────────────────────
def classify_severity(dnbr: ee.Image, burned_mask: ee.Image) -> ee.Image:
    """
    USGS dNBR thresholds -> 4-class severity map, masked to burned area.

    Class | Label          | dNBR range
    ------|----------------|--------------------
      1   | Low            |  0.10 - 0.27
      2   | Moderate-Low   |  0.27 - 0.44
      3   | Moderate-High  |  0.44 - 0.66
      4   | High           |  > 0.66
      0   | Unburned/other |  < 0.10
    """
    severity = (dnbr
                .where(dnbr.gte(0.10).And(dnbr.lt(0.27)), 1)
                .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 2)
                .where(dnbr.gte(0.44).And(dnbr.lt(0.66)), 3)
                .where(dnbr.gte(0.66),                    4)
                .where(dnbr.lt(0.10),                     0))

    # TODO: revisit masking choice — should unburned pixels stay masked
    # (nodata) or be written as a real class 0 for ML training?
    return severity.updateMask(burned_mask).rename("Severity").toInt8()


# ─────────────────────────────────────────────
# 7. Export a single fire
# ─────────────────────────────────────────────
def export_fire(row: dict, buffer_m: int = 20_000, drive_folder: str = "Wildfire_Burn_Labels") -> None:
    """
    Full pipeline for one fire record.
    Exports two assets to Google Drive:
      * <Id>_BurnedArea.tif   (MODIS, 500 m)
      * <Id>_Severity.tif     (Landsat, 30 m)
    """
    fire_id    = row["Id"]
    lat        = float(row["lat"])
    lon        = float(row["lon"])
    start      = date_str(parse_date(row["start_date"]))
    end        = date_str(parse_date(row["end_date"]))

    print(f"\n{'─'*60}")
    print(f"Fire: {fire_id}  |  {lat:.4f}, {lon:.4f}  |  {start} -> {end}")

    # ROI rule: derive radius from the fire `area` (CONFIRMED to be in
    # square metres) using the equivalent-circle radius * 1.5, with
    # --buffer acting as a floor so small fires still get enough
    # spatial context. Falls back to --buffer when `area` is missing
    # or unparsable.
    #
    # TODO: Replace this circular-buffer ROI with the actual TS-SatFire
    #       bounding box if/when available, so each fire uses its real
    #       footprint instead of a circle.
    area_m2 = None
    try:
        area_m2 = float(row["area"])
        radius_m = max(buffer_m, int(math.sqrt(area_m2 / math.pi) * 1.5))
    except (ValueError, TypeError, KeyError):
        radius_m = buffer_m

    region = ee.Geometry.Point([lon, lat]).buffer(radius_m).bounds()
    if area_m2 is not None:
        print(f"    ROI buffer: {radius_m/1000:.1f} km  "
              f"(area = {area_m2:,.0f} m² ≈ {area_m2/1e6:.2f} km²)")
    else:
        print(f"    ROI buffer: {radius_m/1000:.1f} km  (no `area` reported)")

    # A. MODIS burned-area mask
    print("  [1/4] Fetching MODIS burned-area mask ...")
    burned_mask = get_modis_burned_mask(region, start, end)

    # B. Pre-fire Landsat composite (up to 120 days before start)
    pre_end   = date_str(parse_date(start) - timedelta(days=1))
    pre_start = date_str(parse_date(start) - timedelta(days=120))
    print(f"  [2/4] Pre-fire composite: {pre_start} -> {pre_end}")
    pre_composite = get_landsat_composite(region, pre_start, pre_end, window_days=0)
    pre_nbr       = compute_nbr(pre_composite).select("NBR")

    # C. Post-fire Landsat composite (up to 120 days after end)
    post_start = date_str(parse_date(end) + timedelta(days=1))
    post_end   = date_str(parse_date(end) + timedelta(days=120))
    print(f"  [3/4] Post-fire composite: {post_start} -> {post_end}")
    post_composite = get_landsat_composite(region, post_start, post_end, window_days=0)
    post_nbr       = compute_nbr(post_composite).select("NBR")

    # D. dNBR & severity
    print("  [4/4] Computing dNBR and severity classes ...")
    dnbr     = pre_nbr.subtract(post_nbr).rename("dNBR")
    severity = classify_severity(dnbr, burned_mask)

    # E. Export to Drive
    export_params_base = dict(
        region=region,
        scale=500,
        crs="EPSG:4326",
        maxPixels=1e13,
        folder=drive_folder,
    )

    task_ba = ee.batch.Export.image.toDrive(
        image=burned_mask.toInt8(),
        description=f"{fire_id}_BurnedArea",
        fileNamePrefix=f"{fire_id}_BurnedArea",
        **export_params_base,
    )

    task_sv = ee.batch.Export.image.toDrive(
        image=severity,
        description=f"{fire_id}_Severity",
        fileNamePrefix=f"{fire_id}_Severity",
        **{**export_params_base, "scale": 30},
    )

    task_ba.start()
    task_sv.start()
    print("  Export tasks submitted:")
    print(f"      * {fire_id}_BurnedArea  (MODIS 500 m, GDrive/{drive_folder}/)")
    print(f"      * {fire_id}_Severity    (Landsat 30 m, GDrive/{drive_folder}/)")

    # TODO: write a per-event metadata sidecar JSON (sensor list, scene
    # counts, date windows, CRS, ROI bounds) to data/raw/satellite_labels/.


# ─────────────────────────────────────────────
# 8. Main: read CSV and loop over fires
# ─────────────────────────────────────────────
def main(csv_path: str, drive_folder: str = "Wildfire_Burn_Labels", buffer_m: int = 20_000) -> None:
    """
    Parameters
    ----------
    csv_path     : path to the input CSV file (e.g. data/raw/csv/fire_all.csv).
    drive_folder : Google Drive folder name for exports.
    buffer_m     : minimum ROI buffer radius in metres (default 20 km).
    """
    csv_path = str(Path(csv_path))
    print(f"\n{'═'*60}")
    print("  Phase 1 - Burn Label Extraction (GEE)")
    print(f"{'═'*60}")
    print(f"  Input  : {csv_path}")
    print(f"  Output : Google Drive / {drive_folder}/")
    print(f"{'═'*60}\n")

    # Load CSV — tab or comma delimited
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df.columns = [c.strip() for c in df.columns]

    required = {"Id", "lat", "lon", "start_date", "end_date"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERROR] CSV is missing required columns: {missing}")
        print(f"        Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Found {len(df)} fire record(s) in CSV.\n")
    errors = []

    for _, row in df.iterrows():
        try:
            export_fire(row.to_dict(), buffer_m=buffer_m, drive_folder=drive_folder)
        except Exception as e:
            print(f"  [SKIP] Fire {row.get('Id', '?')} failed: {e}")
            errors.append((row.get("Id", "?"), str(e)))

    print(f"\n{'═'*60}")
    print(f"  Submitted {len(df) - len(errors)}/{len(df)} fires successfully.")
    if errors:
        print("  Failed fires:")
        for fid, msg in errors:
            print(f"    * {fid}: {msg}")
    print("\n  Monitor exports at: https://code.earthengine.google.com/tasks")
    print(f"{'═'*60}\n")

    # Save a summary JSON next to the input CSV.
    summary = {
        "phase": "01_burn_label_extraction",
        "total": len(df),
        "submitted": len(df) - len(errors),
        "failed": errors,
        "drive_folder": drive_folder,
    }
    summary_path = csv_path.replace(".csv", "_burn_labels_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved -> {summary_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1: Export burned-area mask (MODIS) and dNBR severity (Landsat 8/9) for fires in a CSV."
    )
    parser.add_argument("csv", help="Path to input CSV file (e.g. data/raw/csv/fire_all.csv).")
    parser.add_argument(
        "--folder", default="Wildfire_Burn_Labels",
        help="Google Drive folder name for exports (default: Wildfire_Burn_Labels)."
    )
    parser.add_argument(
        "--buffer", type=int, default=20_000,
        help="Minimum ROI buffer radius in metres (default: 20000)."
    )
    args = parser.parse_args()

    main(args.csv, drive_folder=args.folder, buffer_m=args.buffer)
