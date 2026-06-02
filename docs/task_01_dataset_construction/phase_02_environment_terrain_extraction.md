# Task 1 / Phase 2 — Environment & Terrain Extraction

> Part of **Task 1 — Dataset Construction & Data Preparation**. See
> [`task_01_overview.md`](./task_01_overview.md) for the surrounding
> context.

- **Assigned student:** Student 2
- **Status:** Completed first version (stacked weather export, 2026-05-24)
- **Script:** `scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py`
- **Config example:** `configs/environment_terrain_config.example.yaml`

## Inputs

A CSV of fire events containing at least the following columns:

| Column       | Type    | Notes                                  |
|--------------|---------|----------------------------------------|
| `Id`         | string  | Unique fire identifier                 |
| `area`       | float   | Reported fire area in **m²**           |
| `lat`        | float   | Fire centroid latitude (deg)           |
| `lon`        | float   | Fire centroid longitude (deg)          |
| `start_date` | date    | Fire start date                        |
| `end_date`   | date    | Fire end date                          |

By convention this file lives at `data/raw/csv/fire_all.csv`.

## Outputs

Per fire event, exported to Google Drive (then synced into
`data/raw/environment_terrain/`):

### Default — stacked weather (recommended)

| File | Description |
|------|-------------|
| `<Id>__terrain__static.tif` | Static SRTM terrain (30 m) |
| `<Id>__weather__stacked.tif` | All daily ERA5 weather in **one multi-band GeoTIFF** (~9 km) |

Example: a fire with 20 weather days and 6 variables produces about
**120 bands** in the stacked weather file, named like:

```
temperature_2m_day001_20170707
relative_humidity_pct_day001_20170707
wind_speed_ms_day001_20170707
wind_dir_deg_day001_20170707
surface_pressure_day001_20170707
total_precipitation_day001_20170707
temperature_2m_day002_20170708
...
```

Google Earth Engine's `ImageCollection.toBands()` may prefix band names
with the image index (`0_`, `1_`, …). The day index and compact date are
still embedded in each name.

### Optional — legacy daily weather

Pass `--daily-weather-exports` to restore the old behaviour: one
GeoTIFF per day:

```
<Id>__weather__YYYY-MM-DD.tif
```

This creates many more GEE export tasks (one per day per fire) and is
**not recommended** for the full 138-fire CSV.

## Weather date window

For each fire, weather is extracted for every calendar day from:

```
(start_date - lookback_days)  →  end_date
```

Default `--lookback-days` is **3**. This window is unchanged from the
original daily-export design; only the **export packaging** changed
(one stacked file instead of many daily files).

## How to run

**Dry run** (no GEE calls):

```bash
python scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py \
    data/raw/csv/fire_all.csv \
    --limit 1 --dry-run
```

**Live run — stacked weather (default):**

```bash
python scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py \
    data/raw/csv/fire_all.csv \
    --project wildfire-area-regression \
    --folder Wildfire_Environment_Exports \
    --lookback-days 3 \
    --min-buffer 20000 \
    --weather-scale 9000 \
    --terrain-scale 30
```

**Legacy daily weather:**

```bash
python scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py \
    data/raw/csv/fire_all.csv \
    --daily-weather-exports
```

## Organizing Downloaded Phase 2 Outputs

After downloading the exported GeoTIFFs from Google Drive, place them in:

```
data/raw/environment_terrain/
```

Then run:

```bash
python scripts/task1/phase_02_environment_terrain/organize_environment_terrain.py
```

This organizes the flat Phase 2 GeoTIFF outputs into one folder per fire
ID, which makes Phase 3 geospatial alignment easier.

Expected organized structure:

```
data/raw/environment_terrain/
├── <fire_id>/
│   ├── <fire_id>__terrain__static.tif
│   └── <fire_id>__weather__stacked.tif
```

Optional custom input folder:

```bash
python scripts/task1/phase_02_environment_terrain/organize_environment_terrain.py \
    --input-dir data/raw/environment_terrain
```

## Summary JSON

After submission, the script writes
`data/raw/csv/fire_all_environment_export_summary.json` with one entry
per fire under `events`:

| Field | Meaning |
|-------|---------|
| `terrain_task_id` | GEE task ID for terrain export |
| `weather_task_id` | GEE task ID for stacked (or daily) weather |
| `weather_days` | Number of days in the weather window |
| `weather_start_date` | First weather date (incl. lookback) |
| `weather_end_date` | Last weather date |
| `weather_band_count_expected` | `weather_days × 6` (approx. band count) |
| `weather_mode` | `stacked` or `daily` |

## Data sources

- **ERA5-Land Daily Aggregated** (`ECMWF/ERA5_LAND/DAILY_AGGR`)
- **NASA SRTM** (`USGS/SRTMGL1_003`)

## Weather variables (per day)

- `temperature_2m`
- `relative_humidity_pct` (derived from `temperature_2m` and `dewpoint_temperature_2m`)
- `wind_speed_ms` (derived from `u_/v_component_of_wind_10m`)
- `wind_dir_deg` (derived from `u_/v_component_of_wind_10m`)
- `surface_pressure`
- `total_precipitation` (when available in the source collection)

## Terrain variables (static)

- `elevation`
- `slope`
- `aspect`

## Stacked weather implementation

The script builds one multi-band image per fire:

1. For each date in the weather window, call `extract_daily_weather()`.
2. Rename bands server-side to include day index and date, e.g.
   `temperature_2m_day001_20170707`.
3. Collect daily images in an `ImageCollection` and call `.toBands()`.
4. Export once as `<Id>__weather__stacked.tif` at `--weather-scale 9000`.

Key functions in `scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py`:

- `build_stacked_weather_image(dates, region, utm_crs)` — stacks all days
- `_rename_daily_weather_bands(img, day_index, date_str)` — band naming
- `export_fire(..., daily_weather_exports=False)` — default stacked mode

## Band dtype fix (2026-05-24)

A first live Phase 2 run failed with:

```
Exported bands must have compatible data types;
found inconsistent types: Int16 and Byte. (Error code: 3)
```

Cause: Earth Engine requires **all bands in a single GeoTIFF export
to share a single dtype**. The SRTM-derived terrain image had
`elevation` as `Int16` and `slope`/`aspect` as another type.

Fix: both `extract_terrain()` and `extract_daily_weather()` call
`.toFloat()` before `.clip(region)` so every exported band is
`Float32`.

## ERA5 precipitation band fix (2026-05-24)

Weather exports failed with:

```
Image.select: Band pattern 'total_precipitation' did not match any bands.
```

Cause: the `ECMWF/ERA5_LAND/DAILY_AGGR` collection exposes daily
precipitation as **`total_precipitation_sum`**, not
`total_precipitation`. The script was selecting a band name that does
not exist in this collection revision.

Fix: the script now uses two aligned lists — `ERA5_BANDS_SRC` (raw
collection names, including `total_precipitation_sum`) and
`ERA5_BANDS_OUT` (stable output names, including
`total_precipitation`) — and selects with rename:

```python
.select(ERA5_BANDS_SRC, ERA5_BANDS_OUT)
```

Output GeoTIFFs still expose the band as `total_precipitation` for
downstream notebooks and docs.

## ROI rule (after 2026-05-23 fix)

The `area` column is **confirmed to be in square metres (m²)**, the
same conclusion reached in Phase 1. The ROI radius is therefore:

```
radius_m = max(min_buffer_m, sqrt(area_m2 / pi) * 1.5)
```

with `--min-buffer` (default **20,000 m = 20 km**) acting as a floor.
This is exactly the same formula as Phase 1, so weather/terrain ROIs
match burn-label ROIs for the same fire.

If `area` is missing or unparsable, the script falls back to
`--min-buffer`.

## Known issues / TODO

1. **`area` unit — RESOLVED 2026-05-23.** Confirmed to be **m²**.
   Phase 2 now uses the same dynamic ROI formula as Phase 1.
2. **ERA5 scale.** ERA5-Land is exported at 9000 m by default and is
   **not** aligned to the Landsat 30 m grid. Resampling/alignment is
   Phase 3's job.
3. **Terrain alignment.** SRTM terrain is exported at 30 m, but still
   needs to be reprojected onto the common reference grid in Phase 3.
4. **GEE `toBands()` band prefixes.** Stacked exports may include
   numeric prefixes on band names; Phase 3 should parse by day index
   and variable suffix, not assume exact band order without inspection.
5. **Alignment.** All rasters from Phase 2 still need to be aligned to
   the common 30 m reference grid in **Task 1 / Phase 3** — that is
   Role 3's responsibility, not Phase 2.
