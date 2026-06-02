# Task 1 / Phase 1 — Burn Label Extraction

> Part of **Task 1 — Dataset Construction & Data Preparation**. See
> [`task_01_overview.md`](./task_01_overview.md) for the surrounding
> context.

- **Assigned student:** Student 1
- **Status:** Completed first version
- **Script:** `scripts/task1/phase_01_burn_labels/extract_burn_labels.py`
- **Config example:** `configs/burn_label_config.example.yaml`

## Inputs

A CSV of fire events containing at least the following columns:

| Column       | Type    | Notes                                  |
|--------------|---------|----------------------------------------|
| `Id`         | string  | Unique fire identifier                 |
| `area`       | float   | Reported fire area (unit TBC; see TODO)|
| `lat`        | float   | Fire centroid latitude (deg)           |
| `lon`        | float   | Fire centroid longitude (deg)          |
| `start_date` | date    | Fire start date                        |
| `end_date`   | date    | Fire end date                          |

By convention this file lives at `data/raw/csv/fire_all.csv`.

## Outputs

Per fire event, two GeoTIFFs are exported to Google Drive (then synced
into `data/raw/satellite_labels/`):

- `<Id>_BurnedArea.tif` — binary burned-area mask from MODIS MCD64A1 (500 m).
- `<Id>_Severity.tif`   — 4-class burn-severity map from Landsat 8/9 dNBR (30 m).

## Organizing Downloaded Phase 1 Outputs

After downloading the exported GeoTIFFs from Google Drive, place them in:

```
data/raw/satellite_labels/
```

They may arrive as a flat list of files (e.g. `20900040_BurnedArea.tif`,
`20900040_Severity.tif`, …). To group them into one folder per fire ID,
run:

```bash
python scripts/task1/phase_01_burn_labels/organize_satellite_labels.py
```

Or specify a custom folder:

```bash
python scripts/task1/phase_01_burn_labels/organize_satellite_labels.py --input-dir data/raw/satellite_labels
```

This produces a layout such as:

```
data/raw/satellite_labels/
├── 20900040/
│   ├── 20900040_BurnedArea.tif
│   └── 20900040_Severity.tif
├── 20899967/
│   ├── 20899967_BurnedArea.tif
│   └── 20899967_Severity.tif
```

Organizing by fire ID makes Phase 3 alignment easier. The script only
moves `.tif` / `.tiff` files sitting **directly** in the input folder;
files already in subfolders are not touched. If a destination file
already exists, it is skipped (not overwritten).

Severity classes:

| Class | Label           | dNBR range   |
|-------|-----------------|--------------|
| 0     | Unburned/other  | < 0.10       |
| 1     | Low             | 0.10 – 0.27  |
| 2     | Moderate-Low    | 0.27 – 0.44  |
| 3     | Moderate-High   | 0.44 – 0.66  |
| 4     | High            | > 0.66       |

## Data sources

- **MODIS MCD64A1** (`MODIS/061/MCD64A1`) — monthly burned-area product.
- **Landsat 8/9 Collection 2 SR** (`LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`).

## Important ROI Fix

The first version of the script used the CSV `area` column to estimate
ROI size, assuming the value was in **km²**. During testing, the CSV
`area` values produced unrealistic buffers of more than 10,000 km
(e.g. `ROI buffer: 11638.0 km`, `13514.5 km`, `17535.0 km`) and pulled
in 50,000+ Landsat scenes for a single fire.

**Confirmed 2026-05-23: the `area` column is in SQUARE METRES (m²),
not km².** Verified by inspecting the magnitude of values in
`data/raw/csv/fire_all.csv` (medians in the 10⁶–10⁸ range correspond
to a few km² up to a few hundred km², which is the realistic
wildfire size range).

The ROI formula has therefore been corrected to:

```
radius_m = max(buffer_m, sqrt(area_m2 / pi) * 1.5)
```

i.e. the equivalent-circle radius of the burned area times 1.5, with
`--buffer` (default **20,000 m = 20 km**) acting as a floor so small
fires still get enough spatial context. If `area` is missing or
unparsable, the script falls back to the `--buffer` value.

Examples with the corrected formula (rounded):

| `area` (m²)   | Implied fire size | New radius |
|---------------|-------------------|------------|
| 1.9 × 10⁸     | 190 km²           | ≈ 12 km → floored to 20 km |
| 2.5 × 10⁸     | 250 km²           | ≈ 13 km → floored to 20 km |
| 4.3 × 10⁸     | 430 km²           | ≈ 18 km → floored to 20 km |
| 1.0 × 10⁹     | 1,000 km²         | ≈ 27 km    |
| 5.0 × 10⁹     | 5,000 km²         | ≈ 60 km    |

**Future improvement:**
- Prefer actual TS-SatFire bounding boxes if available, so each fire
  uses its real footprint instead of a circle.

> **Warning** — before running the full CSV, always test one fire and
> verify the printed `ROI buffer` is in the kilometres-to-tens-of-km
> range (not thousands) and that the Landsat scene count is sane.
> Run `python scripts/task1/validation/validate_fire_csv.py data/raw/csv/fire_all.csv`
> first; it will summarise the `area` column and flag it if the
> magnitude does not look like m².

## Method

1. Build an ROI around the fire point using the formula above. The
   `area` column is treated as **m²** (confirmed). `--buffer` is the
   floor.
2. Pull a MODIS burned-area mask covering the fire window
   (`start_date - 15 d → end_date + 15 d`).
3. Build a pre-fire Landsat composite (up to 120 days before
   `start_date`) and compute pre-fire NBR.
4. Build a post-fire Landsat composite (up to 120 days after
   `end_date`) and compute post-fire NBR.
5. Compute `dNBR = pre_NBR - post_NBR`.
6. Apply USGS dNBR thresholds to classify into 4 severity classes and
   restrict the result to the MODIS burned footprint.
7. Export both rasters to Google Drive.

## Daily fire-mask time series (incremental update)

Phase 1 also exports a **single cumulative final burned footprint** in
`<Id>_BurnedArea.tif`. For fire-spread **time-series** modeling, an
additional stacked daily mask is available:

| Layer | Meaning |
|-------|---------|
| `BurnedArea.tif` | Final cumulative burned footprint (one band) |
| `<Id>__fire_mask__stacked.tif` | Daily cumulative mask — one band per day from `start_date` to `end_date` (1 = burned by that date, 0 = not yet) |

Script: `scripts/task1/phase_01_burn_labels/extract_daily_fire_masks.py`

Uses MODIS MCD64A1 `BurnDate` with rule: `BurnDate > 0 AND BurnDate <= day_of_year`.

```bash
python scripts/task1/phase_01_burn_labels/extract_daily_fire_masks.py data/raw/csv/fire_all.csv --fire-id <fire_id>
```

Download to `data/raw/fire_mask_stacked/`, organize with
`scripts/task1/phase_01_burn_labels/organize_fire_mask_stacks.py`, then align with
`scripts/task1/phase_03_alignment/align_fire_mask_update.py` into
`data/fire_mask_update/aligned_compressed/<fire_id>/labels/fire_mask_stacked_30m_aligned.tif`.

## Known issues / TODO

1. **`area` unit — RESOLVED 2026-05-23.** Confirmed to be **m²**. The
   ROI formula now reads `radius = max(buffer, sqrt(area/π) * 1.5)`.
   See "Important ROI Fix" above.
2. **Use real bounding boxes when available.** Prefer the actual
   TS-SatFire bounding box (when available) instead of
   `point + circular buffer`.
3. **Class 0 vs. nodata.** Currently unburned pixels are masked out
   (nodata). Decide whether to write them as a real class 0 for
   downstream ML.
4. **Per-event metadata.** Export a small JSON sidecar per event
   (sensor list, scene counts, date windows, CRS, ROI bounds).
5. **Alignment.** All rasters from Phase 1 still need to be aligned to
   the common 30 m reference grid in **Task 1 / Phase 3** — that is
   Role 3's responsibility, not Phase 1.
