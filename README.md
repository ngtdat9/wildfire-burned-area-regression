# Wildfire Burned-Area & Burn-Severity Regression

> **Status: Postponed.** This project is postponed due to a lack of
> computation resources.
>
> **Known data issue (unresolved):** burned-area, severity, and daily
> fire-mask labels may be incomplete for up to 57 of the 138 fires. Fires
> starting on or after the 17th of a month appear to lose that month's
> MODIS burn data because of the date filter used during extraction.
> Check these labels before reusing the dataset.

A reproducible **spatio-temporal wildfire dataset pipeline** that turns
publicly available remote-sensing and reanalysis products into
machine-learning-ready data for **burned-area** and **burn-severity**
prediction.

## Objective

Given a CSV of fire events (id, location, dates), build a per-event
dataset combining:

- MODIS-derived burned-area masks,
- Landsat 8/9 dNBR burn-severity maps,
- ERA5-Land daily weather variables (stacked multi-band),
- SRTM-derived static terrain (elevation, slope, aspect),

so that downstream models can learn to predict burned area / severity
from weather and terrain inputs.

## Current scope: Task 1

The full project is organised into **tasks**. This repository is
currently focused on **Task 1 — Dataset Construction & Data
Preparation**, which is **completed** as a first working version (Phases 1–4).
Task 2+ (modelling) is not defined yet.

Future tasks (modelling, evaluation, downstream analysis) are **not
defined yet**.

### Task 1 phases

| Phase | Title                              | Owner      | Status                          |
|-------|------------------------------------|------------|---------------------------------|
| 1     | Burn label extraction              | Student 1  | Completed first version         |
| 2     | Environment & terrain extraction   | Student 2  | Completed first version         |
| 3     | Geospatial alignment               | Student 3  | Completed first version         |
| 4     | Temporal tensor construction       | Student 4  | Completed first version         |

### Phase 3 summary

Phase 3 aligned Phase 1 labels and Phase 2 weather/terrain onto the
**same reference grid** per fire event. The **Landsat severity map**
(`<fire_id>_Severity.tif`) was used as the **30 m reference grid**.

Final aligned data (with internal GeoTIFF compression) lives under:

```
data/aligned_compressed/<fire_id>/
├── labels/
│   ├── burned_area_30m_aligned.tif
│   └── severity_30m_aligned.tif
├── terrain/
│   └── terrain_30m_aligned.tif
├── weather/
│   └── weather_stacked_30m_aligned.tif
└── metadata.json
```

The compressed `.tif` files are normal GeoTIFFs — open them directly
with **Rasterio**, **GDAL**, or **QGIS**. No manual decompression is
required.

**Student 4 (Phase 4)** should use `data/aligned_compressed/` for tensor
construction and split train/validation/test by **entire wildfire
event**, not random pixels or patches.

### Label layers (after fire-mask update is merged)

| File | Type | Description |
|------|------|-------------|
| `burned_area_30m_aligned.tif` | Final mask | Cumulative burned footprint at end of event |
| `fire_mask_stacked_30m_aligned.tif` | Daily stack | Cumulative fire mask per day (time series) |
| `severity_30m_aligned.tif` | Final map | dNBR burn-severity damage classes |
| `weather_stacked_30m_aligned.tif` | Daily stack | ERA5 weather bands per day |
| `terrain_30m_aligned.tif` | Static | Elevation, slope, aspect |

The fire-mask update is built in `data/fire_mask_update/` first, then
merged:

```bash
rsync -av data/fire_mask_update/aligned_compressed/ data/aligned_compressed/
```

## Folder structure

```
wildfire-burned-area-regression/
├── data/
│   ├── raw/
│   │   ├── csv/                    # input CSVs, manifests, inventory
│   │   ├── satellite_labels/       # Phase 1 (by fire_id/)
│   │   ├── environment_terrain/    # Phase 2 (by fire_id/)
│   │   └── fire_mask_stacked/      # daily fire-mask GEE downloads
│   ├── aligned/                    # temporary alignment workspace (often empty)
│   ├── aligned_compressed/         # Phase 3 final deliverable (by fire_id/)
│   ├── fire_mask_update/           # incremental daily fire-mask alignment
│   ├── tensors/                    # Phase 4 temporal NPZ (gitignored)
│   └── splits/                     # Phase 4 event splits
├── scripts/
│   └── task1/                      # Task 1 entrypoints (by phase)
├── notebooks/
├── configs/
├── docs/
├── outputs/                        # figures, validation maps
├── reports/
└── requirements.txt
```

See `docs/folder_structure.md` for a per-folder explanation.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
earthengine authenticate   # Phases 1 & 2 only (Google Earth Engine)
```

## Phase 1 — Burn label extraction

See `docs/task_01_dataset_construction/phase_01_burn_label_extraction.md`.

```bash
python scripts/task1/phase_01_burn_labels/extract_burn_labels.py \
    data/raw/csv/fire_all.csv \
    --folder Wildfire_Burn_Labels \
    --buffer 20000
```

Organize downloads:

```bash
python scripts/task1/phase_01_burn_labels/organize_satellite_labels.py
```

## Phase 2 — Environment & terrain extraction

See `docs/task_01_dataset_construction/phase_02_environment_terrain_extraction.md`.

```bash
python scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py \
    data/raw/csv/fire_all.csv \
    --project wildfire-area-regression \
    --folder Wildfire_Environment_Exports \
    --weather-scale 9000 \
    --terrain-scale 30
```

Organize downloads:

```bash
python scripts/task1/phase_02_environment_terrain/organize_environment_terrain.py
```

## Phase 3 — Geospatial alignment & compression

See `docs/task_01_dataset_construction/phase_03_geospatial_alignment.md`.

**Check which fires are ready to align:**

```bash
python scripts/task1/validation/check_phase1_phase2_inventory.py
```

**Align one fire:**

```bash
python scripts/task1/phase_03_alignment/align_fire_event_layers.py --fire-id <fire_id>
```

**Compress aligned GeoTIFFs** (writes to `data/aligned_compressed/`,
optionally deletes uncompressed `data/aligned/<fire_id>/` after success):

```bash
python scripts/task1/phase_03_alignment/compress_aligned_geotiffs.py --delete-original-after-success
```

**Typical one-fire workflow (align then compress):**

```bash
python scripts/task1/phase_03_alignment/align_fire_event_layers.py --fire-id <fire_id>
python scripts/task1/phase_03_alignment/compress_aligned_geotiffs.py \
    --fire-id <fire_id> \
    --delete-original-after-success
```

## Daily fire-mask update (time-series labels)

Adds `fire_mask_stacked_30m_aligned.tif` — one band per day showing
cumulative burned area through that date. Does **not** overwrite the
main `data/aligned_compressed/` folder until you merge.

**Extract (GEE, one fire):**

```bash
python scripts/task1/phase_01_burn_labels/extract_daily_fire_masks.py \
    data/raw/csv/fire_all.csv \
    --fire-id <fire_id>
```

**Organize downloads:**

```bash
python scripts/task1/phase_01_burn_labels/organize_fire_mask_stacks.py
```

**Align to 30 m severity grid:**

```bash
python scripts/task1/phase_03_alignment/align_fire_mask_update.py --fire-id <fire_id>
```

**Merge into main dataset after verification:**

```bash
rsync -av data/fire_mask_update/aligned_compressed/ data/aligned_compressed/
```

## Helper scripts

```bash
python scripts/task1/validation/validate_fire_csv.py data/raw/csv/fire_all.csv
python scripts/task1/utils/build_event_manifest.py data/raw/csv/fire_all.csv
```

## Phase 4 — Event splits & temporal tensors

See `docs/task_01_dataset_construction/phase_04_tensor_construction.md` and
`docs/task_01_dataset_construction/task_01_completion_summary.md`.

**X** = fire mask + weather for [t−3…t]; **Y** = fire mask at t+1. Splits are
by **entire wildfire event** (70/15/15, seed 42), not random pixels.

```bash
python scripts/task1/phase_04_tensors/create_event_splits.py
python scripts/task1/phase_04_tensors/build_temporal_tensors.py --split train --skip-existing
```

Large NPZ files belong on **Google Drive**, not Git.

## Documentation

| Doc | Purpose |
|-----|---------|
| `docs/current_status.md` | Project status snapshot |
| `docs/task_01_dataset_construction/task_01_overview.md` | Task 1 overview |
| `docs/task_01_dataset_construction/phase_03_geospatial_alignment.md` | Phase 3 spec & delivery |
| `docs/task_01_dataset_construction/phase_04_tensor_construction.md` | Phase 4 temporal tensors |
| `docs/task_01_dataset_construction/task_01_completion_summary.md` | Task 1 completion summary |
| `docs/task_01_dataset_construction/known_issues_and_todos.md` | Open TODOs |
