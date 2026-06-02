# Task 1 / Phase 4 — Tensor Structuring & ML Export

> Part of **Task 1 — Dataset Construction & Data Preparation**. See
> [`task_01_overview.md`](./task_01_overview.md) for the surrounding
> context.

- **Assigned student:** Student 4
- **Status:** Completed first version
- **Scripts:**
  - `scripts/task1/phase_04_tensors/build_tensors.py` — stack aligned GeoTIFFs into per-fire tensors
  - `scripts/task1/phase_04_tensors/build_splits.py` — event-level train / val / test split
  - *(removed)* `generate_mock_data.py` — was a dev-only helper; not shipped in this repo
- **Notebook:** `notebooks/03_phase4_tensor_export_colab.ipynb` — Colab-ready end-to-end pipeline

## Purpose

Convert the aligned, compressed GeoTIFFs from Phase 3 into a format that
machine learning models can consume directly:

- Stack all per-fire layers into a single feature tensor per fire event.
- Aggregate variable-length daily weather into a fixed-size summary.
- Split fire events into train / validation / test sets at the **event level**.

## Inputs (per fire)

Reads from `data/aligned_compressed/<fire_id>/` — the Phase 3 deliverable.
Only fires with `fire_complete=True` in `compression_report.csv` are processed.

| Path | Content | Bands |
|------|---------|-------|
| `labels/burned_area_30m_aligned.tif` | MODIS binary burned mask | 1 (uint8) |
| `labels/severity_30m_aligned.tif` | Landsat dNBR 4-class severity | 1 (uint8) |
| `terrain/terrain_30m_aligned.tif` | SRTM elevation, slope, aspect | 3 (float32) |
| `weather/weather_stacked_30m_aligned.tif` | ERA5 daily weather (6 vars × N days) | N×6 (float32) |

## Feature engineering — weather aggregation

Weather bands vary in count across fires (different fire durations).
To produce a **fixed-size feature tensor** for every fire, the 6 ERA5
variables are each reduced to **mean** and **max** over the fire's full
duration:

```
Input:  (N_days × 6, H, W)  — e.g. 330 bands for a 55-day fire
Output: (12, H, W)           — 6 variables × 2 aggregations
```

Bands are read **one at a time** (streaming) to avoid loading multi-GB
weather stacks into RAM, which would crash the Colab kernel.

### Weather variables (6)

| Variable | Units |
|----------|-------|
| `temperature_2m` | °C |
| `relative_humidity_pct` | % |
| `wind_speed_ms` | m/s |
| `wind_dir_deg` | ° |
| `surface_pressure` | Pa |
| `total_precipitation` | mm |

## Feature tensor layout (15 channels)

The final feature tensor per fire has shape **(H, W, 15)** float32:

| Index | Band name | Source | Type |
|-------|-----------|--------|------|
| 0 | `elevation_m` | terrain tif | static |
| 1 | `slope_deg` | terrain tif | static |
| 2 | `aspect_deg` | terrain tif | static |
| 3 | `temperature_2m_mean` | weather tif | mean over fire duration |
| 4 | `temperature_2m_max` | weather tif | max over fire duration |
| 5 | `relative_humidity_pct_mean` | weather tif | mean |
| 6 | `relative_humidity_pct_max` | weather tif | max |
| 7 | `wind_speed_ms_mean` | weather tif | mean |
| 8 | `wind_speed_ms_max` | weather tif | max |
| 9 | `wind_dir_deg_mean` | weather tif | mean |
| 10 | `wind_dir_deg_max` | weather tif | max |
| 11 | `surface_pressure_mean` | weather tif | mean |
| 12 | `surface_pressure_max` | weather tif | max |
| 13 | `total_precipitation_mean` | weather tif | mean |
| 14 | `total_precipitation_max` | weather tif | max |

Labels are saved separately (not mixed into the feature array):

| Key | Shape | dtype | Content |
|-----|-------|-------|---------|
| `burned_area` | (H, W) | uint8 | 0 = unburned, 1 = burned |
| `severity` | (H, W) | uint8 | 0–4 dNBR class |

## Outputs

### Per-fire tensors

```
data/tensors/<fire_id>/
├── tensor.npz        — compressed numpy archive
│     keys: features (H,W,15)  burned_area (H,W)  severity (H,W)
└── band_info.json    — channel names and aggregation descriptions
```

### Global manifest

```
data/tensors/dataset_manifest.json
```

Records every processed fire: success flag, raster dimensions,
number of weather days, and the full band-info list.

### Split files

```
data/splits/
├── splits.json     — full manifest (seed, fractions, fire IDs per split)
├── train.txt       — one fire_id per line
├── val.txt
└── test.txt
```

## Train / val / test split

| Set | Fraction | Fire count (39 fires) |
|-----|----------|-----------------------|
| train | 70 % | 27 |
| val | 15 % | 6 |
| test | 15 % | 6 |

- **Unit of splitting: entire wildfire event.** Never split by pixel
  or patch — that leaks spatial information across splits and makes
  model evaluation misleadingly optimistic.
- Random seed: **42** (fixed for reproducibility).
- Split files committed to Git so all team members use identical
  train / val / test fire sets.

## How to run

### Option A — Google Colab (recommended, no local disk needed)

Open `notebooks/03_phase4_tensor_export_colab.ipynb` in Colab,
mount Google Drive, set `ALIGNED_DIR` to the shared Drive path, and
run all cells. Tensors and splits are written back to Drive.

### Option B — Local (requires rasterio)

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) generate mock data for testing without real data
# (optional) create minimal mock aligned data locally for dry-runs — not in repo

# Build tensors for all fires
python scripts/task1/phase_04_tensors/build_tensors.py

# Create train/val/test split
python scripts/task1/phase_04_tensors/build_splits.py
```

Single fire:

```bash
python scripts/task1/phase_04_tensors/build_tensors.py --fire-id 24332647
```

Custom directories:

```bash
python scripts/task1/phase_04_tensors/build_tensors.py \
    --aligned-dir /path/to/aligned_compressed \
    --tensor-dir  /path/to/tensors
```

## Loading a tensor downstream

```python
import numpy as np

data = np.load("data/tensors/<fire_id>/tensor.npz")
X          = data["features"]     # (H, W, 15) float32 — input features
y_burned   = data["burned_area"]  # (H, W)     uint8   — binary label
y_severity = data["severity"]     # (H, W)     uint8   — 4-class label

# Flatten for tabular ML (Random Forest, XGBoost, …)
X_flat = X.reshape(-1, 15)           # (H*W, 15)
y_flat = y_burned.reshape(-1)        # (H*W,)
```

## Delivery to Task 2 (modelling)

The modelling team should:

- Read fire IDs from `data/splits/train.txt`, `val.txt`, `test.txt`.
- Load tensors for each fire from `data/tensors/<fire_id>/tensor.npz`.
- Normalise features before training (StandardScaler or MinMaxScaler
  fitted on train set only, applied to val and test).
- Never retrain the scaler on val or test data.
- Never mix fires across the split boundary.

## Completion checklist

- [x] `build_tensors.py` implemented and tested
- [x] `build_splits.py` implemented and tested
- [x] `generate_mock_data.py` added for offline testing
- [x] Colab notebook (`03_phase4_tensor_export_colab.ipynb`) working
- [x] Weather streaming fix applied (prevents Colab OOM kernel crash)
- [x] `dataset_manifest.json` committed to Git
- [x] Split files (`splits.json`, `train.txt`, `val.txt`, `test.txt`) committed to Git
- [x] `.gitignore` updated to allow split and manifest files
- [ ] Extend tensor to include `fire_mask_stacked_30m_aligned.tif`
      once the daily fire-mask update pipeline is complete

## Related docs

- Phase 3: [`phase_03_geospatial_alignment.md`](./phase_03_geospatial_alignment.md)
- Open TODOs: [`known_issues_and_todos.md`](./known_issues_and_todos.md)
- Role distribution: [`role_distribution.md`](./role_distribution.md)
