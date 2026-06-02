# Task 1 / Phase 4 — Temporal Tensor Construction

> Part of **Task 1 — Dataset Construction & Data Preparation**. See
> [`task_01_overview.md`](./task_01_overview.md) and
> [`task_01_completion_summary.md`](./task_01_completion_summary.md).

- **Status:** Completed first version (temporal fire-spread pipeline)
- **Scripts (canonical):**
  - `scripts/task1/phase_04_tensors/create_event_splits.py`
  - `scripts/task1/phase_04_tensors/build_temporal_tensors.py`
- **Legacy baseline:** `scripts/task1/phase_04_tensors/build_tensors.py`, `build_splits.py` (static weather mean/max)

## Purpose

Export **spatio-temporal** training samples for next-day fire spread:

```
X = fire mask + weather for [t-3, t-2, t-1, t]
Y = fire mask at t+1
```

Split train/val/test by **entire wildfire event** (never by pixel or sample).

## Inputs

`data/aligned_compressed/<fire_id>/` with:

- `labels/fire_mask_stacked_30m_aligned.tif`
- `labels/severity_30m_aligned.tif`
- `terrain/terrain_30m_aligned.tif`
- `weather/weather_stacked_30m_aligned.tif`

## How to run

```bash
python scripts/task1/phase_04_tensors/create_event_splits.py
python scripts/task1/phase_04_tensors/build_temporal_tensors.py --split train --skip-existing
```

Resume + free disk:

```bash
python scripts/task1/phase_04_tensors/build_temporal_tensors.py --split train --skip-existing --delete-source-after-success
```

## Outputs

```
data/splits/train_events.txt
data/tensors/<split>/<fire_id>/<fire_id>_<YYYYMMDD>.npz
data/tensors/sample_index.csv
data/tensors/tensor_manifest.json
```

NPZ keys: `X` (4,7,H,W), `terrain`, `severity`, `Y` (1,H,W).

**Do not commit large `.npz` files** — share via Google Drive.

## Related

- Legacy static tensors: [`phase_04_tensor_structuring.md`](./phase_04_tensor_structuring.md)
- Phase 3: [`phase_03_geospatial_alignment.md`](./phase_03_geospatial_alignment.md)
