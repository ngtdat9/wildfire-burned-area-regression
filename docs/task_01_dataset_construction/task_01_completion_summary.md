# Task 1 — Completion Summary

**Status:** Task 1 is **completed** as a first working version (Phases 1–4).

## 1. Objective

Build a reproducible, per-wildfire-event geospatial dataset from public
remote-sensing and reanalysis sources, then export **temporal fire-spread**
ML samples for downstream modelling (Task 2+).

## 2. Completed phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Burn labels (MODIS burned area, Landsat severity, daily fire-mask stacks) | Completed |
| 2 | ERA5 weather stacks + SRTM terrain | Completed |
| 3 | 30 m alignment, compression, validation, visualization | Completed |
| 4 | Event splits + temporal tensor export | Completed |

**Source code:** `scripts/task1/`

## 3. Repository layout (Task 1 scripts)

```
scripts/task1/
├── phase_01_burn_labels/
├── phase_02_environment_terrain/
├── phase_03_alignment/
├── phase_04_tensors/
├── validation/
├── visualization/
└── utils/
```

Run from project root, e.g. `python scripts/task1/phase_04_tensors/create_event_splits.py`.

Large aligned GeoTIFFs and NPZ tensors are shared via **Google Drive**, not GitHub.

## 4. Final data outputs

| Path | Description |
|------|-------------|
| `data/aligned_compressed/<fire_id>/` | Main aligned dataset (GeoTIFFs, deflate-compressed) |
| `data/splits/train_events.txt` (etc.) | Event-level train/val/test fire IDs |
| `data/tensors/` | Temporal NPZ samples + `sample_index.csv` |

Per-fire aligned layout:

```
data/aligned_compressed/<fire_id>/
├── labels/
│   ├── burned_area_30m_aligned.tif
│   ├── severity_30m_aligned.tif
│   └── fire_mask_stacked_30m_aligned.tif
├── terrain/terrain_30m_aligned.tif
├── weather/weather_stacked_30m_aligned.tif
└── metadata.json
```

**Not in GitHub:** large `.tif`, `.npz`, and data folders — share via **Google Drive**.

## 5. Temporal ML format (Phase 4)

Fire-spread samples (one NPZ per day-step):

```
X = fire mask + weather for [t-3, t-2, t-1, t]   # shape (4, 7, H, W)
Y = fire mask at t+1                             # shape (1, H, W)
```

Static per sample: `terrain` (3, H, W), `severity` (1, H, W).

**Split rule:** train/val/test by **entire wildfire event** — never by pixel
or random temporal sample (prevents leakage).

## 6. How to reproduce

### Phase 1 — Burn labels

```bash
python scripts/task1/validation/validate_fire_csv.py data/raw/csv/fire_all.csv
python scripts/task1/phase_01_burn_labels/extract_burn_labels.py data/raw/csv/fire_all.csv
python scripts/task1/phase_01_burn_labels/organize_satellite_labels.py
```

Daily fire masks (incremental):

```bash
python scripts/task1/phase_01_burn_labels/extract_daily_fire_masks.py data/raw/csv/fire_all.csv
python scripts/task1/phase_01_burn_labels/organize_fire_mask_stacks.py
```

### Phase 2 — Environment & terrain

```bash
python scripts/task1/phase_02_environment_terrain/extract_environment_terrain.py data/raw/csv/fire_all.csv
python scripts/task1/phase_02_environment_terrain/organize_environment_terrain.py
```

### Phase 3 — Alignment & compression

```bash
python scripts/task1/validation/check_phase1_phase2_inventory.py
python scripts/task1/phase_03_alignment/align_fire_event_layers.py
python scripts/task1/phase_03_alignment/compress_aligned_geotiffs.py --delete-original-after-success
python scripts/task1/validation/validate_aligned_fire_data.py --limit 5
```

Fire-mask merge (if built in update folder):

```bash
rsync -av data/fire_mask_update/aligned_compressed/ data/aligned_compressed/
```

### Phase 4 — Splits & temporal tensors

```bash
python scripts/task1/phase_04_tensors/create_event_splits.py
python scripts/task1/phase_04_tensors/build_temporal_tensors.py --split train --skip-existing --delete-source-after-success
```

Resume without rebuilding finished fires:

```bash
python scripts/task1/phase_04_tensors/build_temporal_tensors.py --split train --skip-existing
```

## 7. Known limitations

1. **ERA5 on 30 m grid** — weather was bilinearly upsampled to the Landsat
   severity grid for ML compatibility; this does **not** add real sub-km
   weather spatial detail.
2. **Large GeoTIFFs** — aligned rasters can be hundreds of MB per fire.
3. **Storage-heavy tensors** — NPZ export can require substantial disk;
   use `--skip-existing`, `--delete-source-after-success`, and Drive sharing.
4. **Event-based splits are mandatory** — random pixel/sample splits would
   leak spatial information and inflate metrics.
5. **Legacy Phase 4** — `scripts/task1/phase_04_tensors/build_tensors.py`
   aggregates weather to mean/max (static baseline); superseded by temporal
   `build_temporal_tensors.py`.

## 8. Handoff to Task 2+ (modelling)

1. Load splits from `data/splits/train_events.txt`, `val_events.txt`, `test_events.txt`.
2. Index samples via `data/tensors/sample_index.csv`.
3. Load NPZ paths listed in `x_path` (share tensors via Drive, not Git).
4. Fit normalization on **train fires only**.
5. Model next-day fire spread from temporal `X` (+ static terrain/severity).

Task 2+ scope (training, evaluation, deployment) is **not yet defined** in this repo.

## Related docs

- [`task_01_overview.md`](./task_01_overview.md)
- [`phase_04_tensor_construction.md`](./phase_04_tensor_construction.md)
- [`phase_03_geospatial_alignment.md`](./phase_03_geospatial_alignment.md)
- [`../current_status.md`](../current_status.md)
