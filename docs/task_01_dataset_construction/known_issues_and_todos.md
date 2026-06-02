# Task 1 — Known Issues & TODOs

Single source of truth for unresolved questions and pending work
inside Task 1 (Dataset Construction & Data Preparation).

## Completed (Task 1 — first versions)

- [x] **Phase 1 — burn label extraction** (Student 1). MODIS burned area
      + Landsat severity exported and organized under
      `data/raw/satellite_labels/<fire_id>/`.
- [x] **Phase 2 — environment & terrain extraction** (Student 2). SRTM
      terrain + stacked ERA5 weather exported and organized under
      `data/raw/environment_terrain/<fire_id>/`.
- [x] **Phase 1 + Phase 2 inventory check.**
      `scripts/task1/validation/check_phase1_phase2_inventory.py` →
      `data/raw/csv/phase1_phase2_inventory.csv`.
- [x] **Phase 3 — geospatial alignment** (Student 3). All ready layers
      aligned to the Landsat severity 30 m reference grid.
- [x] **GeoTIFF compression.** Aligned outputs compressed with internal
      `deflate` compression and stored under
      `data/aligned_compressed/<fire_id>/`.
- [x] **`area` unit — RESOLVED 2026-05-23.** Confirmed **m²**; Phases 1
      and 2 use `radius_m = max(buffer_m, sqrt(area_m2 / π) * 1.5)`.
- [x] **Daily fire-mask update scripts** (incremental). GEE extract,
      organize, and align pipeline for `fire_mask_stacked_30m_aligned.tif`.
      See Phase 1 and Phase 3 docs.

## Remaining TODO

### Daily fire-mask update (incremental)

- [ ] **Full export run.** Submit GEE tasks for all fires in
      `fire_all.csv` → Drive folder `Wildfire_FireMask_Stacked`.
- [ ] **Download & organize** into `data/raw/fire_mask_stacked/`.
- [ ] **Align all fires** with `scripts/task1/phase_03_alignment/align_fire_mask_update.py`.
- [ ] **Merge** into main dataset:
      `rsync -av data/fire_mask_update/aligned_compressed/ data/aligned_compressed/`

### Phase 3 / handoff

- [ ] **Spot-check aligned outputs.** Verify a few random fire folders
      in `data/aligned_compressed/` visually in QGIS or with Rasterio.
- [ ] **Corrupted raw inputs.** Log and optionally re-export any fire
      whose Phase 1/2 GeoTIFFs fail to open during alignment.

### Phase 4 — Tensor construction & event-based splitting (Student 4) — completed

- [x] Build ML-ready tensors from `data/aligned_compressed/<fire_id>/`.
      → `data/tensors/<fire_id>/tensor.npz`
- [x] Split train / validation / test by **entire wildfire event** — not
      random pixels or patches.
      → `data/splits/splits.json`, `train.txt`, `val.txt`, `test.txt`
- [x] Tensor format chosen: compressed NumPy NPZ.
- [x] `docs/task_01_dataset_construction/phase_04_tensor_structuring.md` added.

## Cross-cutting (open)

- [ ] **Centralise config.** GEE project ID is still hard-coded in
      `scripts/task1/phase_01_burn_labels/extract_burn_labels.py`; should read from
      `configs/gee_config.yaml`.
- [ ] **Per-event metadata sidecars** for Phase 1/2 raw exports (Phase 3
      already writes `metadata.json` per aligned fire).
- [ ] Prefer TS-SatFire bounding boxes instead of point + buffer (Phase 1
      & 2).
- [ ] Decide class 0 vs nodata for severity unburned pixels (Phase 1).

## Phase 1 — Burn label extraction (Student 1)

- [x] **2026-05-23 ROI fix.** Dynamic ROI from `area` in m².
- [ ] Prefer actual TS-SatFire bounding box when available.
- [ ] Class 0 vs nodata for unburned severity pixels.
- [ ] Per-event metadata sidecar JSON for raw exports.

## Phase 2 — Environment & terrain extraction (Student 2)

- [x] **2026-05-23 ROI fix**, **2026-05-24 dtype/precipitation/stacked
      weather fixes** — see `phase_02_environment_terrain_extraction.md`.
- [ ] Stacked weather band names may include GEE `toBands()` index
      prefixes; Phase 4 should document parsing convention.
- [ ] Wind direction is circular; future work may use u/v components.

## Phase 1 + Phase 2 Inventory Check

Before alignment (or to audit raw inputs):

```bash
python scripts/task1/validation/check_phase1_phase2_inventory.py
```

Output: `data/raw/csv/phase1_phase2_inventory.csv`

Only rows with `ready_for_phase3=True` should be aligned.

## Phase 3 — Geospatial alignment (Student 3) — completed

See [`phase_03_geospatial_alignment.md`](./phase_03_geospatial_alignment.md)
for full spec, compression settings, and delivery notes for Student 4.

Known limitation (documented, not a blocker):

- ERA5 bilinear upsampling to 30 m adds grid compatibility, not real
  weather spatial detail.

## Phase 4 — Tensor structuring & ML export (Student 4) — completed first version

See [`phase_04_tensor_structuring.md`](./phase_04_tensor_structuring.md) for full spec.

- [x] Event-based train/val/test split (70/15/15, seed=42, by fire event).
- [x] 15-channel feature tensor: terrain (3) + aggregated weather (12).
- [x] Per-fire `tensor.npz` + `band_info.json` under `data/tensors/`.
- [x] `dataset_manifest.json` for downstream task discovery.

Open item:
- [ ] Extend to include `fire_mask_stacked_30m_aligned.tif` as an
      additional input channel once the daily fire-mask update is complete.
