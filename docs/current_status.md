# Current Status

Snapshot of where the project stands. The project is organised into
**tasks**; each task has one or more phases.

## Tasks

| Task | Title                                   | Status                  |
|------|-----------------------------------------|-------------------------|
| 1    | Dataset construction & data preparation | **Completed first version** (all 4 phases done) |
| 2+   | *(modelling, evaluation, downstream analysis)* | **Not defined yet** |

> Tasks beyond Task 1 are intentionally **not defined yet** — no
> folders, scripts, or docs should be created for them until they are
> formally scoped. See `docs/project_overview.md` for the long-term
> rationale.

## Task 1 — Dataset construction & data preparation

| Phase | Title                              | Owner      | Status                  |
|-------|------------------------------------|------------|-------------------------|
| 1     | Burn label extraction              | Student 1  | Completed first version |
| 2     | Environment & terrain extraction   | Student 2  | Completed first version |
| 3     | Geospatial alignment               | Student 3  | Completed first version |
| 4     | Tensor structuring & ML export     | Student 4  | Completed first version |

### Phase 1 — Burn label extraction
- Completed first version by **Student 1**.
- Raw outputs organized under `data/raw/satellite_labels/<fire_id>/`.
- See [`task_01_dataset_construction/phase_01_burn_label_extraction.md`](./task_01_dataset_construction/phase_01_burn_label_extraction.md).
- Scripts: `scripts/extract_burn_labels.py`, `scripts/organize_satellite_labels.py`.

### Phase 2 — Environment & terrain extraction
- Completed first version by **Student 2**.
- Raw outputs organized under `data/raw/environment_terrain/<fire_id>/`.
- See [`task_01_dataset_construction/phase_02_environment_terrain_extraction.md`](./task_01_dataset_construction/phase_02_environment_terrain_extraction.md).
- Scripts: `scripts/extract_environment_terrain.py`, `scripts/organize_environment_terrain.py`.

### Phase 3 — Geospatial alignment
- Completed first version by **Student 3**.
- Aligned Phase 1 labels and Phase 2 weather/terrain onto the Landsat
  severity **30 m reference grid** per fire event.
- Final deliverable: compressed GeoTIFFs under
  `data/aligned_compressed/<fire_id>/`.
- See [`task_01_dataset_construction/phase_03_geospatial_alignment.md`](./task_01_dataset_construction/phase_03_geospatial_alignment.md).
- Scripts: `scripts/align_fire_event_layers.py`, `scripts/compress_aligned_geotiffs.py`.

### Daily fire-mask update *(incremental — in progress)*
- Stacked daily cumulative fire masks for time-series modeling.
- Scripts: `scripts/extract_daily_fire_masks.py`, `scripts/organize_fire_mask_stacks.py`, `scripts/align_fire_mask_update.py`.
- Update folder: `data/fire_mask_update/aligned_compressed/` → merge with `rsync` into main dataset.
- Test fire `20778153` aligned successfully; full run pending for all fires.

### Phase 4 — Tensor structuring & event-based splitting
- Completed first version by **Student 4**.
- Reads aligned GeoTIFFs from `data/aligned_compressed/` (39 fires).
- Stacks terrain + aggregated weather into 15-channel feature tensors.
- Outputs land under `data/tensors/` (per-fire `.npz` files) and
  `data/splits/` (train/val/test split by entire wildfire event).
- See [`task_01_dataset_construction/phase_04_tensor_structuring.md`](./task_01_dataset_construction/phase_04_tensor_structuring.md).
- Scripts: `scripts/build_tensors.py`, `scripts/build_splits.py`.
- Colab notebook: `notebooks/03_phase4_tensor_export_colab.ipynb`.

## Tasks 2+ *(not defined yet)*

Future tasks will cover modelling, evaluation, and any downstream
analysis on top of the Task 1 dataset. They have **no scope, no
folders, no docs, and no code** in this repository yet. They will be
added once Task 1 reaches a stable deliverable.
