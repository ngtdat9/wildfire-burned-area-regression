# Task 1 — Role Distribution

Task 1 (Dataset Construction & Data Preparation) is split into four
phases. Each phase is owned by one student / role.

## Current assignments

| Role        | Phase | Responsibility                              | Script                                          | Status                  |
|-------------|-------|---------------------------------------------|-------------------------------------------------|-------------------------|
| Student 1   | 1     | Burn label extraction (MODIS + Landsat)     | `scripts/extract_burn_labels.py`                | Completed first version |
| Student 2   | 2     | Environment & terrain extraction (ERA5 + SRTM) | `scripts/extract_environment_terrain.py`     | Completed first version |
| Student 3   | 3     | Geospatial alignment onto common 30 m grid  | `scripts/align_fire_event_layers.py`            | Completed first version |
| Student 4   | 4     | Tensor structuring & ML-ready dataset export | `scripts/build_tensors.py`, `scripts/build_splits.py` | Completed first version |

## Hand-off contract between phases

Each phase consumes the output of the previous phase, so the file
naming, CRS, and metadata conventions must stay consistent.

- **Phase 1 → Phase 3**: produces
  `data/raw/satellite_labels/<fire_id>_*.tif`.
- **Phase 2 → Phase 3**: produces
  `data/raw/environment_terrain/<fire_id>__<layer>__<date>.tif`.
- **Phase 3 → Phase 4**: produces compressed aligned GeoTIFFs under
  `data/aligned_compressed/<fire_id>/`. See
  [`phase_03_geospatial_alignment.md`](./phase_03_geospatial_alignment.md).
- **Phase 4 → downstream tasks**: produces ML-ready tensors under
  `data/tensors/<fire_id>/tensor.npz` and event-level splits under
  `data/splits/`. See [`phase_04_tensor_structuring.md`](./phase_04_tensor_structuring.md).

## Role expectations

- **Student 1 (Phase 1)** — owns the burn-label and severity
  pipeline. Must keep `phase_01_burn_label_extraction.md` and the
  Phase 1 TODOs in `known_issues_and_todos.md` up to date.
- **Student 2 (Phase 2)** — owns the weather and terrain pipeline.
  Must keep `phase_02_environment_terrain_extraction.md` and the
  Phase 2 TODOs in `known_issues_and_todos.md` up to date.
- **Student 3 (Phase 3)** — owns the geospatial alignment pipeline.
  Must keep `phase_03_geospatial_alignment.md` and the Phase 3 TODOs
  in `known_issues_and_todos.md` up to date.
- **Student 4 (Phase 4)** — owns the tensor structuring and split
  pipeline. Must keep `phase_04_tensor_structuring.md` and the Phase 4
  TODOs in `known_issues_and_todos.md` up to date.
