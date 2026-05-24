# Task 1 — Role Distribution

Task 1 (Dataset Construction & Data Preparation) is split into four
phases. Each phase is owned by one student / role.

## Current assignments

| Role        | Phase | Responsibility                              | Script                                          | Status                  |
|-------------|-------|---------------------------------------------|-------------------------------------------------|-------------------------|
| Student 1   | 1     | Burn label extraction (MODIS + Landsat)     | `scripts/extract_burn_labels.py`                | Completed first version |
| Student 2   | 2     | Environment & terrain extraction (ERA5 + SRTM) | `scripts/extract_environment_terrain.py`     | Completed first version |
| Role 3      | 3     | Geospatial alignment onto common 30 m grid  | *(no script yet)*                               | **Not started**         |
| Role 4      | 4     | Tensor structuring & ML-ready dataset export | *(no script yet)*                              | **Not started**         |

## Hand-off contract between phases

Each phase consumes the output of the previous phase, so the file
naming, CRS, and metadata conventions must stay consistent.

- **Phase 1 → Phase 3**: produces
  `data/raw/satellite_labels/<fire_id>_*.tif`.
- **Phase 2 → Phase 3**: produces
  `data/raw/environment_terrain/<fire_id>__<layer>__<date>.tif`.
- **Phase 3 → Phase 4**: will produce aligned per-event GeoTIFF stacks
  under `data/aligned/<fire_id>/`. *(Spec to be defined when Phase 3
  starts.)*
- **Phase 4 → downstream tasks**: will produce ML-ready tensors and
  split files under `data/tensors/` and `data/splits/`. *(Spec to be
  defined when Phase 4 starts.)*

## Role expectations

- **Student 1 (Phase 1)** — owns the burn-label and severity
  pipeline. Must keep `phase_01_burn_label_extraction.md` and the
  Phase 1 TODOs in `known_issues_and_todos.md` up to date.
- **Student 2 (Phase 2)** — owns the weather and terrain pipeline.
  Must keep `phase_02_environment_terrain_extraction.md` and the
  Phase 2 TODOs in `known_issues_and_todos.md` up to date.
- **Role 3 (Phase 3)** — *future role.* Will be responsible for
  reprojecting and resampling every layer onto a common 30 m
  reference grid. Will create `phase_03_*.md` when work starts.
- **Role 4 (Phase 4)** — *future role.* Will be responsible for
  stacking aligned layers into tensors, building splits, and writing
  dataset-level metadata. Will create `phase_04_*.md` when work
  starts.
