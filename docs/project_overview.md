# Project Overview

## Title
**Wildfire Burned-Area & Burn-Severity Regression**

## Goal
Build a reproducible, end-to-end **spatio-temporal wildfire pipeline**
that turns publicly available remote-sensing and reanalysis products
into machine-learning-ready data for burned-area and burn-severity
prediction.

## Project structure: tasks

The full project is organised into **tasks**, each composed of one or
more phases. Only **Task 1** is currently in scope.

| Task | Title                                  | Status                     |
|------|----------------------------------------|----------------------------|
| 1    | Dataset construction & data preparation | **In progress**           |
| 2+   | *(modelling, evaluation, downstream analysis)* | **Not defined yet** |

> Future tasks (Task 2 and beyond) will cover model design, training,
> evaluation, and any further analysis. They are intentionally **not
> scoped, not folderised, and not documented** yet. Do not create
> `task_02_*`, `task_03_*`, or `task_04_*` folders inside `docs/` or
> anywhere else in this repository until those tasks are formally
> defined.

## Task 1 — Dataset construction & data preparation

Task 1 is the only task currently active. It builds a per-fire-event
dataset combining:

**CSV `area` unit.** The `area` column in the input fire CSV is in
**square metres (m²)** (confirmed 2026-05-23). Phases 1 and 2 derive
the ROI radius as `max(buffer, sqrt(area / π) * 1.5)`.

| Layer                | Source                                  | Native scale | Use                                  |
|----------------------|-----------------------------------------|--------------|--------------------------------------|
| Fire event reference | TS-SatFire / fire event CSV             | point        | per-event metadata (lat, lon, dates) |
| Burned area mask     | MODIS MCD64A1                           | 500 m        | binary label                         |
| Burn severity (NBR)  | Landsat 8/9 SR (Collection 2)           | 30 m         | 4-class severity from dNBR           |
| Daily weather        | ERA5-Land Daily Aggregated              | ~9 km        | temperature, humidity, wind, etc.    |
| Static terrain       | NASA SRTM (USGS/SRTMGL1_003)            | 30 m         | elevation, slope, aspect             |

### Task 1 phases

1. **Phase 1 — Burn label extraction** *(completed first version, Student 1)*
2. **Phase 2 — Environment & terrain extraction** *(completed first version, Student 2)*
3. **Phase 3 — Geospatial alignment** *(not started)*
4. **Phase 4 — Tensor structuring & ML export** *(not started)*

See:
- [`task_01_dataset_construction/task_01_overview.md`](./task_01_dataset_construction/task_01_overview.md)
- [`task_01_dataset_construction/phase_01_burn_label_extraction.md`](./task_01_dataset_construction/phase_01_burn_label_extraction.md)
- [`task_01_dataset_construction/phase_02_environment_terrain_extraction.md`](./task_01_dataset_construction/phase_02_environment_terrain_extraction.md)
- [`task_01_dataset_construction/role_distribution.md`](./task_01_dataset_construction/role_distribution.md)
- [`task_01_dataset_construction/known_issues_and_todos.md`](./task_01_dataset_construction/known_issues_and_todos.md)

## Long-term goal
Once all four phases of Task 1 are complete, the project will produce
a single **ML-ready dataset** indexed by `fire_id`, with consistent
shape, CRS, and timestamping across the burn label, severity, weather,
and terrain channels. Future tasks (Task 2+) will then consume that
dataset.
