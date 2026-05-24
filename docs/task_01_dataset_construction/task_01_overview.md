# Task 1 — Dataset Construction & Data Preparation

> Task 1 is the **first task** of the broader wildfire project. Its
> sole purpose is to build a clean, per-event, reproducible dataset
> from public remote-sensing and reanalysis sources. Modelling,
> evaluation, and any downstream analysis happen in **later tasks**
> (Task 2+), which are intentionally **not defined yet**.

## Scope of Task 1

Build a reproducible, per-fire-event dataset that combines:

- **Burn labels** — burned-area mask + dNBR burn-severity map.
- **Environmental drivers** — daily weather + static terrain.
- **Event metadata** — a single source of truth indexed by `fire_id`.

Everything in this task is **data preparation only**. No models are
trained inside Task 1.

## Phases of Task 1

| Phase | Title                              | Owner      | Status                          |
|-------|------------------------------------|------------|---------------------------------|
| 1     | Burn label extraction              | Student 1  | Completed first version         |
| 2     | Environment & terrain extraction   | Student 2  | Completed first version         |
| 3     | Geospatial alignment               | Student 3  | Completed first version         |
| 4     | Tensor structuring & ML export     | Student 4  | **Not started**                 |

- Phase 1 details: [`phase_01_burn_label_extraction.md`](./phase_01_burn_label_extraction.md)
- Phase 2 details: [`phase_02_environment_terrain_extraction.md`](./phase_02_environment_terrain_extraction.md)
- Phase 3 details: [`phase_03_geospatial_alignment.md`](./phase_03_geospatial_alignment.md)
- Role / student assignment: [`role_distribution.md`](./role_distribution.md)
- Open TODOs across the task: [`known_issues_and_todos.md`](./known_issues_and_todos.md)

## Inputs to Task 1

A single fire-event CSV at `data/raw/csv/fire_all.csv` containing at
least:

| Column       | Type    | Notes                                  |
|--------------|---------|----------------------------------------|
| `Id`         | string  | Unique fire identifier                 |
| `area`       | float   | Fire area in **m²** (confirmed)        |
| `lat`        | float   | Fire centroid latitude (deg)           |
| `lon`        | float   | Fire centroid longitude (deg)          |
| `start_date` | date    | Fire start date                        |
| `end_date`   | date    | Fire end date                          |

## Outputs of Task 1

### Raw exports (Phases 1 & 2)

Organized by fire ID:

- `data/raw/satellite_labels/<fire_id>/`
  - `<fire_id>_BurnedArea.tif`, `<fire_id>_Severity.tif`
- `data/raw/environment_terrain/<fire_id>/`
  - `<fire_id>__terrain__static.tif`, `<fire_id>__weather__stacked.tif`

### Aligned + compressed deliverable (Phase 3)

**Final aligned dataset for Phase 4:**

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

All layers share the **Landsat severity 30 m reference grid**. GeoTIFFs
use internal `deflate` compression but remain normal `.tif` files
readable by Rasterio, GDAL, and QGIS.

### Not yet built (Phase 4)

- `data/tensors/` — ML-ready tensor stacks
- `data/splits/` — train/val/test split by **entire wildfire event**

## Relationship to future tasks

Task 1 produces the inputs that future tasks will consume:

- **Task 2+** — *not defined yet.* Will cover model design, training,
  evaluation, and further analysis.
