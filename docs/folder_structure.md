# Folder Structure

This document explains every top-level folder in the
`wildfire-burned-area-regression/` repository.

```
wildfire-burned-area-regression/
├── data/
│   ├── raw/
│   │   ├── csv/
│   │   ├── satellite_labels/
│   │   ├── environment_terrain/
│   │   └── fire_mask_stacked/      # daily fire-mask GEE downloads
│   ├── aligned/
│   ├── aligned_compressed/
│   ├── fire_mask_update/           # incremental daily fire-mask alignment
│   ├── tensors/
│   └── splits/
├── scripts/
├── notebooks/
├── configs/
├── docs/
│   ├── project_overview.md
│   ├── current_status.md
│   ├── folder_structure.md
│   └── task_01_dataset_construction/
│       ├── task_01_overview.md
│       ├── phase_01_burn_label_extraction.md
│       ├── phase_02_environment_terrain_extraction.md
│       ├── phase_03_geospatial_alignment.md
│       ├── role_distribution.md
│       └── known_issues_and_todos.md
├── reports/
│   └── weekly_reports/
├── logs/
├── .gitignore
├── requirements.txt
└── README.md
```

## `data/`
Everything the pipeline reads or writes.

- **`data/raw/csv/`** — input event CSVs (e.g. `fire_all.csv`) and
  derived manifests such as `event_manifest.json`.
- **`data/raw/satellite_labels/`** — GeoTIFFs produced by Task 1 /
  Phase 1 (`*_BurnedArea.tif`, `*_Severity.tif`) after sync from
  Google Drive.
- **`data/raw/environment_terrain/`** — GeoTIFFs produced by Task 1 /
  Phase 2 (`*__terrain__static.tif`, `*__weather__stacked.tif`; legacy
  daily mode: `*__weather__YYYY-MM-DD.tif`)
  after sync from Google Drive.
- **`data/raw/fire_mask_stacked/`** — stacked daily cumulative fire-mask
  GeoTIFFs from GEE (`<fire_id>__fire_mask__stacked.tif`), organized by
  fire ID after download from Drive folder `Wildfire_FireMask_Stacked`.
- **`data/aligned/`** — temporary workspace for Phase 3 alignment
  (uncompressed). Often empty after compression + cleanup.
- **`data/aligned_compressed/`** — *(Task 1 / Phase 3, completed)* final
  aligned, internally compressed GeoTIFFs per fire event. **Phase 4
  input.**
- **`data/fire_mask_update/`** — incremental update for daily stacked
  fire masks. Aligned output:
  `data/fire_mask_update/aligned_compressed/<fire_id>/labels/
  fire_mask_stacked_30m_aligned.tif`. Merge into the main dataset with
  `rsync -av data/fire_mask_update/aligned_compressed/
  data/aligned_compressed/` after verification. Summary:
  `data/fire_mask_update/fire_mask_update_summary.csv`.
- **`data/tensors/`** — *(Task 1 / Phase 4, not started)* ML-ready
  stacked tensors (e.g. `.npz` files), one per fire event.
- **`data/splits/`** — *(Task 1 / Phase 4, not started)* train / val
  / test index files.

> Heavy artefacts (GeoTIFFs, NetCDFs, NPZ, large CSVs) are
> **gitignored**. The empty folders are kept under version control
> via `.gitkeep` placeholder files.

## `scripts/`
Runnable command-line entrypoints. All are designed to be run from
the **project root**:

- `extract_burn_labels.py` — Task 1 / Phase 1 entrypoint.
- `extract_daily_fire_masks.py` — incremental daily fire-mask stack
  export (MODIS MCD64A1 BurnDate) to Google Drive.
- `extract_environment_terrain.py` — Task 1 / Phase 2 entrypoint.
- `organize_satellite_labels.py` — group flat Phase 1 Drive downloads.
- `organize_environment_terrain.py` — group flat Phase 2 downloads.
- `organize_fire_mask_stacks.py` — group flat fire-mask stack downloads.
- `align_fire_event_layers.py` — Task 1 / Phase 3 alignment.
- `align_fire_mask_update.py` — align fire-mask stacks into
  `data/fire_mask_update/`.
- `compress_aligned_geotiffs.py` — deflate-compress Phase 3 aligned
  layers into `data/aligned_compressed/`.
- `validate_fire_csv.py` — quick sanity check on the input CSV.
- `build_event_manifest.py` — turn the input CSV into a JSON manifest.

## `notebooks/`
Exploratory Jupyter notebooks used to sanity-check exported data:

- `01_check_burn_label_outputs.ipynb`
- `02_check_environment_terrain_outputs.ipynb`

## `configs/`
YAML example configs (committed); local copies (`*.yaml` without the
`.example` suffix) should be gitignored if students start customising
them.

## `docs/`
Project documentation.

- **Top level** (project-wide):
  - `project_overview.md` — the whole project.
  - `current_status.md` — task & phase progress.
  - `folder_structure.md` — this file.
- **`docs/task_01_dataset_construction/`** — everything specific to
  **Task 1** (dataset construction & data preparation):
  - `task_01_overview.md`
  - `phase_01_burn_label_extraction.md`
  - `phase_02_environment_terrain_extraction.md`
  - `role_distribution.md`
  - `known_issues_and_todos.md`

> Future tasks (Task 2+) would each get their own sibling folder
> (e.g. `docs/task_02_*/`) **once they are scoped**. None exist yet
> and none should be created speculatively.

## `reports/weekly_reports/`
Short weekly progress reports — one Markdown file per week per
student (or one shared report per week, depending on team
preference).

## `logs/`
Run logs from scripts (text/JSON). Gitignored by default; the
directory is preserved via `.gitkeep`.
