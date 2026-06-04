# Task 1 scripts (`scripts/task1/`)

All Task 1 entrypoints live here. Run from the **project root**, for example:

```bash
python scripts/task1/phase_03_alignment/align_fire_event_layers.py --fire-id 20562846
python scripts/task1/phase_04_tensors/create_event_splits.py
python scripts/task1/phase_04_tensors/build_temporal_tensors.py --split train --skip-existing
```

## Layout

```
scripts/task1/
├── phase_01_burn_labels/
├── phase_02_environment_terrain/
├── phase_03_alignment/
├── phase_04_tensors/          # temporal + legacy static baseline
├── validation/
├── visualization/
└── utils/
```

Large data (`data/aligned_compressed/`, `data/tensors/`) are **not** in Git — share via Google Drive.

See [`docs/task_01_dataset_construction/task_01_completion_summary.md`](../../docs/task_01_dataset_construction/task_01_completion_summary.md).
