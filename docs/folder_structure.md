# Folder Structure

Top-level layout for `wildfire-burned-area-regression/`:

```
wildfire-burned-area-regression/
├── data/                       # pipeline I/O (large files gitignored)
├── scripts/
│   └── task1/                  # Task 1 entrypoints (by phase)
├── notebooks/
├── configs/
├── docs/
├── outputs/
├── reports/weekly_reports/
├── logs/
├── requirements.txt
└── README.md
```

## `data/`

| Path | Role |
|------|------|
| `data/raw/` | Phase 1–2 exports, CSVs |
| `data/aligned_compressed/` | Phase 3 aligned GeoTIFFs per fire |
| `data/tensors/` | Phase 4 temporal NPZ + `sample_index.csv` |
| `data/splits/` | Event-level train/val/test lists |

Share large artefacts via **Google Drive**, not GitHub.

## `scripts/task1/`

Task 1 is **completed** (Phases 1–4, first working versions). Example commands:

```bash
python scripts/task1/phase_03_alignment/align_fire_event_layers.py --fire-id 20562846
python scripts/task1/phase_04_tensors/create_event_splits.py
python scripts/task1/phase_04_tensors/build_temporal_tensors.py --split train --skip-existing
```

```
scripts/task1/
├── phase_01_burn_labels/       # extract + organize burn labels & fire masks
├── phase_02_environment_terrain/
├── phase_03_alignment/
├── phase_04_tensors/             # temporal + legacy static baseline
├── validation/
├── visualization/
└── utils/
```

Phase 4 temporal format: **X** = [t−3…t] (fire mask + weather), **Y** = fire mask at t+1; splits by **entire event**.

## `docs/`, `notebooks/`, `configs/`, `outputs/`, `reports/`, `logs/`

See [`current_status.md`](./current_status.md) and [`task_01_dataset_construction/task_01_completion_summary.md`](./task_01_dataset_construction/task_01_completion_summary.md).
