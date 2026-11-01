# Current Status

**Project status: Postponed** due to a lack of computation resources on GitHub.

## Tasks

| Task | Title                                   | Status                  |
|------|-----------------------------------------|-------------------------|
| 1    | Dataset construction & data preparation | **Completed** (first working version) |
| 2+   | *(modelling, evaluation, downstream analysis)* | **Not defined yet** |

## Task 1 — Completed (Phases 1–4)

| Phase | Title                              | Status                  |
|-------|------------------------------------|-------------------------|
| 1     | Burn label extraction              | Completed first version |
| 2     | Environment & terrain extraction   | Completed first version |
| 3     | Geospatial alignment               | Completed first version |
| 4     | Temporal tensor construction       | Completed first version |

**Scripts:** `scripts/task1/` (organized by phase).  
**Summary:** [`task_01_dataset_construction/task_01_completion_summary.md`](./task_01_dataset_construction/task_01_completion_summary.md)

### Phase 4 (temporal fire-spread)

- **X** = fire mask + weather for [t−3, t−2, t−1, t]
- **Y** = fire mask at t+1
- Event split: 70/15/15 by fire ID (seed 42) under `data/splits/`
- Tensors: `data/tensors/` (share via Drive, not Git)
- Scripts: `scripts/task1/phase_04_tensors/create_event_splits.py`, `build_temporal_tensors.py`
- Docs: [`phase_04_tensor_construction.md`](./task_01_dataset_construction/phase_04_tensor_construction.md)

### Deliverables

- `data/aligned_compressed/<fire_id>/`
- `data/tensors/` — temporal NPZ samples
- `data/splits/` — `train_events.txt`, `val_events.txt`, `test_events.txt`

## Tasks 2+

Not yet scoped in this repository.
