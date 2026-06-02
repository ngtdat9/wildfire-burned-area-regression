# Project Overview

## Title
**Wildfire Burned-Area & Burn-Severity Regression**

## Goal
Build a reproducible **spatio-temporal wildfire pipeline** for burned-area
and **fire-spread** prediction from public remote-sensing and reanalysis data.

## Tasks

| Task | Title                                  | Status                     |
|------|----------------------------------------|----------------------------|
| 1    | Dataset construction & data preparation | **Completed** (first version) |
| 2+   | *(modelling, evaluation, downstream analysis)* | **Not defined yet** |

## Task 1 — Completed

Phases 1–4 are complete as first working versions:

1. Burn labels (MODIS + Landsat + daily fire-mask stacks)  
2. ERA5 weather + SRTM terrain  
3. 30 m geospatial alignment and compression  
4. Temporal tensor export + **event-based** train/val/test splits  

**Code:** `scripts/task1/` (run from project root)  
**Data:** `data/aligned_compressed/`, `data/tensors/`, `data/splits/` — large files via **Google Drive**, not GitHub.

### Temporal ML format

```
X = [t-3, t-2, t-1, t]  # fire mask + weather
Y = fire mask at t+1
```

Splits are by **entire wildfire event**, not random pixels or samples.

### Known limitation

ERA5 weather was upsampled to the 30 m grid for compatibility; this does not
add real sub-kilometre spatial detail.

See [`task_01_dataset_construction/task_01_completion_summary.md`](./task_01_dataset_construction/task_01_completion_summary.md).

## Task 2+

Modelling and evaluation are **not yet defined** in this repo.
