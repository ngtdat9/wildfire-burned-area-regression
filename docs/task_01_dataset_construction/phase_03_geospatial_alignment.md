# Task 1 / Phase 3 — Geospatial Alignment

> Part of **Task 1 — Dataset Construction & Data Preparation**. See
> [`task_01_overview.md`](./task_01_overview.md) for the surrounding
> context.

- **Assigned student:** Student 3
- **Status:** Completed first version
- **Scripts:**
  - `scripts/align_fire_event_layers.py` — reproject/resample layers
  - `scripts/compress_aligned_geotiffs.py` — storage-safe compression
  - `scripts/check_phase1_phase2_inventory.py` — pre-alignment inventory
  - `scripts/test_geotiff_compression.py` — optional one-fire compression test

## Purpose

Align all outputs from Phase 1 and Phase 2 so every layer for a fire
event shares the same:

- CRS
- affine transform
- pixel resolution
- width and height
- spatial extent (bounding box)

This makes pixel-wise stacking for machine learning possible in Phase 4.

## Inputs (per fire)

| Path | Layer |
|------|-------|
| `data/raw/satellite_labels/<fire_id>/<fire_id>_BurnedArea.tif` | MODIS burned area mask |
| `data/raw/satellite_labels/<fire_id>/<fire_id>_Severity.tif` | **Reference grid** (Landsat dNBR) |
| `data/raw/environment_terrain/<fire_id>/<fire_id>__terrain__static.tif` | SRTM elevation, slope, aspect |
| `data/raw/environment_terrain/<fire_id>/<fire_id>__weather__stacked.tif` | ERA5 stacked weather bands |

Prerequisites:

1. Phase 1 outputs organized under `data/raw/satellite_labels/<fire_id>/`
2. Phase 2 outputs organized under `data/raw/environment_terrain/<fire_id>/`
3. Inventory check: `python scripts/check_phase1_phase2_inventory.py`

Only fire IDs with `ready_for_phase3=True` in
`data/raw/csv/phase1_phase2_inventory.csv` were processed.

## Reference grid

**Reference:** `<fire_id>_Severity.tif` (Landsat dNBR burn severity, ~30 m)

Severity is the finest **label** layer used for per-fire spatial detail.
All other layers were resampled to this grid.

## Resampling methods

| Layer | Method | Why |
|-------|--------|-----|
| Burned area | nearest neighbour | class / mask data |
| Severity | nearest neighbour | class map (reference copy) |
| Terrain | bilinear | continuous elevation, slope, aspect |
| Weather (all bands) | bilinear | continuous variables |

Bilinear interpolation was **not** used on class labels (would create
invalid fractional class values).

## Pipeline (completed workflow)

Phase 3 was run as a **two-step pipeline**:

1. **Align** raw layers → temporary `data/aligned/<fire_id>/`
2. **Compress** → final `data/aligned_compressed/<fire_id>/`, optionally
   deleting the uncompressed aligned folder to free disk space

## Final outputs (deliverable)

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

Run-level reports:

- `data/aligned/alignment_summary.csv` — alignment run log
- `data/aligned_compressed/compression_report.csv` — compression run log

### metadata.json (per fire)

Records reference grid details, input/output paths, and resampling
methods. Copied into the compressed folder for Phase 4 auditing.

## GeoTIFF compression

Aligned GeoTIFFs were compressed using **internal GeoTIFF compression**
(`deflate`, tiled, predictors by dtype) to reduce storage from ~70 GB
(uncompressed aligned weather at 30 m) to a much smaller footprint.

Important:

- Files remain normal `.tif` GeoTIFFs on disk.
- **No manual decompression** is needed before reading them.
- Open directly with **Rasterio**, **GDAL**, or **QGIS**.

Compression settings (via `scripts/compress_aligned_geotiffs.py`):

- `compress="deflate"`
- `tiled=True`, `blockxsize=256`, `blockysize=256`
- `BIGTIFF="IF_SAFER"`
- `predictor=2` for integer rasters, `predictor=3` for floating point

## How to run (reference commands)

**Check inventory before alignment:**

```bash
python scripts/check_phase1_phase2_inventory.py
```

**Align one fire:**

```bash
python scripts/align_fire_event_layers.py --fire-id <fire_id>
```

**Compress aligned outputs (storage-safe — delete uncompressed copy after success):**

```bash
python scripts/compress_aligned_geotiffs.py --delete-original-after-success
```

**Align + compress one fire (two steps):**

```bash
python scripts/align_fire_event_layers.py --fire-id <fire_id>
python scripts/compress_aligned_geotiffs.py \
    --fire-id <fire_id> \
    --delete-original-after-success
```

## Delivery to Student 4 (Phase 4)

**Student 4** should:

- Read inputs from `data/aligned_compressed/<fire_id>/`
- Use only fire IDs that aligned and compressed successfully (check
  `compression_report.csv` for `fire_complete=True`)
- Build tensors for machine learning from the aligned, compressed layers
- Split **train / validation / test by entire wildfire event** — never
  by random pixels or patches (that would leak spatial information
  across splits)

## Known limitations (first version)

- **ERA5 weather upsampling.** Weather was originally ~9 km resolution.
  Bilinear upsampling to the 30 m Landsat grid makes the grid compatible
  for ML but **does not add real spatial detail**.
- **Wind direction.** Direction in degrees is circular data; a future
  improvement may align u/v wind components instead.
- **Corrupted inputs.** A small number of raw GeoTIFFs may fail to open
  during alignment (e.g. incomplete downloads). These should be logged
  and re-exported if needed.

## Completion checklist

- [x] Inventory CSV created (`phase1_phase2_inventory.csv`)
- [x] Alignment script implemented (`align_fire_event_layers.py`)
- [x] Compression script implemented (`compress_aligned_geotiffs.py`)
- [x] All ready fire events aligned and compressed under
      `data/aligned_compressed/`
- [ ] Spot-check a few random fire folders visually in QGIS or Rasterio
- [ ] Phase 4 tensor construction (Student 4)
- [ ] Event-based train/val/test split (Student 4)

## Related docs

- Phase 1: [`phase_01_burn_label_extraction.md`](./phase_01_burn_label_extraction.md)
- Phase 2: [`phase_02_environment_terrain_extraction.md`](./phase_02_environment_terrain_extraction.md)
- Open TODOs: [`known_issues_and_todos.md`](./known_issues_and_todos.md)
