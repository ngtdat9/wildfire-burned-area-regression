# Task 2 / Student 4 — Temporal Ablations & Final Evaluation

- **Assigned student:** Student 4 (Zohid)
- **Task 2 doc role:** Prithvi-100M Lead → **Backup Plan** (ablations + final
  evaluation support), which is what this delivers.
- **Status:** Implemented for the temporal dataset.

## Dataset (temporal fire-spread)

The Phase-4 temporal pipeline (`mode: temporal_fire_spread`) produces, per fire:

```
tensors/<split>/<fire_id>.npz
    fire_masks  (N_days, H, W)    uint8    daily CUMULATIVE burned mask
    weather     (N_days, 6, H, W) float32  daily ERA5 (6 vars)
    terrain     (3, H, W)         float32  static elevation/slope/aspect
tensors/sample_index.csv      one row per fire (fire_id, split, n_fire_days, ...)
tensors/tensor_manifest.json  window_size, var names, per-split counts
```

Sliding window (W=4): `X = days[i..i+W-1]`, **`Y = fire_masks[i+W]`** (next-day
mask). 137 fires → **95/21/21** split → **1541/316/281** samples.

## The key modeling decision — predict NEW GROWTH

Data analysis of the tensors showed masks are **cumulative** and **~97–99% of
next-day burned pixels are already burned at day t** (`Y_new_growth_frac` ≈
0.5–3%). So a **persistence** prediction ("next-day = today") scores ~0.98
full-extent IoU while predicting *zero spread*. Reporting full-extent IoU would
make every model look identical and excellent — a leakage-like trap.

Therefore the task here is **next-day new growth**: for pixels **unburned at day
t**, predict whether they ignite at t+1. We score **growth IoU/F1** over that
unburned region, with **persistence (growth IoU = 0)** as the must-beat
reference. Full-extent IoU is reported only as context.

Because an unburned pixel's own mask history is all zeros, the spread signal is
the **burning neighbors** — so each input day contributes a
**neighbor-burned-fraction** feature (`scipy.ndimage.uniform_filter` of the mask
at radii 2 and 8 px), alongside that day's weather and the static terrain.
Positives are ~0.1–0.3% of pixels, so training uses **balanced sampling** (all
positives + `neg_ratio`× negatives) and drops NaN-border weather pixels.

## Deliverables

| File | Purpose |
|------|---------|
| `src/data_loader.py` | Split from `sample_index.csv`; per-fire series; neighbor-front features; balanced growth dataset; train-only `Standardizer`. |
| `src/metrics.py` | Growth IoU/F1, event-level aggregation (skips no-growth fires), persistence references. |
| `src/ablations.py` | **A** temporal window (W=1/2/4) · **B** input modality (fire/weather/terrain/combined). Eval on **val**. |
| `src/compare.py` | Model-agnostic `evaluate_on_test` (any model → next-day mask), persistence + pixel-model rows, `reports/task2_model_comparison.md`. |
| `notebooks/04_task2_ablations_evaluation_colab.ipynb` | Self-contained, Colab-free runnable (logic inlined; only needs Drive). |

## Ablations

- **A — Temporal window:** W = 1 vs 2 vs 4 input days. The SOP's headline
  ablation, now feasible because the data is temporal. (Smaller W also yields
  more samples per fire — noted when interpreting.)
- **B — Input modality:** fire-front-only vs weather-only vs terrain-only vs
  combined — isolates each modality's contribution to spread prediction.

(The old static "weather-aggregation proxy" and "single-vs-multi-task" ablations
are retired: the real window ablation replaces the former, and severity/dNBR is
**not present** in the temporal tensors, so there is no second head.)

## Absolute rules (enforced in code)

- Split from `sample_index.csv` reused unchanged; `assert_no_leakage()` verifies
  each fire is in exactly one split.
- `Standardizer` fit on **train** pixels only.
- Ablations evaluate on **val**; **test** opened only in `src.compare`.
- Growth IoU/F1 reported (never accuracy); persistence baseline always shown.

## How to run

### Colab (recommended)
Open `notebooks/04_...ipynb`, set `TENSOR_DIR` to the Drive tensors folder, run all.

### Local / CLI
```bash
pip install -r requirements.txt          # scikit-learn, xgboost, scipy
python -m src.ablations  --tensor-dir /path/to/tensors      # A/B on val
python -m src.compare --eval-baseline --tensor-dir /path/to/tensors  # test table
```

## Adding a teammate's model to the comparison

Wrap any model as a next-day-mask predictor (works for U-Net / ConvLSTM / Prithvi
or pixel models) and call the shared scorer:

```python
from src import compare
def predict_window(fire_masks_win, weather_win, terrain):   # (W,H,W),(W,6,H,W),(3,H,W)
    ...                                                      # return next_day_mask (H,W)
res = compare.evaluate_on_test("ConvLSTM", predict_window, owner="Student 3",
                               inputs="...", hyperparameters="...", limitations="...")
compare.save_model_result(res)        # -> outputs/models/convlstm.json
```
Then `python -m src.compare` regenerates `reports/task2_model_comparison.md`.

## Caveats / coordination

- `temperature_2m` is in **Kelvin**; `wind_dir_deg` ∈ [−180,180] and `aspect` ∈
  [0,360] are circular (left as-is; could be sin/cos encoded later).
- Some fires have no spread in their windows (zero positives) → skipped in growth
  metrics (NaN-filtered), not errors.
- Student 3 owns the deep-model DataLoader; this pixel/tabular loader shares the
  same split + metrics so numbers stay comparable. Student 1 owns the official
  XGBoost baseline; the pixel model here is an ablation/eval instrument.

## Stretch goal — Prithvi feasibility spike
Time-box ~1 day: load Prithvi-100M, adapt its input to the windowed channels,
attach a tiny next-day-mask head, overfit one fire. If it won't converge on
Colab-free, document the negative result as a model card (a valid SOP deliverable).
