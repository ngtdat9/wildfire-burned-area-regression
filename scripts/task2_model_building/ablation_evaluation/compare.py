"""
compare.py
==========
Task 2 — final evaluation support for the TEMPORAL fire-spread task (Student 4).

One model-agnostic interface for scoring ANY architecture (XGBoost, U-Net,
ConvLSTM, Prithvi) on the locked test fires, so the comparison is fair. A model
is wrapped as:

    predict_window(fire_masks_win (W,H,W), weather_win (W,6,H,W),
                   terrain (3,H,W)) -> next_day_mask (H,W) in {0,1}

For each window we score the **new growth** over the unburned region
(growth = next-day burned AND unburned at day t), event-level (per fire, then
averaged). Full-extent IoU is reported only for context. The persistence
baseline (next-day = today) is included as the must-beat reference.

The test split is opened ONLY here.

Per-model JSON (also writable by hand by a teammate reporting numbers):
    {"model": "ConvLSTM", "owner": "Student 3",
     "growth_iou": 0.21, "growth_f1": 0.30, "full_extent_iou": 0.98,
     "inputs": "...", "hyperparameters": "...", "limitations": "..."}

Run:
    python -m src.compare --eval-baseline      # score Student 4 pixel model + persistence
    python -m src.compare                       # just regenerate the report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data_loader as dl
from . import metrics as M

MODELS_DIR = Path("outputs/models")
REPORT_PATH = Path("reports/task2_model_comparison.md")


# --------------------------------------------------------------------------
# Model-agnostic test scoring
# --------------------------------------------------------------------------
def evaluate_on_test(model_name, predict_window, owner="", window=4,
                     test_rows=None, **card):
    """Score a next-day-mask predictor on the test fires (growth, event-level)."""
    if test_rows is None:
        test_rows = dl.split_fires("test")

    iou_pf, f1_pf, full_pf = [], [], []
    for row in test_rows:
        s = dl.safe_load_series(row)
        if s is None:                                 # corrupted .npz — skip
            continue
        fm = np.asarray(s["fire_masks"])
        n_days, H, W = fm.shape
        gt_all, gp_all, full_vals = [], [], []
        for i in range(n_days - window):
            mask_t = fm[i + window - 1].astype(bool)
            y_next = fm[i + window].astype(bool)
            pred = predict_window(fm[i:i + window], np.asarray(s["weather"][i:i + window]),
                                  np.asarray(s["terrain"])).astype(bool)
            region = ~mask_t
            gt_all.append((y_next & region).ravel())
            gp_all.append((pred & region).ravel())
            full_vals.append(M.iou(y_next, pred))
        if not gt_all:
            continue
        gt = np.concatenate(gt_all); gp = np.concatenate(gp_all)
        if gt.sum() == 0:                            # no spread in this fire
            continue
        iou_pf.append(M.iou(gt, gp)); f1_pf.append(M.f1_binary(gt, gp))
        full_pf.append(float(np.nanmean(full_vals)))

    def _m(x):
        a = np.array([v for v in x if np.isfinite(v)])
        return float(a.mean()) if a.size else float("nan")

    res = {"model": model_name, "owner": owner,
           "growth_iou": _m(iou_pf), "growth_f1": _m(f1_pf),
           "full_extent_iou": _m(full_pf), "n_test_fires_scored": len(iou_pf),
           "evaluated_at": time.strftime("%Y-%m-%d %H:%M")}
    res.update(card)
    return res


def save_model_result(result, models_dir=MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    slug = result["model"].lower().replace(" ", "_").replace("/", "_")
    p = models_dir / f"{slug}.json"
    p.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Persistence baseline + Student 4 pixel model
# --------------------------------------------------------------------------
def persistence_result(window=4):
    def predict_window(fire_masks_win, weather_win, terrain):
        return fire_masks_win[-1]                    # next-day = today (no spread)
    return evaluate_on_test(
        "Persistence (baseline)", predict_window, owner="reference",
        window=window, inputs="today's mask", hyperparameters="none",
        limitations="Predicts zero spread; growth IoU = 0 by construction. "
                    "High full-extent IoU shows why we score growth.")


def evaluate_pixel_baseline(window=4, modality="combined",
                            radii=dl.DEFAULT_RADII, max_pos=20_000,
                            neg_ratio=5.0, seed=42, chunk=500_000):
    """Train Student 4's pixel growth model on train; return its test result."""
    from .ablations import get_classifier
    train_rows = dl.split_fires("train")
    Xtr, ytr = dl.build_pixel_dataset(train_rows, window=window, radii=radii,
                                      modality=modality, max_pos_per_fire=max_pos,
                                      neg_ratio=neg_ratio, seed=seed)
    sc = dl.Standardizer().fit(Xtr)
    clf, backend = get_classifier(seed); clf.fit(sc.transform(Xtr), ytr)

    def predict_window(fire_masks_win, weather_win, terrain):
        H, W = fire_masks_win.shape[-2:]
        mask_t = fire_masks_win[-1].astype(bool)
        out = mask_t.copy().ravel()                  # already-burned stays burned
        region = np.where(~mask_t.ravel())[0]
        series = {"fire_masks": fire_masks_win, "weather": weather_win,
                  "terrain": terrain}
        for s0 in range(0, region.size, chunk):      # chunk to bound memory
            sel = region[s0:s0 + chunk]
            feats = dl.window_features(series, 0, window, radii, modality, sel)
            valid = np.isfinite(feats).all(axis=1)
            pr = np.zeros(sel.size, bool)
            if valid.any():
                pr[valid] = clf.predict(sc.transform(feats[valid])).astype(bool)
            out[sel] = pr
        return out.reshape(H, W)

    return evaluate_on_test(
        f"Pixel growth model (Student 4)", predict_window, owner="Student 4",
        window=window,
        inputs=f"unburned-region pixels: {window}-day fire-front + weather + terrain",
        hyperparameters=f"{backend}, modality={modality}, radii={radii}, "
                        f"neg_ratio={neg_ratio}",
        limitations="Pixel-level (no learned spatial decoder); ablation/eval "
                    "instrument, not the official XGBoost baseline (Student 1).")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _fmt(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "—"


def render_report(models_dir=MODELS_DIR, report_path=REPORT_PATH):
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(models_dir.glob("*.json"))] if models_dir.exists() else []

    lines = [
        "# Task 2 — Model Comparison (next-day fire spread, event-level test set)",
        "",
        "Scored on the **new-growth** target (pixels unburned at day t that ignite "
        "at t+1) over the locked test fires, via shared `src/data_loader.py` + "
        "`src/metrics.py`. **Growth IoU/F1** are the headline metrics; full-extent "
        "IoU is context only (persistence inflates it). Every model should beat "
        "**Persistence (growth IoU = 0)**.",
        "",
        "| Model | Owner | Growth IoU | Growth F1 | Full-extent IoU (context) |",
        "|-------|-------|-----------:|----------:|--------------------------:|",
    ]
    if not rows:
        lines.append("| _(no results yet — drop JSONs in `outputs/models/`)_ | | | | |")
    for r in rows:
        lines.append(f"| {r.get('model','?')} | {r.get('owner','')} "
                     f"| {_fmt(r.get('growth_iou'))} | {_fmt(r.get('growth_f1'))} "
                     f"| {_fmt(r.get('full_extent_iou'))} |")

    lines += ["", "## Model cards", ""]
    for r in rows:
        lines += [
            f"### {r.get('model','?')}  ({r.get('owner','')})",
            f"- **Inputs:** {r.get('inputs','—')}",
            f"- **Hyperparameters:** {r.get('hyperparameters','—')}",
            f"- **Growth:** IoU {_fmt(r.get('growth_iou'))}, F1 {_fmt(r.get('growth_f1'))}"
            f"  | full-extent IoU {_fmt(r.get('full_extent_iou'))}",
            f"- **Limitations:** {r.get('limitations','—')}", ""]

    lines += [
        "## How to add your model",
        "```python",
        "from src import compare",
        "def predict_window(fire_masks_win, weather_win, terrain):  # (W,H,W),(W,6,H,W),(3,H,W)",
        "    ...                                                     # return next_day_mask (H,W)",
        "res = compare.evaluate_on_test('U-Net', predict_window, owner='Student 2',",
        "                               inputs='...', hyperparameters='...', limitations='...')",
        "compare.save_model_result(res)",
        "```",
        "Then run `python -m src.compare` to regenerate this table.", ""]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    p = argparse.ArgumentParser(description="Task 2 comparison report (temporal).")
    p.add_argument("--eval-baseline", action="store_true",
                   help="Score Student 4 pixel model + persistence on test first.")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--tensor-dir")
    args = p.parse_args()
    if args.tensor_dir:
        dl.configure(tensor_dir=args.tensor_dir)
    if args.eval_baseline:
        save_model_result(persistence_result(window=args.window))
        save_model_result(evaluate_pixel_baseline(window=args.window))
        print("Saved persistence + pixel-model results.")
    print(f"Wrote {render_report()}")


if __name__ == "__main__":
    main()
