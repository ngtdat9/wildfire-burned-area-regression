"""
ablations.py
============
Task 2 — Student 4 ablation studies on the TEMPORAL fire-spread data.

Task: predict next-day NEW GROWTH (pixels unburned at day t that ignite at t+1)
over the unburned region. One fast shared model (XGBoost → sklearn fallback),
the committed event-level split, the shared loader/metrics. Evaluate on **val**;
the test split is reserved for the final table (src/compare.py).

Two ablations (each changes one factor, split + loader fixed):

  A. Temporal window   W = 1 vs 2 vs 4 input days  (the SOP's headline ablation,
                       now feasible because the data is temporal)
  B. Input modality    fire-front vs weather vs terrain vs combined

A persistence reference (growth IoU = 0) is reported alongside so the numbers
are interpretable.

Run:
    python -m src.ablations
    python -m src.ablations --max-pos-per-fire 20000 --neg-ratio 5
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import data_loader as dl
from . import metrics as M

OUTPUT_DIR = Path("outputs/ablations")


def get_classifier(seed: int = 42):
    """XGBoost if available, else sklearn HistGradientBoosting, else LogReg."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed), "xgboost"
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=seed), "sklearn.HGB"
    except Exception:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, n_jobs=-1,
                                  class_weight="balanced", random_state=seed), "sklearn.LogReg"


def _train_eval(train_rows, val_rows, window, modality, max_pos, neg_ratio, seed):
    t0 = time.time()
    Xtr, ytr = dl.build_pixel_dataset(
        train_rows, window=window, modality=modality,
        max_pos_per_fire=max_pos, neg_ratio=neg_ratio, seed=seed)
    if Xtr.shape[0] == 0 or len(np.unique(ytr)) < 2:
        return {"window": window, "modality": modality, "error": "no usable training samples"}
    sc = dl.Standardizer().fit(Xtr)                  # fit on TRAIN only
    clf, backend = get_classifier(seed)
    clf.fit(sc.transform(Xtr), ytr)

    Xv, yv, fidx = dl.build_pixel_dataset(
        val_rows, window=window, modality=modality,
        max_pos_per_fire=max_pos, neg_ratio=neg_ratio, seed=seed + 1,
        return_fire_index=True)
    pred = clf.predict(sc.transform(Xv))
    g = M.growth_by_fire(yv, pred, fidx)
    return {
        "window": window, "modality": modality, "backend": backend,
        "n_features": Xtr.shape[1], "n_train_px": int(Xtr.shape[0]),
        "growth_iou_event": g["iou"]["mean"],
        "growth_f1_event": g["f1"]["mean"],
        "growth_iou_pooled": M.iou(yv, pred),
        "n_val_fires_scored": g["iou"]["n_fires_scored"],
        "seconds": round(time.time() - t0, 1),
    }


def ablation_window(train_rows, val_rows, max_pos, neg_ratio, seed):
    return {"ablation": "A_temporal_window",
            "results": [_train_eval(train_rows, val_rows, w, "combined",
                                    max_pos, neg_ratio, seed)
                        for w in (1, 2, 4)]}


def ablation_modality(train_rows, val_rows, max_pos, neg_ratio, seed):
    return {"ablation": "B_input_modality",
            "results": [_train_eval(train_rows, val_rows, 4, m,
                                    max_pos, neg_ratio, seed)
                        for m in dl.MODALITIES]}


def run_all(max_pos=20_000, neg_ratio=5.0, seed=42, output_dir=OUTPUT_DIR):
    dl.assert_no_leakage()
    train_rows, val_rows = dl.split_fires("train"), dl.split_fires("val")
    print(f"Ablations: train={len(train_rows)} fires, val={len(val_rows)} fires")

    # Persistence reference (full-extent IoU is inflated; growth IoU is 0).
    _pf = [M.persistence_full_extent_iou(s["fire_masks"], window=4)
           for s in (dl.safe_load_series(r) for r in val_rows) if s is not None]
    pers_full = float(np.nanmean(_pf)) if _pf else float("nan")

    report = {
        "meta": {"task": "next_day_growth", "seed": seed,
                 "max_pos_per_fire": max_pos, "neg_ratio": neg_ratio,
                 "n_train_fires": len(train_rows), "n_val_fires": len(val_rows),
                 "python": platform.python_version()},
        "persistence_reference": {
            "growth_iou": M.persistence_growth_iou(),       # 0.0 by definition
            "full_extent_iou_val_mean": pers_full,          # ~0.97, for context
            "note": "Persistence predicts no spread: growth IoU=0 but full-extent "
                    "IoU is high — why we score growth, not full extent."},
        "A_temporal_window": ablation_window(train_rows, val_rows, max_pos, neg_ratio, seed),
        "B_input_modality": ablation_modality(train_rows, val_rows, max_pos, neg_ratio, seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ablation_results.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {out}")
    return report


def main():
    p = argparse.ArgumentParser(description="Student 4 temporal ablations.")
    p.add_argument("--max-pos-per-fire", type=int, default=20_000)
    p.add_argument("--neg-ratio", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--tensor-dir")
    args = p.parse_args()
    if args.tensor_dir:
        dl.configure(tensor_dir=args.tensor_dir)
    rep = run_all(max_pos=args.max_pos_per_fire, neg_ratio=args.neg_ratio,
                  seed=args.seed, output_dir=args.output_dir)
    print("\n=== A. Temporal window (growth IoU, event-mean) ===")
    for r in rep["A_temporal_window"]["results"]:
        print(f"  W={r.get('window')}: IoU={r.get('growth_iou_event')}  "
              f"F1={r.get('growth_f1_event')}  [{r.get('backend')}]")
    print("=== B. Input modality (W=4) ===")
    for r in rep["B_input_modality"]["results"]:
        print(f"  {r.get('modality'):9s}: IoU={r.get('growth_iou_event')}  "
              f"F1={r.get('growth_f1_event')}")


if __name__ == "__main__":
    main()
