"""
metrics.py
==========
Task 2 — shared evaluation for the TEMPORAL next-day fire-spread task (Student 4).

The prediction target is **new growth**: pixels unburned at day t that become
burned at t+1. We score the binary growth prediction over that unburned region.

Why not full-extent IoU? Masks are cumulative and ~97-99% of next-day burned
pixels are already burned at day t, so a "persistence" prediction (next-day =
today) scores ~0.98 full-extent IoU while predicting zero actual spread. The
honest, discriminative metric is **growth IoU/F1 over the unburned region**, and
the reference baseline is persistence (which gets growth IoU = 0).

All metrics are pure numpy and aggregate **per fire, then average** so big fires
don't dominate. Fires with no growth in a window are skipped (NaN-filtered).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------
# Binary growth metrics
# --------------------------------------------------------------------------
def binary_counts(y_true, y_pred):
    yt = np.asarray(y_true).astype(bool).ravel()
    yp = np.asarray(y_pred).astype(bool).ravel()
    return {"tp": int((yt & yp).sum()), "fp": int((~yt & yp).sum()),
            "fn": int((yt & ~yp).sum()), "tn": int((~yt & ~yp).sum())}


def iou(y_true, y_pred):
    """Jaccard for the positive (new-growth) class."""
    c = binary_counts(y_true, y_pred)
    d = c["tp"] + c["fp"] + c["fn"]
    return c["tp"] / d if d > 0 else float("nan")


def f1_binary(y_true, y_pred):
    c = binary_counts(y_true, y_pred)
    p = c["tp"] / (c["tp"] + c["fp"] + EPS)
    r = c["tp"] / (c["tp"] + c["fn"] + EPS)
    return 2 * p * r / (p + r + EPS)


def precision_recall(y_true, y_pred):
    c = binary_counts(y_true, y_pred)
    return (c["tp"] / (c["tp"] + c["fp"] + EPS),
            c["tp"] / (c["tp"] + c["fn"] + EPS))


def area_bias(y_true, y_pred):
    """Growth-area bias: predicted vs actual new-growth pixels."""
    t = int(np.asarray(y_true).astype(bool).sum())
    p = int(np.asarray(y_pred).astype(bool).sum())
    return {"true_px": t, "pred_px": p, "rel_bias": (p - t) / (t + EPS)}


def growth_metrics(y_true, y_pred):
    p, r = precision_recall(y_true, y_pred)
    m = {"iou": iou(y_true, y_pred), "f1": f1_binary(y_true, y_pred),
         "precision": p, "recall": r}
    m.update(area_bias(y_true, y_pred))
    return m


# --------------------------------------------------------------------------
# Event-level aggregation (per fire, then average; skips NaN fires)
# --------------------------------------------------------------------------
def aggregate_by_fire(y_true, y_pred, fire_index, metric_fn):
    fire_index = np.asarray(fire_index)
    per = {}
    for fid in np.unique(fire_index):
        sel = fire_index == fid
        per[int(fid)] = metric_fn(y_true[sel], y_pred[sel])
    vals = np.array([v for v in per.values() if np.isfinite(v)])
    return {"mean": float(vals.mean()) if vals.size else float("nan"),
            "std": float(vals.std()) if vals.size else float("nan"),
            "per_fire": per, "n_fires_scored": int(vals.size)}


def growth_by_fire(y_true, y_pred, fire_index):
    """Event-level growth IoU and F1 (per fire, then averaged)."""
    return {"iou": aggregate_by_fire(y_true, y_pred, fire_index, iou),
            "f1": aggregate_by_fire(y_true, y_pred, fire_index, f1_binary)}


# --------------------------------------------------------------------------
# Persistence baseline (the reference every model must beat)
# --------------------------------------------------------------------------
def persistence_full_extent_iou(fire_masks, window):
    """Full-extent IoU of 'next-day = today' across a fire's windows.

    Demonstrates why full-extent IoU is misleading — this is typically ~0.97+
    while predicting zero spread. Report it for context only.
    """
    fm = np.asarray(fire_masks)
    vals = []
    for i in range(fm.shape[0] - window):
        mask_t = fm[i + window - 1].astype(bool)
        y_next = fm[i + window].astype(bool)
        vals.append(iou(y_next, mask_t))
    v = np.array([x for x in vals if np.isfinite(x)])
    return float(v.mean()) if v.size else float("nan")


def persistence_growth_iou():
    """Persistence predicts NO new growth, so its growth IoU is exactly 0.
    Provided as a named constant for the comparison table."""
    return 0.0
