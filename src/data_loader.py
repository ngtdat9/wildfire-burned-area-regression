"""
data_loader.py
==============
Task 2 — shared loader for the TEMPORAL fire-spread tensors (Student 4).

Dataset format (Phase 4 temporal pipeline, `mode: temporal_fire_spread`):

    tensors/<split>/<fire_id>.npz
        fire_masks  (N_days, H, W)    uint8    daily CUMULATIVE burned mask
        weather     (N_days, 6, H, W) float32  daily ERA5 (6 vars)
        terrain     (3, H, W)         float32  static elevation/slope/aspect
    tensors/sample_index.csv      one row per fire (fire_id, split, n_fire_days, ...)
    tensors/tensor_manifest.json  window_size, var names, per-split counts

Sliding window (W = window_size, default 4):
    X = days [i .. i+W-1],   Y = fire_masks[i+W]   (next-day mask)

Two facts from data analysis drive the modeling here:

1. Masks are cumulative and ~97-99% of next-day burned pixels are ALREADY
   burned at day t. So we predict the **new growth only**, restricted to pixels
   that are UNBURNED at day t. A persistence baseline ("no new growth") is the
   reference every model must beat (see src/metrics.py).
2. For an unburned pixel, its own mask history is all zeros — the spread signal
   comes from BURNING NEIGHBORS. So each input day contributes a
   **neighbor-burned-fraction** feature (local mean of the mask) at one or more
   radii, alongside that day's weather and the static terrain.

Configuring the Drive path (Colab):
    from src import data_loader as dl
    dl.configure(tensor_dir="/content/drive/MyDrive/bigdata/tensors")
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import uniform_filter
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_TENSOR_DIR = Path(os.environ.get("WILDFIRE_TENSOR_DIR", "data/tensors"))
_TENSOR_DIR = DEFAULT_TENSOR_DIR

# Spatial downsample. MODIS fire labels are ~500 m so the 30 m grid is ~16x
# oversampled; STRIDE=4 (-> 120 m) loses little and runs ~16x faster. Set to 1
# for full resolution.
STRIDE = int(os.environ.get("WILDFIRE_STRIDE", 4))

WEATHER_VAR_NAMES = ["temperature_2m", "relative_humidity_pct", "wind_speed_ms",
                     "wind_dir_deg", "surface_pressure", "total_precipitation"]
TERRAIN_BAND_NAMES = ["elevation_m", "slope_deg", "aspect_deg"]
N_WEATHER_VARS = 6
DEFAULT_WINDOW = 4
DEFAULT_RADII = (2, 8)          # neighbor-fraction windows: 5x5 and 17x17 px

MODALITIES = ("fire", "weather", "terrain", "combined")


def configure(tensor_dir=None, stride=None) -> None:
    global _TENSOR_DIR, STRIDE
    if tensor_dir is not None:
        _TENSOR_DIR = Path(tensor_dir)
    if stride is not None:
        STRIDE = int(stride)


def tensor_dir() -> Path:
    return _TENSOR_DIR


# --------------------------------------------------------------------------
# Split index
# --------------------------------------------------------------------------
def load_index() -> list[dict]:
    """Return the per-fire rows from sample_index.csv (or the manifest)."""
    idx = _TENSOR_DIR / "sample_index.csv"
    if idx.exists():
        with open(idx, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:                              # normalize numeric fields
            for k in ("n_fire_days", "n_samples", "height", "width",
                      "weather_offset"):
                if k in r and r[k] != "":
                    r[k] = int(r[k])
        return rows
    man = _TENSOR_DIR / "tensor_manifest.json"
    if man.exists():
        fires = json.loads(man.read_text()).get("fires", [])
        return [r for r in fires if r.get("success", True)]
    raise FileNotFoundError(
        f"No sample_index.csv or tensor_manifest.json in {_TENSOR_DIR}. "
        "Call configure(tensor_dir=...) to point at the Drive tensors folder.")


def split_fires(split: str) -> list[dict]:
    """Per-fire rows for one split ('train'|'val'|'test')."""
    return [r for r in load_index() if r.get("split") == split]


def assert_no_leakage() -> None:
    seen: dict[str, str] = {}
    for r in load_index():
        fid, sp = r["fire_id"], r.get("split")
        if fid in seen and seen[fid] != sp:
            raise AssertionError(f"Leakage: {fid} in {seen[fid]} and {sp}")
        seen[fid] = sp


def fire_path(row: dict) -> Path:
    p = Path(row.get("tensor_file", ""))
    if p.exists():
        return p
    return _TENSOR_DIR / row["split"] / f"{row['fire_id']}.npz"


# --------------------------------------------------------------------------
# Per-fire arrays
# --------------------------------------------------------------------------
def load_fire_series(row: dict, mmap: bool = True) -> dict:
    """Load one fire's time series, downsampled by STRIDE.

    The .npz are saved with savez_compressed, so mmap does NOT help (accessing a
    key decompresses the whole array). We slice + COPY immediately and close the
    file so only the small downsampled arrays stay resident — a strided view
    would otherwise pin the full multi-GB array in RAM. For very large fires,
    prefer building a downsampled cache first (see the Colab notebook)."""
    z = np.load(fire_path(row))
    s = slice(None, None, STRIDE)
    out = {"fire_masks": np.ascontiguousarray(z["fire_masks"][:, s, s]),
           "weather": np.ascontiguousarray(z["weather"][:, :, s, s]),
           "terrain": np.ascontiguousarray(z["terrain"][:, s, s])}
    z.close()
    return out


# Fires skipped at load time (corrupted/unreadable .npz) are recorded here.
SKIPPED_FIRES: list[tuple] = []


def safe_load_series(row: dict, mmap: bool = True):
    """Load a fire, returning None (and recording it) if the .npz is corrupted
    or unreadable — so one bad file never crashes a training/eval run."""
    try:
        s = load_fire_series(row, mmap=mmap)
        _ = s["fire_masks"].shape          # force the zip header to be read
        return s
    except Exception as e:                  # BadZipFile, truncated upload, etc.
        SKIPPED_FIRES.append((row.get("fire_id"), row.get("split"),
                              type(e).__name__))
        print(f"  [skip] {row.get('fire_id')} [{row.get('split')}]: "
              f"{type(e).__name__} (corrupted/unreadable .npz)")
        return None


def validate_index(rows=None, window=DEFAULT_WINDOW) -> dict:
    """Scan fires and bucket them: ok / empty (no burned pixels) / corrupt.
    Use to prune sample_index.csv or just report dataset health."""
    rows = rows if rows is not None else load_index()
    ok, empty, corrupt = [], [], []
    for r in rows:
        s = safe_load_series(r)
        if s is None:
            corrupt.append(r["fire_id"]); continue
        fm = np.asarray(s["fire_masks"])
        (empty if float(fm[-1].mean()) < 1e-5 else ok).append(r["fire_id"])
    return {"ok": ok, "empty": empty, "corrupt": corrupt}


def neighbor_fraction(mask2d: np.ndarray, radius: int) -> np.ndarray:
    """Fraction of burned pixels in a (2*radius+1) square window — the
    fire-front proximity signal."""
    m = mask2d.astype("float32")
    if _HAVE_SCIPY:
        return uniform_filter(m, size=2 * radius + 1, mode="constant")
    # Fallback: integral-image box mean (no scipy).
    H, W = m.shape
    ii = np.zeros((H + 1, W + 1), "float64")
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    out = np.zeros_like(m)
    for y in range(H):
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        for x in range(W):
            x0, x1 = max(0, x - radius), min(W, x + radius + 1)
            s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            out[y, x] = s / ((y1 - y0) * (x1 - x0))
    return out


# --------------------------------------------------------------------------
# Feature schema
# --------------------------------------------------------------------------
def feature_names(window: int, radii=DEFAULT_RADII,
                  modality: str = "combined") -> list[str]:
    names: list[str] = []
    use_fire = modality in ("fire", "combined")
    use_wx = modality in ("weather", "combined")
    use_terr = modality in ("terrain", "combined")
    for d in range(window):                          # day t-(W-1) .. t
        lag = window - 1 - d
        if use_fire:
            for r in radii:
                names.append(f"firefrac_r{r}_lag{lag}")
        if use_wx:
            for v in WEATHER_VAR_NAMES:
                names.append(f"{v}_lag{lag}")
    if use_terr:
        names += list(TERRAIN_BAND_NAMES)
    return names


# --------------------------------------------------------------------------
# Build a pixel dataset over the UNBURNED region (growth prediction)
# --------------------------------------------------------------------------
def _window_features(series, i, window, radii, modality, sel_flat, nf_cache=None):
    """Feature matrix for the chosen pixels (flat indices) of one window.

    nf_cache: optional {day: [flat neighbor-fraction map per radius]} so the
    neighbor maps are computed once per fire instead of once per window."""
    fm = series["fire_masks"]
    H, W = fm.shape[-2:]
    use_fire = modality in ("fire", "combined")
    use_wx = modality in ("weather", "combined")
    use_terr = modality in ("terrain", "combined")

    cols = []
    for d in range(i, i + window):
        if use_fire:
            maps = (nf_cache[d] if nf_cache is not None
                    else [neighbor_fraction(np.asarray(fm[d]), r).reshape(-1) for r in radii])
            for nf in maps:
                cols.append(nf[sel_flat])
        if use_wx:
            wd = np.asarray(series["weather"][d]).reshape(N_WEATHER_VARS, -1)
            for v in range(N_WEATHER_VARS):
                cols.append(wd[v][sel_flat])
    if use_terr:
        tr = np.asarray(series["terrain"]).reshape(3, -1)
        for b in range(3):
            cols.append(tr[b][sel_flat])
    return np.stack(cols, axis=1).astype("float32")


def window_features(series, i, window, radii, modality, sel_flat):
    """Public: feature matrix for given flat pixel indices of one window.
    Used at evaluation time to predict over the full unburned region."""
    return _window_features(series, i, window, radii, modality, sel_flat)


def build_pixel_dataset(fire_rows, window=DEFAULT_WINDOW, radii=DEFAULT_RADII,
                        modality="combined", max_pos_per_fire=8_000,
                        neg_ratio=4.0, seed=42, return_fire_index=False):
    """Build a balanced pixel dataset for NEXT-DAY GROWTH prediction.

    For every sliding window of every fire, consider only pixels UNBURNED at
    day t. Target = 1 if the pixel becomes burned at t+1 (new growth), else 0.
    Keep all positives (capped) and `neg_ratio`x as many sampled negatives.
    Pixels with any NaN feature (weather border nodata) are dropped.

    Returns X (N,F) float32, y (N,) uint8 [, fire_index (N,) int32].
    """
    rng = np.random.default_rng(seed)
    Xs, ys, idxs = [], [], []

    for fi, row in enumerate(fire_rows):
        series = safe_load_series(row)
        if series is None:                            # corrupted .npz — skip
            continue
        fm = series["fire_masks"]
        n_days = fm.shape[0]
        use_fire = modality in ("fire", "combined")
        nf_cache = ({d: [neighbor_fraction(np.asarray(fm[d]), r).reshape(-1) for r in radii]
                     for d in range(n_days)} if use_fire else None)
        for i in range(n_days - window):
            mask_t = np.asarray(fm[i + window - 1]).reshape(-1).astype(bool)
            y_next = np.asarray(fm[i + window]).reshape(-1).astype(bool)
            region = ~mask_t                          # unburned at day t
            growth = region & y_next                  # became burned at t+1
            pos = np.where(growth)[0]
            neg = np.where(region & ~y_next)[0]
            if pos.size == 0:
                continue                              # no spread in this window
            if pos.size > max_pos_per_fire:
                pos = rng.choice(pos, max_pos_per_fire, replace=False)
            n_neg = min(neg.size, int(neg_ratio * pos.size))
            if n_neg > 0:
                neg = rng.choice(neg, n_neg, replace=False)
            sel = np.concatenate([pos, neg])
            feats = _window_features(series, i, window, radii, modality, sel, nf_cache)

            valid = np.isfinite(feats).all(axis=1)    # drop NaN-border pixels
            feats = feats[valid]
            lab = np.zeros(sel.size, "uint8"); lab[:pos.size] = 1
            lab = lab[valid]

            Xs.append(feats); ys.append(lab)
            if return_fire_index:
                idxs.append(np.full(lab.size, fi, "int32"))

    if not Xs:
        nf = len(feature_names(window, radii, modality))
        empty = (np.empty((0, nf), "float32"), np.empty((0,), "uint8"))
        return (*empty, np.empty((0,), "int32")) if return_fire_index else empty

    X = np.concatenate(Xs); y = np.concatenate(ys)
    if return_fire_index:
        return X, y, np.concatenate(idxs)
    return X, y


# --------------------------------------------------------------------------
# Standardizer (numpy-only; FIT ON TRAIN ONLY)
# --------------------------------------------------------------------------
class Standardizer:
    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        s = np.nanstd(X, axis=0); s[s == 0] = 1.0; self.std_ = s
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
