# -*- coding: utf-8 -*-
"""
Compute normalization statistics for U-Net weather and terrain channels.

Output JSON keys:
    weather_mean
    weather_std
    terrain_mean
    terrain_std
    weather_var_names
    terrain_band_names

This script computes stats from tensors_npy_cache/train by default.
Fire masks are not normalized.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CACHE_ROOT, STATS_PATH


WEATHER_VAR_NAMES = [
    "temperature_2m",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "surface_pressure",
    "total_precipitation",
]

TERRAIN_BAND_NAMES = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
]


def update_stats_from_array(arr, sums, sq_sums, counts, channel_idx):
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(arr)

    if not mask.any():
        return

    valid = arr[mask]

    sums[channel_idx] += valid.sum()
    sq_sums[channel_idx] += np.square(valid).sum()
    counts[channel_idx] += valid.size


def compute_stats(cache_root: Path, split: str = "train"):
    split_dir = cache_root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    fire_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

    if not fire_dirs:
        raise RuntimeError(f"No fire folders found in: {split_dir}")

    weather_sum = np.zeros(6, dtype=np.float64)
    weather_sq_sum = np.zeros(6, dtype=np.float64)
    weather_count = np.zeros(6, dtype=np.float64)

    terrain_sum = np.zeros(3, dtype=np.float64)
    terrain_sq_sum = np.zeros(3, dtype=np.float64)
    terrain_count = np.zeros(3, dtype=np.float64)

    for fire_dir in tqdm(fire_dirs, desc=f"Computing stats from {split}"):
        weather_path = fire_dir / "weather.npy"
        terrain_path = fire_dir / "terrain.npy"

        if not weather_path.exists() or not terrain_path.exists():
            continue

        weather = np.load(weather_path, mmap_mode="r")  # [N, 6, H, W]
        terrain = np.load(terrain_path, mmap_mode="r")  # [3, H, W]

        n_days = weather.shape[0]

        for day in range(n_days):
            for c in range(6):
                update_stats_from_array(
                    arr=weather[day, c],
                    sums=weather_sum,
                    sq_sums=weather_sq_sum,
                    counts=weather_count,
                    channel_idx=c,
                )

        for c in range(3):
            update_stats_from_array(
                arr=terrain[c],
                sums=terrain_sum,
                sq_sums=terrain_sq_sum,
                counts=terrain_count,
                channel_idx=c,
            )

    weather_mean = weather_sum / np.maximum(weather_count, 1)
    weather_var = weather_sq_sum / np.maximum(weather_count, 1) - np.square(weather_mean)
    weather_std = np.sqrt(np.maximum(weather_var, 1e-12))

    terrain_mean = terrain_sum / np.maximum(terrain_count, 1)
    terrain_var = terrain_sq_sum / np.maximum(terrain_count, 1) - np.square(terrain_mean)
    terrain_std = np.sqrt(np.maximum(terrain_var, 1e-12))

    return {
        "split": split,
        "weather_var_names": WEATHER_VAR_NAMES,
        "terrain_band_names": TERRAIN_BAND_NAMES,
        "weather_mean": weather_mean.tolist(),
        "weather_std": weather_std.tolist(),
        "terrain_mean": terrain_mean.tolist(),
        "terrain_std": terrain_std.tolist(),
        "weather_count": weather_count.tolist(),
        "terrain_count": terrain_count.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute U-Net normalization stats.")

    parser.add_argument("--cache-root", type=str, default=str(CACHE_ROOT))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-json", type=str, default=str(STATS_PATH))

    return parser.parse_args()


def main():
    args = parse_args()

    cache_root = Path(args.cache_root)
    output_json = Path(args.output_json)

    print("=" * 80)
    print("Compute U-Net Normalization Stats")
    print("=" * 80)
    print("Cache root:", cache_root)
    print("Split:", args.split)
    print("Output JSON:", output_json)

    stats = compute_stats(
        cache_root=cache_root,
        split=args.split,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved stats:")
    print(output_json)

    print("\nWeather stats:")
    for name, mean, std in zip(
        stats["weather_var_names"],
        stats["weather_mean"],
        stats["weather_std"],
    ):
        print(f"{name:25s} mean={mean:.4f}, std={std:.4f}")

    print("\nTerrain stats:")
    for name, mean, std in zip(
        stats["terrain_band_names"],
        stats["terrain_mean"],
        stats["terrain_std"],
    ):
        print(f"{name:25s} mean={mean:.4f}, std={std:.4f}")


if __name__ == "__main__":
    main()