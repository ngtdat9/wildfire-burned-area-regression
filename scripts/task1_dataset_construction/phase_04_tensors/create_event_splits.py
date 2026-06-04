"""
create_event_splits.py
======================
Phase 4 — Event-based train / validation / test split for temporal
fire-spread modeling.

Only includes fire folders under ``data/aligned_compressed/`` that contain
all required layers for temporal tensor construction. Splitting is done at
the **entire wildfire event** level — never by pixel or sample.

Output:
    data/splits/train_events.txt
    data/splits/val_events.txt
    data/splits/test_events.txt
    data/splits/split_summary.json

Usage (from project root):

    python scripts/task1/phase_04_tensors/create_event_splits.py
    python scripts/task1/phase_04_tensors/create_event_splits.py \\
        --train 0.7 --val 0.15 --test 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_INPUT_DIR = Path("data/aligned_compressed")
DEFAULT_SPLITS_DIR = Path("data/splits")

REQUIRED_FILES = [
    Path("labels/fire_mask_stacked_30m_aligned.tif"),
    Path("labels/severity_30m_aligned.tif"),
    Path("terrain/terrain_30m_aligned.tif"),
    Path("weather/weather_stacked_30m_aligned.tif"),
]


def eligible_fire_ids(input_dir: Path) -> list[str]:
    """Return sorted fire IDs with all required temporal tensor inputs."""
    ids: list[str] = []
    if not input_dir.is_dir():
        return ids

    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if all((child / rel).is_file() for rel in REQUIRED_FILES):
            ids.append(child.name)
    return ids


def event_split(
    fire_ids: list[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total:.4f}")

    rng = np.random.default_rng(seed)
    shuffled = list(fire_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 3:
        raise ValueError(
            f"Need at least 3 eligible fires for train/val/test split, found {n}."
        )

    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))
    n_test = n - n_train - n_val

    if n_test < 1:
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        n_test = n - n_train - n_val

    if n_test < 1:
        raise ValueError(
            f"Cannot form non-empty test split with {n} fires. "
            "Adjust fractions or add more fires."
        )

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_outputs(
    train: list[str],
    val: list[str],
    test: list[str],
    splits_dir: Path,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    source_dir: str,
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, ids in [
        ("train_events", train),
        ("val_events", val),
        ("test_events", test),
    ]:
        (splits_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")

    summary = {
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "total_fires": len(train) + len(val) + len(test),
        "source_dir": source_dir,
        "required_files": [str(p) for p in REQUIRED_FILES],
        "note": (
            "Split is at the entire wildfire event level. "
            "Never split by pixel, patch, or temporal sample."
        ),
        "train": train,
        "val": val,
        "test": test,
    }
    (splits_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create event-level train/val/test splits for Phase 4."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        sys.exit(1)

    fire_ids = eligible_fire_ids(args.input_dir)
    if not fire_ids:
        print(
            f"[ERROR] No eligible fires under {args.input_dir}. "
            "Each fire needs fire_mask, severity, terrain, and weather stacks."
        )
        sys.exit(1)

    try:
        train, val, test = event_split(
            fire_ids, args.train, args.val, args.test, args.seed
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    write_outputs(
        train,
        val,
        test,
        args.splits_dir,
        args.train,
        args.val,
        args.test,
        args.seed,
        str(args.input_dir),
    )

    n = len(fire_ids)
    print(f"\n{'=' * 60}")
    print("Event-based split created")
    print(f"{'=' * 60}")
    print(f"Eligible fires : {n}")
    print(f"Seed           : {args.seed}")
    print(f"Train          : {len(train):3d}  ({len(train)/n:.1%})")
    print(f"Val            : {len(val):3d}  ({len(val)/n:.1%})")
    print(f"Test           : {len(test):3d}  ({len(test)/n:.1%})")
    print(f"Output dir     : {args.splits_dir.resolve()}")
    print("  train_events.txt, val_events.txt, test_events.txt")
    print("  split_summary.json")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
