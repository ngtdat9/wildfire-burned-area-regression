"""
build_splits.py
===============
Phase 4 — Event-based train / validation / test split.

Splits wildfire event IDs into train, validation, and test sets.
Splitting is done at the FIRE EVENT level — never by pixel or patch —
to prevent spatial information leakage between splits.

Input (in priority order):
  1. data/tensors/dataset_manifest.json  (fires with success=True)
  2. data/aligned_compressed/compression_report.csv  (fires with fire_complete=True)

Output:
  data/splits/train.txt    — one fire_id per line
  data/splits/val.txt
  data/splits/test.txt
  data/splits/splits.json  — full split manifest with counts and metadata

Usage (from project root):

    python scripts/build_splits.py
    python scripts/build_splits.py --train 0.7 --val 0.15 --test 0.15
    python scripts/build_splits.py --seed 99
    python scripts/build_splits.py --splits-dir data/splits
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_MANIFEST   = Path("data/tensors/dataset_manifest.json")
DEFAULT_REPORT_CSV = Path("data/aligned_compressed/compression_report.csv")
DEFAULT_SPLITS_DIR = Path("data/splits")


def load_from_manifest(path: Path) -> list[str]:
    with open(path) as f:
        manifest = json.load(f)
    return [e["fire_id"] for e in manifest.get("fires", []) if e.get("success")]


def load_from_report(path: Path) -> list[str]:
    seen: dict[str, bool] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fid = row["fire_id"]
            ok  = row.get("fire_complete", "").strip().lower() == "true"
            seen[fid] = seen.get(fid, True) and ok
    return [fid for fid, ok in seen.items() if ok]


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
    n_train = max(1, round(n * train_frac))
    n_val   = max(1, round(n * val_frac))
    n_test  = n - n_train - n_val

    if n_test < 1:
        n_val  = max(0, n_val - 1)
        n_test = 1

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]
    return train, val, test


def write_splits(
    train: list[str],
    val: list[str],
    test: list[str],
    splits_dir: Path,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in [("train", train), ("val", val), ("test", test)]:
        (splits_dir / f"{split_name}.txt").write_text("\n".join(ids) + "\n")

    manifest = {
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "counts":    {"train": len(train),  "val": len(val),  "test": len(test)},
        "note": (
            "Split is at the fire-event level. Never split by pixel or patch — "
            "that leaks spatial information across train/val/test."
        ),
        "train": train,
        "val":   val,
        "test":  test,
    }
    (splits_dir / "splits.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: event-level train / validation / test split."
    )
    parser.add_argument("--train",      type=float, default=0.70,
                        help="Fraction for training set.   (default: 0.70)")
    parser.add_argument("--val",        type=float, default=0.15,
                        help="Fraction for validation set. (default: 0.15)")
    parser.add_argument("--test",       type=float, default=0.15,
                        help="Fraction for test set.       (default: 0.15)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for reproducibility. (default: 42)")
    parser.add_argument("--manifest",   type=Path,  default=DEFAULT_MANIFEST,
                        help=f"Tensor manifest JSON. (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--report-csv", type=Path,  default=DEFAULT_REPORT_CSV,
                        help=f"Compression report CSV fallback. (default: {DEFAULT_REPORT_CSV})")
    parser.add_argument("--splits-dir", type=Path,  default=DEFAULT_SPLITS_DIR,
                        help=f"Output directory. (default: {DEFAULT_SPLITS_DIR})")
    args = parser.parse_args()

    if args.manifest.exists():
        fire_ids = load_from_manifest(args.manifest)
        source = str(args.manifest)
    elif args.report_csv.exists():
        fire_ids = load_from_report(args.report_csv)
        source = str(args.report_csv)
        print(f"Note: manifest not found, falling back to {args.report_csv}")
    else:
        print(
            f"No input found.\n"
            f"  Expected: {args.manifest}\n"
            f"  Fallback: {args.report_csv}\n"
            "Run build_tensors.py first, or generate_mock_data.py for testing.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not fire_ids:
        print("No valid fire IDs found in input.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(fire_ids)} fire IDs from {source}")

    train, val, test = event_split(
        fire_ids, args.train, args.val, args.test, args.seed
    )

    write_splits(train, val, test, args.splits_dir, args.train, args.val, args.test, args.seed)

    n = len(fire_ids)
    print(f"\nSplit {n} fire events  (seed={args.seed}):")
    print(f"  train : {len(train):3d} fires  ({len(train)/n:.0%})")
    print(f"  val   : {len(val):3d} fires  ({len(val)/n:.0%})")
    print(f"  test  : {len(test):3d} fires  ({len(test)/n:.0%})")
    print(f"\nOutput: {args.splits_dir}/")
    print("  splits.json, train.txt, val.txt, test.txt")


if __name__ == "__main__":
    main()
