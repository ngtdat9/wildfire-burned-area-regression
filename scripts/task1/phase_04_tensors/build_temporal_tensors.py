"""
build_temporal_tensors.py
=========================
Phase 4 — Build spatio-temporal fire-spread training samples.

For each eligible fire event, create one compressed NPZ per temporal
sample:

    X = fire mask + weather for [t-3, t-2, t-1, t]
    Y = fire mask at t+1

Static context (same for all timesteps in a fire):
    terrain (3 bands) + severity (1 band)

Samples from the same fire always stay in the same split (train/val/test).

Usage (from project root):

    python scripts/task1/phase_04_tensors/create_event_splits.py
    python scripts/task1/phase_04_tensors/build_temporal_tensors.py --split train
    python scripts/task1/phase_04_tensors/build_temporal_tensors.py \\
        --fire-id 20562846 --split all
    python scripts/task1/phase_04_tensors/build_temporal_tensors.py \\
        --split train --limit 3
    python scripts/task1/phase_04_tensors/build_temporal_tensors.py \\
        --split train --skip-existing
    python scripts/task1/phase_04_tensors/build_temporal_tensors.py \\
        --fire-id 20562846 --force-rebuild-fire
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

DEFAULT_INPUT_DIR = Path("data/aligned_compressed")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_DIR = Path("data/tensors")

REQUIRED_FILES = {
    "fire_mask": Path("labels/fire_mask_stacked_30m_aligned.tif"),
    "severity": Path("labels/severity_30m_aligned.tif"),
    "terrain": Path("terrain/terrain_30m_aligned.tif"),
    "weather": Path("weather/weather_stacked_30m_aligned.tif"),
}

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "surface_pressure",
    "total_precipitation",
]

DATE_RE = re.compile(r"(\d{8})$")
SPLIT_FILES = {
    "train": "train_events.txt",
    "val": "val_events.txt",
    "test": "test_events.txt",
}

INDEX_FIELDNAMES = [
    "sample_id",
    "split",
    "fire_id",
    "input_start_day",
    "input_end_day",
    "target_day",
    "target_date",
    "x_path",
    "x_shape",
    "y_shape",
    "status",
]

NPZ_REQUIRED_KEYS = ("X", "terrain", "severity", "Y")


def parse_date(description: str | None) -> str | None:
    if not description:
        return None
    match = DATE_RE.search(description)
    return match.group(1) if match else None


def fire_mask_band_map(path: Path) -> dict[str, int]:
    """Map YYYYMMDD date -> 1-based band index."""
    with rasterio.open(path) as src:
        mapping: dict[str, int] = {}
        for idx, desc in enumerate(src.descriptions or [], start=1):
            date = parse_date(desc)
            if date:
                mapping[date] = idx
        return mapping


def weather_band_map(path: Path) -> dict[str, list[int]]:
    """Map YYYYMMDD date -> list of 1-based weather band indices (6 vars)."""
    with rasterio.open(path) as src:
        by_date: dict[str, list[tuple[int, str]]] = {}
        for idx, desc in enumerate(src.descriptions or [], start=1):
            date = parse_date(desc)
            if not date:
                continue
            by_date.setdefault(date, []).append((idx, desc or ""))

        mapping: dict[str, list[int]] = {}
        for date, items in by_date.items():
            items.sort(key=lambda x: x[0])
            mapping[date] = [idx for idx, _ in items]
        return mapping


def common_sorted_dates(fire_map: dict[str, int], weather_map: dict[str, list[int]]) -> list[str]:
    common = sorted(set(fire_map) & set(weather_map))
    return common


def source_fire_dir(input_dir: Path, fire_id: str) -> Path:
    return input_dir / fire_id


def source_is_available(input_dir: Path, fire_id: str) -> bool:
    fire_dir = source_fire_dir(input_dir, fire_id)
    if not fire_dir.is_dir():
        return False
    return all((fire_dir / rel).is_file() for rel in REQUIRED_FILES.values())


def expected_target_dates(
    input_dir: Path,
    fire_id: str,
    input_window: int,
) -> list[str]:
    """Return YYYYMMDD target dates that should have NPZ samples."""
    fire_dir = source_fire_dir(input_dir, fire_id)
    paths = {key: fire_dir / rel for key, rel in REQUIRED_FILES.items()}
    fire_map = fire_mask_band_map(paths["fire_mask"])
    weather_map = weather_band_map(paths["weather"])
    dates = common_sorted_dates(fire_map, weather_map)

    min_needed = input_window + 1
    if len(dates) < min_needed:
        return []

    return [dates[t_idx + 1] for t_idx in range(input_window - 1, len(dates) - 1)]


def verify_npz(path: Path) -> bool:
    """Return True when NPZ exists and contains required arrays."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return all(key in data.files for key in NPZ_REQUIRED_KEYS)
    except Exception:
        return False


def load_existing_index(output_dir: Path) -> list[dict]:
    path = output_dir / "sample_index.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_for_fire(index_rows: list[dict], fire_id: str) -> list[dict]:
    return [row for row in index_rows if row.get("fire_id") == fire_id]


def assess_fire_progress(
    fire_id: str,
    split: str,
    input_dir: Path,
    output_dir: Path,
    input_window: int,
    existing_index: list[dict],
) -> tuple[str, list[dict], list[Path]]:
    """
    Decide whether to skip, build, or rebuild a fire.

    Returns:
        action: 'skip', 'build', or 'rebuild'
        existing_rows: index rows already stored for this fire
        expected_paths: NPZ paths that should exist for a complete fire
    """
    existing_rows = rows_for_fire(existing_index, fire_id)
    source_ok = source_is_available(input_dir, fire_id)

    if source_ok:
        target_dates = expected_target_dates(input_dir, fire_id, input_window)
        expected_paths = [
            sample_output_path(output_dir, split, fire_id, d) for d in target_dates
        ]
    elif existing_rows:
        expected_paths = [output_dir / row["x_path"] for row in existing_rows]
    else:
        return "build", [], []

    if not expected_paths:
        return "build", existing_rows, []

    missing_paths = [p for p in expected_paths if not p.is_file()]
    invalid_paths = [p for p in expected_paths if p.is_file() and not verify_npz(p)]
    indexed_valid = all(
        verify_npz(output_dir / row["x_path"]) for row in existing_rows
    ) if existing_rows else False

    complete = (
        len(missing_paths) == 0
        and len(invalid_paths) == 0
        and len(existing_rows) >= len(expected_paths)
        and indexed_valid
    )

    if complete:
        return "skip", existing_rows, expected_paths
    if existing_rows or missing_paths or invalid_paths:
        return "rebuild", existing_rows, expected_paths
    return "build", existing_rows, expected_paths


def delete_source_fire_folder(input_dir: Path, fire_id: str) -> bool:
    fire_dir = source_fire_dir(input_dir, fire_id)
    if fire_dir.is_dir():
        shutil.rmtree(fire_dir)
        return True
    return False


def merge_sample_index(
    existing_index: list[dict],
    new_rows: list[dict],
    rebuilt_fire_ids: set[str],
) -> list[dict]:
    """Keep untouched fires, replace rows for rebuilt/processed fires."""
    kept = [row for row in existing_index if row.get("fire_id") not in rebuilt_fire_ids]
    kept_ids = {row["sample_id"] for row in kept}
    for row in new_rows:
        if row["sample_id"] not in kept_ids:
            kept.append(row)
    return kept


def load_split_map(splits_dir: Path) -> dict[str, str]:
    """Return fire_id -> split name."""
    mapping: dict[str, str] = {}
    for split, filename in SPLIT_FILES.items():
        path = splits_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")
        for line in path.read_text().splitlines():
            fid = line.strip()
            if fid:
                mapping[fid] = split
    return mapping


def list_target_fires(
    split_arg: str,
    split_map: dict[str, str],
    fire_id: str | None,
    limit: int | None,
) -> list[tuple[str, str]]:
    """Return list of (fire_id, split) to process."""
    if fire_id:
        if fire_id not in split_map:
            raise ValueError(
                f"Fire {fire_id} not found in split files under {DEFAULT_SPLITS_DIR}."
            )
        return [(fire_id, split_map[fire_id])]

    pairs = [(fid, sp) for fid, sp in sorted(split_map.items())]
    if split_arg != "all":
        pairs = [(fid, sp) for fid, sp in pairs if sp == split_arg]
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def read_band(path: Path, band_idx: int) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(band_idx)


def read_static_terrain_severity(
    terrain_path: Path,
    severity_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(terrain_path) as terr, rasterio.open(severity_path) as sev:
        terrain = terr.read().astype(np.float32)  # (3, H, W)
        severity = sev.read(1).astype(np.float32)  # (H, W)
    return terrain, severity


def build_sample_arrays(
    fire_mask_path: Path,
    weather_path: Path,
    input_dates: list[str],
    target_date: str,
    fire_map: dict[str, int],
    weather_map: dict[str, list[int]],
    terrain: np.ndarray,
    severity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X dynamic, terrain, severity, Y.

    X dynamic shape: (T, 7, H, W) = 1 fire mask + 6 weather vars per day.
    Y shape: (1, H, W)
    """
    timesteps: list[np.ndarray] = []

    for date in input_dates:
        fm = read_band(fire_mask_path, fire_map[date]).astype(np.float32)
        wx_bands = weather_map[date]
        if len(wx_bands) != len(WEATHER_VARS):
            raise ValueError(
                f"Expected {len(WEATHER_VARS)} weather bands on {date}, "
                f"found {len(wx_bands)}."
            )
        wx = np.stack(
            [read_band(weather_path, b).astype(np.float32) for b in wx_bands],
            axis=0,
        )
        step = np.concatenate([fm[np.newaxis, ...], wx], axis=0)  # (7, H, W)
        timesteps.append(step)

    x_dynamic = np.stack(timesteps, axis=0)  # (T, 7, H, W)
    y = read_band(fire_mask_path, fire_map[target_date]).astype(np.uint8)
    y = y[np.newaxis, ...]  # (1, H, W)

    severity_band = severity[np.newaxis, ...]  # (1, H, W)
    return x_dynamic, terrain, severity_band, y


def sample_output_path(
    output_dir: Path,
    split: str,
    fire_id: str,
    target_date: str,
) -> Path:
    return output_dir / split / fire_id / f"{fire_id}_{target_date}.npz"


def process_fire(
    fire_id: str,
    split: str,
    input_dir: Path,
    output_dir: Path,
    input_window: int,
    overwrite: bool,
) -> tuple[list[dict], dict]:
    fire_dir = source_fire_dir(input_dir, fire_id)
    paths = {key: fire_dir / rel for key, rel in REQUIRED_FILES.items()}

    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key} for {fire_id}: {path}")

    fire_map = fire_mask_band_map(paths["fire_mask"])
    weather_map = weather_band_map(paths["weather"])
    dates = common_sorted_dates(fire_map, weather_map)

    min_needed = input_window + 1  # window ending at t plus target t+1
    if len(dates) < min_needed:
        return [], {
            "fire_id": fire_id,
            "split": split,
            "samples_created": 0,
            "skipped_reason": f"only {len(dates)} common days, need {min_needed}",
        }

    terrain, severity = read_static_terrain_severity(
        paths["terrain"], paths["severity"]
    )

    rows: list[dict] = []
    created = 0

    for t_idx in range(input_window - 1, len(dates) - 1):
        input_dates = dates[t_idx - input_window + 1 : t_idx + 1]
        target_date = dates[t_idx + 1]

        out_path = sample_output_path(output_dir, split, fire_id, target_date)
        sample_id = f"{fire_id}_{target_date}"

        if out_path.is_file() and not overwrite and verify_npz(out_path):
            with np.load(out_path) as data:
                x_shape = tuple(int(x) for x in data["X"].shape)
                y_shape = tuple(int(x) for x in data["Y"].shape)
            rows.append({
                "sample_id": sample_id,
                "split": split,
                "fire_id": fire_id,
                "input_start_day": t_idx - input_window + 2,  # 1-based
                "input_end_day": t_idx + 1,
                "target_day": t_idx + 2,
                "target_date": target_date,
                "x_path": str(out_path.relative_to(output_dir)),
                "x_shape": str(x_shape),
                "y_shape": str(y_shape),
                "status": "skipped_exists",
            })
            continue

        x_dynamic, terrain_arr, severity_arr, y = build_sample_arrays(
            paths["fire_mask"],
            paths["weather"],
            input_dates,
            target_date,
            fire_map,
            weather_map,
            terrain,
            severity,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            X=x_dynamic.astype(np.float32),
            terrain=terrain_arr.astype(np.float32),
            severity=severity_arr.astype(np.float32),
            Y=y.astype(np.uint8),
            fire_id=fire_id,
            split=split,
            input_dates=np.array(input_dates),
            target_date=target_date,
        )

        rows.append({
            "sample_id": sample_id,
            "split": split,
            "fire_id": fire_id,
            "input_start_day": t_idx - input_window + 2,
            "input_end_day": t_idx + 1,
            "target_day": t_idx + 2,
            "target_date": target_date,
            "x_path": str(out_path.relative_to(output_dir)),
            "x_shape": str(x_dynamic.shape),
            "y_shape": str(y.shape),
            "status": "created",
        })
        created += 1

    return rows, {
        "fire_id": fire_id,
        "split": split,
        "samples_created": created,
        "common_days": len(dates),
    }


def write_sample_index(rows: list[dict], output_dir: Path) -> None:
    path = output_dir / "sample_index.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    output_dir: Path,
    input_window: int,
    fire_stats: list[dict],
    sample_count: int,
) -> None:
    manifest = {
        "format": "temporal_fire_spread_v1",
        "input_window": input_window,
        "number_of_fires_processed": len(fire_stats),
        "number_of_samples": sample_count,
        "feature_description": {
            "X": (
                f"Dynamic input shape (T, 7, H, W) where T={input_window}. "
                "Channels per day: fire_mask (1) + weather vars (6): "
                + ", ".join(WEATHER_VARS)
            ),
            "terrain": "Static (3, H, W): elevation, slope, aspect",
            "severity": "Static (1, H, W): final dNBR severity class 0-4",
        },
        "target_description": {
            "Y": "Next-day cumulative fire mask (1, H, W), uint8 binary 0/1",
        },
        "temporal_rule": (
            "X uses fire mask + weather for [t-3, t-2, t-1, t]; "
            "Y is fire mask at t+1. Dates aligned by YYYYMMDD in band descriptions."
        ),
        "split_rule": (
            "Entire fire events assigned to train/val/test. "
            "All samples from one fire share the same split."
        ),
        "fires": fire_stats,
    }
    (output_dir / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build temporal fire-spread NPZ samples from aligned GeoTIFFs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which split to process (default: train).",
    )
    parser.add_argument("--input-window", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fire-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip fires whose NPZ outputs already look complete.",
    )
    parser.add_argument(
        "--force-rebuild-fire",
        action="store_true",
        help="Rebuild --fire-id even if valid tensors already exist.",
    )
    parser.add_argument(
        "--delete-source-after-success",
        action="store_true",
        help=(
            "Delete data/aligned_compressed/<fire_id>/ after a fire is built "
            "successfully in this run. Skipped fires are not rebuilt or deleted."
        ),
    )
    args = parser.parse_args()

    if args.force_rebuild_fire and not args.fire_id:
        print("[ERROR] --force-rebuild-fire requires --fire-id")
        sys.exit(1)

    if args.input_window < 2:
        print("[ERROR] --input-window must be >= 2")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        sys.exit(1)

    try:
        split_map = load_split_map(args.splits_dir)
        targets = list_target_fires(args.split, split_map, args.fire_id, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not targets:
        print(f"[ERROR] No fires to process for split={args.split}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Temporal tensor builder")
    print(f"{'=' * 60}")
    print(f"Input dir     : {args.input_dir.resolve()}")
    print(f"Output dir    : {args.output_dir.resolve()}")
    print(f"Split         : {args.split}")
    print(f"Input window  : {args.input_window}")
    print(f"Fires         : {len(targets)}")
    print(f"Overwrite     : {args.overwrite}")
    print(f"Skip existing : {args.skip_existing}")
    print(f"Force rebuild : {args.force_rebuild_fire}")
    print(f"Delete source : {args.delete_source_after_success}")
    print(f"{'=' * 60}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_index = load_existing_index(args.output_dir)

    new_rows: list[dict] = []
    fire_stats: list[dict] = []
    errors: list[str] = []
    skipped_fires: list[str] = []
    built_fires: list[str] = []
    rebuilt_fire_ids: set[str] = set()

    for fire_id, split in targets:
        force_this_fire = args.force_rebuild_fire and args.fire_id == fire_id
        fire_overwrite = args.overwrite or force_this_fire

        if args.skip_existing and not force_this_fire:
            action, existing_rows, expected_paths = assess_fire_progress(
                fire_id,
                split,
                args.input_dir,
                args.output_dir,
                args.input_window,
                existing_index,
            )
            if action == "skip":
                source_ok = source_is_available(args.input_dir, fire_id)
                suffix = (
                    f"({len(existing_rows)} samples)"
                    if source_ok
                    else f"({len(existing_rows)} samples; source folder missing)"
                )
                print(f"[SKIP] {fire_id} already has valid tensors {suffix}")
                skipped_fires.append(fire_id)
                fire_stats.append({
                    "fire_id": fire_id,
                    "split": split,
                    "samples_created": 0,
                    "status": "skipped_existing",
                    "expected_samples": len(expected_paths),
                })
                continue
            if action == "rebuild":
                print(
                    f"[REBUILD] {fire_id} output incomplete "
                    f"({len(existing_rows)}/{len(expected_paths)} indexed samples)"
                )
            else:
                print(f"[BUILD] {fire_id} generating tensors")
        else:
            if force_this_fire:
                print(f"[REBUILD] {fire_id} forced rebuild")
            else:
                print(f"[BUILD] {fire_id} generating tensors")

        rebuilt_fire_ids.add(fire_id)

        try:
            rows, stats = process_fire(
                fire_id,
                split,
                args.input_dir,
                args.output_dir,
                args.input_window,
                fire_overwrite,
            )
            new_rows.extend(rows)
            stats["status"] = "built"
            fire_stats.append(stats)
            built_fires.append(fire_id)
            print(
                f"    samples: {stats.get('samples_created', 0)} created, "
                f"{len(rows)} indexed"
            )

            if args.delete_source_after_success:
                expected = expected_target_dates(
                    args.input_dir, fire_id, args.input_window
                )
                built_paths = [
                    sample_output_path(args.output_dir, split, fire_id, d)
                    for d in expected
                ]
                if expected and all(verify_npz(p) for p in built_paths):
                    if delete_source_fire_folder(args.input_dir, fire_id):
                        print(f"    deleted source folder: {args.input_dir / fire_id}")
                else:
                    print(
                        f"    [WARN] source not deleted for {fire_id}: "
                        "output still incomplete"
                    )
        except FileNotFoundError as exc:
            indexed = rows_for_fire(existing_index, fire_id)
            if indexed and all(
                verify_npz(args.output_dir / row["x_path"]) for row in indexed
            ):
                print(
                    f"[SKIP] {fire_id} already has valid tensors "
                    f"({len(indexed)} samples; source folder missing)"
                )
                skipped_fires.append(fire_id)
                rebuilt_fire_ids.discard(fire_id)
                fire_stats.append({
                    "fire_id": fire_id,
                    "split": split,
                    "samples_created": 0,
                    "status": "skipped_existing",
                    "expected_samples": len(indexed),
                })
                continue
            msg = f"{fire_id}: {exc}"
            errors.append(msg)
            print(f"    [ERROR] {exc}")
        except Exception as exc:
            msg = f"{fire_id}: {exc}"
            errors.append(msg)
            print(f"    [ERROR] {exc}")

    merged_index = merge_sample_index(existing_index, new_rows, rebuilt_fire_ids)
    write_sample_index(merged_index, args.output_dir)
    write_manifest(
        args.output_dir,
        args.input_window,
        fire_stats,
        len(merged_index),
    )

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"  Fires targeted  : {len(targets)}")
    print(f"  Fires built     : {len(built_fires)}")
    print(f"  Fires skipped   : {len(skipped_fires)}")
    print(f"  Samples indexed : {len(merged_index)}")
    print(f"  Errors          : {len(errors)}")
    print(f"  sample_index    : {(args.output_dir / 'sample_index.csv').resolve()}")
    print(f"  manifest        : {(args.output_dir / 'tensor_manifest.json').resolve()}")
    if errors:
        print("\n  Failed fires:")
        for msg in errors[:10]:
            print(f"    * {msg}")
    print(f"{'=' * 60}\n")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
