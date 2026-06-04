"""
align_fire_mask_update.py
=========================
Align stacked daily fire-mask GeoTIFFs to the existing Landsat severity
30 m reference grid and write into the **incremental update** folder:

    data/fire_mask_update/aligned_compressed/<fire_id>/labels/
        fire_mask_stacked_30m_aligned.tif

Does **not** modify the main ``data/aligned_compressed/`` dataset.

Reference grid (first found):
    data/aligned_compressed/<fire_id>/labels/severity_30m_aligned.tif
    data/raw/satellite_labels/<fire_id>/<fire_id>_Severity.tif

Usage (from project root):

    python scripts/task1/phase_03_alignment/align_fire_mask_update.py --fire-id 20778153
    python scripts/task1/phase_03_alignment/align_fire_mask_update.py --limit 5
    python scripts/task1/phase_03_alignment/align_fire_mask_update.py --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


DEFAULT_RAW_DIR = Path("data/raw/fire_mask_stacked")
DEFAULT_ALIGNED_COMPRESSED = Path("data/aligned_compressed")
DEFAULT_LABELS_DIR = Path("data/raw/satellite_labels")
DEFAULT_OUTPUT_ROOT = Path("data/fire_mask_update/aligned_compressed")
SUMMARY_CSV = Path("data/fire_mask_update/fire_mask_update_summary.csv")

OUTPUT_NAME = "fire_mask_stacked_30m_aligned.tif"
RAW_NAME_SUFFIX = "__fire_mask__stacked.tif"


def list_fire_ids(
    raw_dir: Path,
    fire_id: str | None,
    limit: int | None,
) -> list[str]:
    if fire_id:
        raw_path = raw_dir / fire_id / f"{fire_id}__fire_mask__stacked.tif"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw fire mask not found: {raw_path}")
        return [fire_id]

    ids = sorted(
        p.name for p in raw_dir.iterdir()
        if p.is_dir() and (p / f"{p.name}__fire_mask__stacked.tif").is_file()
    )
    if limit is not None:
        ids = ids[:limit]
    return ids


def raw_path_for(fire_id: str, raw_dir: Path) -> Path:
    return raw_dir / fire_id / f"{fire_id}__fire_mask__stacked.tif"


def output_path_for(fire_id: str, output_root: Path) -> Path:
    return output_root / fire_id / "labels" / OUTPUT_NAME


def find_reference_severity(
    fire_id: str,
    aligned_compressed: Path,
    labels_dir: Path,
) -> Path | None:
    aligned = aligned_compressed / fire_id / "labels" / "severity_30m_aligned.tif"
    if aligned.is_file():
        return aligned
    raw = labels_dir / fire_id / f"{fire_id}_Severity.tif"
    if raw.is_file():
        return raw
    return None


def reference_profile(severity_path: Path) -> dict:
    with rasterio.open(severity_path) as src:
        profile = src.profile.copy()
        profile.update(
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
            predictor=2,
            dtype="uint8",
        )
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "profile": profile,
        }


def align_fire_mask(
    src_path: Path,
    dst_path: Path,
    ref: dict,
) -> None:
    """Reproject all bands to reference grid (nearest), uint8 output."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        dst_count = src.count
        dst_data = np.zeros(
            (dst_count, ref["height"], ref["width"]),
            dtype=np.uint8,
        )

        for band_idx in range(1, dst_count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst_data[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref["transform"],
                dst_crs=ref["crs"],
                resampling=Resampling.nearest,
            )

        profile = ref["profile"].copy()
        profile.update(count=dst_count, dtype="uint8")

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(dst_data)
            if src.descriptions:
                dst.descriptions = src.descriptions


def verify_output(dst_path: Path, ref: dict, band_count: int) -> tuple[bool, str]:
    try:
        with rasterio.open(dst_path) as dst:
            ok = (
                dst.width == ref["width"]
                and dst.height == ref["height"]
                and dst.count == band_count
                and dst.crs == ref["crs"]
                and dst.transform == ref["transform"]
                and dst.dtypes[0] == "uint8"
            )
            if not ok:
                return False, "metadata mismatch after alignment"
        return True, "OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def process_one(
    fire_id: str,
    raw_dir: Path,
    aligned_compressed: Path,
    labels_dir: Path,
    output_root: Path,
    skip_existing: bool,
) -> dict:
    src = raw_path_for(fire_id, raw_dir)
    dst = output_path_for(fire_id, output_root)

    row = {
        "fire_id": fire_id,
        "raw_file": str(src),
        "reference_file": "",
        "output_file": str(dst),
        "band_count": "",
        "aligned_success": False,
        "skipped": False,
        "error": "",
    }

    print(f"\n{'─' * 60}")
    print(f"Fire ID: {fire_id}")

    if skip_existing and dst.is_file():
        print(f"  [SKIP] Output exists: {dst}")
        row["skipped"] = True
        row["aligned_success"] = True
        return row

    if not src.is_file():
        row["error"] = f"Missing raw file: {src}"
        print(f"  [ERROR] {row['error']}")
        return row

    ref_path = find_reference_severity(fire_id, aligned_compressed, labels_dir)
    if ref_path is None:
        row["error"] = "No severity reference grid found"
        print(f"  [ERROR] {row['error']}")
        return row

    row["reference_file"] = str(ref_path)
    print(f"  Raw       : {src}")
    print(f"  Reference : {ref_path}")
    print(f"  Output    : {dst}")

    try:
        with rasterio.open(src) as raw:
            row["band_count"] = raw.count

        ref = reference_profile(ref_path)
        align_fire_mask(src, dst, ref)
        ok, msg = verify_output(dst, ref, int(row["band_count"]))
        row["aligned_success"] = ok
        if not ok:
            row["error"] = msg
        print(f"  Bands     : {row['band_count']}")
        print(f"  Status    : {'OK' if ok else msg}")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [ERROR] {row['error']}")
        traceback.print_exc()

    return row


def write_summary(rows: list[dict], summary_path: Path) -> None:
    fieldnames = [
        "fire_id",
        "raw_file",
        "reference_file",
        "output_file",
        "band_count",
        "aligned_success",
        "skipped",
        "error",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align stacked fire masks into data/fire_mask_update/."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--aligned-compressed-dir",
        type=Path,
        default=DEFAULT_ALIGNED_COMPRESSED,
    )
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fire-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    try:
        fire_ids = list_fire_ids(args.raw_dir, args.fire_id, args.limit)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not fire_ids:
        print(f"[ERROR] No organized fire-mask stacks under {args.raw_dir}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Fire-mask update alignment")
    print(f"{'=' * 60}")
    print(f"Raw dir    : {args.raw_dir.resolve()}")
    print(f"Output dir : {args.output_dir.resolve()}")
    print(f"Fires      : {len(fire_ids)}")
    print(f"Skip exist : {args.skip_existing}")
    print(f"{'=' * 60}")

    rows: list[dict] = []
    ok_count = 0
    fail_count = 0

    for fid in fire_ids:
        row = process_one(
            fid,
            args.raw_dir,
            args.aligned_compressed_dir,
            args.labels_dir,
            args.output_dir,
            args.skip_existing,
        )
        rows.append(row)
        if row["aligned_success"]:
            ok_count += 1
        else:
            fail_count += 1

    write_summary(rows, SUMMARY_CSV)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"  Processed : {len(fire_ids)}")
    print(f"  Success   : {ok_count}")
    print(f"  Failed    : {fail_count}")
    print(f"  Report    : {SUMMARY_CSV.resolve()}")
    print(f"{'=' * 60}\n")

    sys.exit(1 if fail_count else 0)


if __name__ == "__main__":
    main()
