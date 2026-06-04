"""
compress_aligned_geotiffs.py
============================
Storage-safe compression for Phase 3 aligned GeoTIFFs.

Compress each fire folder from ``data/aligned/<fire_id>/`` into
``data/aligned_compressed/<fire_id>/``, preserving folder structure.
Optionally delete the original fire folder after **all** GeoTIFFs in
that fire are compressed and verified — freeing disk space progressively.

Usage (from project root):

    # Test one fire, keep original
    python scripts/task1/phase_03_alignment/compress_aligned_geotiffs.py --fire-id 20562846

    # Test one fire, delete original only if everything succeeds
    python scripts/task1/phase_03_alignment/compress_aligned_geotiffs.py \\
        --fire-id 20562846 --delete-original-after-success

    # First 5 fires, delete originals after each success
    python scripts/task1/phase_03_alignment/compress_aligned_geotiffs.py \\
        --limit 5 --delete-original-after-success

    # All remaining aligned fires
    python scripts/task1/phase_03_alignment/compress_aligned_geotiffs.py --delete-original-after-success
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio


DEFAULT_INPUT_DIR = Path("data/aligned")
DEFAULT_OUTPUT_DIR = Path("data/aligned_compressed")
REPORT_NAME = "compression_report.csv"

REQUIRED_LAYER_PATHS = [
    Path("labels/burned_area_30m_aligned.tif"),
    Path("labels/severity_30m_aligned.tif"),
    Path("terrain/terrain_30m_aligned.tif"),
    Path("weather/weather_stacked_30m_aligned.tif"),
]

METADATA_NAME = "metadata.json"
VALID_SUFFIXES = {".tif", ".tiff"}


def bytes_to_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def percent_reduction(original: int, compressed: int) -> float:
    if original <= 0:
        return 0.0
    return round((1.0 - compressed / original) * 100.0, 2)


def predictor_for_dtype(dtype: str) -> int:
    """predictor=2 for integer, predictor=3 for floating point."""
    if np.issubdtype(np.dtype(dtype), np.floating):
        return 3
    return 2


def format_size(num_bytes: int) -> str:
    mb = bytes_to_mb(num_bytes)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def list_fire_ids(
    input_dir: Path,
    fire_id: str | None,
    limit: int | None,
) -> list[str]:
    """Return sorted fire folder names under ``input_dir``."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if fire_id is not None:
        fire_dir = input_dir / fire_id
        if not fire_dir.is_dir():
            raise FileNotFoundError(f"Fire folder not found: {fire_dir}")
        return [fire_id]

    ids = sorted(
        p.name for p in input_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if limit is not None:
        ids = ids[:limit]
    return ids


def collect_geotiff_pairs(src_fire: Path, dst_fire: Path) -> list[tuple[Path, Path]]:
    """Map required source GeoTIFFs to matching destination paths."""
    pairs: list[tuple[Path, Path]] = []
    for rel in REQUIRED_LAYER_PATHS:
        src = src_fire / rel
        if not src.is_file():
            raise FileNotFoundError(f"Missing required file: {src}")
        pairs.append((src, dst_fire / rel))
    return pairs


def warn_if_output_incomplete(dst_fire: Path) -> None:
    """Warn when compressed output exists but is missing expected files."""
    if not dst_fire.exists():
        return
    missing = [
        str(rel) for rel in REQUIRED_LAYER_PATHS
        if not (dst_fire / rel).is_file()
    ]
    if missing:
        print(
            f"[WARN] Output folder exists but is incomplete: {dst_fire}\n"
            f"       Missing: {', '.join(missing)}"
        )


def copy_tags(src: rasterio.io.DatasetReader, dst: rasterio.io.DatasetWriter) -> None:
    tags = src.tags()
    if tags:
        dst.update_tags(**tags)


def verify_geotiff(compressed_path: Path, reference_path: Path) -> tuple[bool, str]:
    """Verify compressed GeoTIFF metadata matches the original."""
    try:
        with rasterio.open(reference_path) as ref, rasterio.open(compressed_path) as comp:
            ok = (
                comp.width == ref.width
                and comp.height == ref.height
                and comp.count == ref.count
                and comp.crs == ref.crs
                and comp.transform == ref.transform
                and comp.dtypes == ref.dtypes
            )
            if not ok:
                return False, "metadata mismatch (CRS/transform/shape/count/dtype)"
        return True, "OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def compress_geotiff(
    src_path: Path,
    dst_path: Path,
    overwrite: bool,
) -> dict:
    """
    Compress one GeoTIFF band-by-band, verify, and return a report row fragment.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    original_size = src_path.stat().st_size

    if dst_path.exists() and not overwrite:
        verified, verify_msg = verify_geotiff(dst_path, src_path)
        compressed_size = dst_path.stat().st_size
        return {
            "skipped": True,
            "verified": verified,
            "error": "" if verified else verify_msg,
            "original_size": original_size,
            "compressed_size": compressed_size,
        }

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
            predictor=predictor_for_dtype(src.dtypes[0]),
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            copy_tags(src, dst)
            if src.nodata is not None:
                dst.nodata = src.nodata
            if src.descriptions:
                dst.descriptions = src.descriptions

            for band_idx in range(1, src.count + 1):
                dst.write(src.read(band_idx), band_idx)
                if src.descriptions and src.descriptions[band_idx - 1]:
                    dst.set_band_description(band_idx, src.descriptions[band_idx - 1])
                band_tags = src.tags(band_idx)
                if band_tags:
                    dst.update_tags(band_idx, **band_tags)

    compressed_size = dst_path.stat().st_size
    verified, verify_msg = verify_geotiff(dst_path, src_path)
    return {
        "skipped": False,
        "verified": verified,
        "error": "" if verified else verify_msg,
        "original_size": original_size,
        "compressed_size": compressed_size,
    }


def copy_metadata(src_fire: Path, dst_fire: Path) -> None:
    """Copy metadata.json when present."""
    src_meta = src_fire / METADATA_NAME
    if src_meta.is_file():
        dst_fire.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_meta, dst_fire / METADATA_NAME)


def output_folder_complete(dst_fire: Path) -> bool:
    """True when every required GeoTIFF exists in the compressed fire folder."""
    return all((dst_fire / rel).is_file() for rel in REQUIRED_LAYER_PATHS)


def delete_original_fire_folder(src_fire: Path) -> None:
    """Remove the original aligned fire folder after successful compression."""
    shutil.rmtree(src_fire)


def process_one_fire(
    fire_id: str,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    delete_original: bool,
) -> list[dict]:
    """
    Compress one fire folder. Returns report rows (one per GeoTIFF).
    """
    src_fire = input_dir / fire_id
    dst_fire = output_dir / fire_id

    print(f"\n{'=' * 70}")
    print(f"Fire ID: {fire_id}")
    print(f"  Source : {src_fire}")
    print(f"  Output : {dst_fire}")
    print(f"{'=' * 70}")

    warn_if_output_incomplete(dst_fire)

    rows: list[dict] = []
    all_verified = True
    any_error = False

    try:
        pairs = collect_geotiff_pairs(src_fire, dst_fire)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return [{
            "fire_id": fire_id,
            "source_file": str(src_fire),
            "output_file": str(dst_fire),
            "original_size_mb": 0,
            "compressed_size_mb": 0,
            "reduction_percent": 0,
            "verified": False,
            "fire_complete": False,
            "original_deleted": False,
            "error": str(exc),
        }]

    for src, dst in pairs:
        rel_name = src.relative_to(src_fire)
        print(f"\n  File: {rel_name}")
        row = {
            "fire_id": fire_id,
            "source_file": str(src),
            "output_file": str(dst),
            "original_size_mb": 0,
            "compressed_size_mb": 0,
            "reduction_percent": 0,
            "verified": False,
            "fire_complete": False,
            "original_deleted": False,
            "error": "",
        }

        try:
            result = compress_geotiff(src, dst, overwrite=overwrite)
            row["original_size_mb"] = bytes_to_mb(result["original_size"])
            row["compressed_size_mb"] = bytes_to_mb(result["compressed_size"])
            row["reduction_percent"] = percent_reduction(
                result["original_size"], result["compressed_size"]
            )
            row["verified"] = result["verified"]
            row["error"] = result["error"]

            action = "skipped (exists)" if result["skipped"] else "compressed"
            print(f"    {action}")
            print(f"    original  : {format_size(result['original_size'])}")
            print(f"    compressed: {format_size(result['compressed_size'])}")
            print(f"    saved     : {row['reduction_percent']:.1f}%")
            print(f"    verified  : {'OK' if row['verified'] else 'FAIL'}")

            if not row["verified"]:
                all_verified = False
                any_error = True
                print(f"    error     : {row['error']}")

        except Exception as exc:
            any_error = True
            all_verified = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    [ERROR] {row['error']}")

        rows.append(row)

    fire_complete = (
        not any_error
        and all_verified
        and output_folder_complete(dst_fire)
    )

    original_deleted = False

    if fire_complete:
        copy_metadata(src_fire, dst_fire)
        print(f"\n  metadata.json copied.")
        print(f"  Fire complete: YES")

        if delete_original:
            if not output_folder_complete(dst_fire):
                print("  [SAFETY] Will NOT delete original — compressed folder incomplete.")
            else:
                delete_original_fire_folder(src_fire)
                original_deleted = True
                print(f"  Original deleted: YES ({src_fire})")
        else:
            print(f"  Original deleted: NO (--delete-original-after-success not set)")
    else:
        print(f"\n  Fire complete: NO — original folder kept.")
        if not output_folder_complete(dst_fire):
            print("  [SAFETY] Compressed output folder is missing required files.")

    for row in rows:
        row["fire_complete"] = fire_complete
        row["original_deleted"] = original_deleted

    return rows


def write_report(report_path: Path, rows: list[dict]) -> None:
    """Write ``compression_report.csv``."""
    fieldnames = [
        "fire_id",
        "source_file",
        "output_file",
        "original_size_mb",
        "compressed_size_mb",
        "reduction_percent",
        "verified",
        "fire_complete",
        "original_deleted",
        "error",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    input_dir: Path,
    output_dir: Path,
    fire_id: str | None,
    limit: int | None,
    overwrite: bool,
    delete_original: bool,
) -> int:
    """Process all selected fire folders."""
    try:
        fire_ids = list_fire_ids(input_dir, fire_id, limit)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not fire_ids:
        print(f"[ERROR] No fire folders found in {input_dir}")
        return 1

    print(f"\n{'#' * 70}")
    print("Storage-safe aligned GeoTIFF compression")
    print(f"{'#' * 70}")
    print(f"Input dir              : {input_dir.resolve()}")
    print(f"Output dir             : {output_dir.resolve()}")
    print(f"Fires to process       : {len(fire_ids)}")
    print(f"Overwrite              : {overwrite}")
    print(f"Delete original on OK  : {delete_original}")
    print(f"{'#' * 70}")

    all_rows: list[dict] = []
    fires_complete = 0
    fires_failed = 0
    fires_deleted = 0

    for fid in fire_ids:
        rows = process_one_fire(
            fid, input_dir, output_dir, overwrite, delete_original
        )
        all_rows.extend(rows)
        if rows and rows[0].get("fire_complete"):
            fires_complete += 1
            if rows[0].get("original_deleted"):
                fires_deleted += 1
        else:
            fires_failed += 1

    report_path = output_dir / REPORT_NAME
    write_report(report_path, all_rows)

    print(f"\n{'#' * 70}")
    print("Summary")
    print(f"{'#' * 70}")
    print(f"Fires processed        : {len(fire_ids)}")
    print(f"Fires complete         : {fires_complete}")
    print(f"Fires failed/incomplete: {fires_failed}")
    print(f"Originals deleted      : {fires_deleted}")
    print(f"Report                 : {report_path.resolve()}")
    print(f"{'#' * 70}\n")

    return 1 if fires_failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compress aligned GeoTIFFs fire-by-fire into data/aligned_compressed/. "
            "Optionally delete each original fire folder after verified success."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Aligned dataset root (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Compressed output root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--fire-id",
        type=str,
        default=None,
        help="Process only this fire ID.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N fire folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite compressed GeoTIFFs that already exist.",
    )
    parser.add_argument(
        "--delete-original-after-success",
        action="store_true",
        help=(
            "After all GeoTIFFs in a fire are compressed and verified, "
            "delete data/aligned/<fire_id>/ to free disk space."
        ),
    )
    args = parser.parse_args()
    sys.exit(run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fire_id=args.fire_id,
        limit=args.limit,
        overwrite=args.overwrite,
        delete_original=args.delete_original_after_success,
    ))


if __name__ == "__main__":
    main()
