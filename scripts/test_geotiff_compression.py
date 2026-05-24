"""
test_geotiff_compression.py
============================
Test GeoTIFF compression on one aligned fire event (or one file) before
uploading the full ~70 GB dataset to Google Drive.

Compares original vs compressed file size and verifies that rasterio can
still open each compressed GeoTIFF.

Usage (from project root):

    # One weather file (largest layer)
    python scripts/test_geotiff_compression.py \\
        --input data/aligned/<fire_id>/weather/weather_stacked_30m_aligned.tif

    # Full fire folder (all four aligned layers)
    python scripts/test_geotiff_compression.py \\
        --input data/aligned/<fire_id>/

    # Overwrite existing test outputs
    python scripts/test_geotiff_compression.py \\
        --input data/aligned/<fire_id>/ \\
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio


DEFAULT_COMPRESSED_ROOT = Path("data/aligned_compressed_test")
REPORT_NAME = "compression_test_report.csv"

FIRE_LAYER_PATHS = [
    Path("labels/burned_area_30m_aligned.tif"),
    Path("labels/severity_30m_aligned.tif"),
    Path("terrain/terrain_30m_aligned.tif"),
    Path("weather/weather_stacked_30m_aligned.tif"),
]

VALID_SUFFIXES = {".tif", ".tiff"}


def format_size(num_bytes: int) -> str:
    """Return human-readable size in MB or GB."""
    mb = num_bytes / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def percent_reduction(original: int, compressed: int) -> float:
    """Return percent size reduction (0 if original is zero)."""
    if original <= 0:
        return 0.0
    return (1.0 - compressed / original) * 100.0


def predictor_for_dtype(dtype: str) -> int:
    """
    Choose GDAL predictor for compression.

    - 2 = horizontal differencing (good for integer rasters)
    - 3 = floating-point predictor
    """
    if np.issubdtype(np.dtype(dtype), np.floating):
        return 3
    return 2


def default_output_path(input_path: Path) -> Path:
    """Build default output path for file or folder input."""
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_compressed{input_path.suffix}")

    fire_id = input_path.name
    return DEFAULT_COMPRESSED_ROOT / fire_id


def collect_inputs(input_path: Path) -> list[tuple[Path, Path]]:
    """
    Return list of (source_file, destination_file) pairs to compress.

    For a folder input, only known aligned layer paths are included.
    """
    input_path = input_path.resolve()

    if input_path.is_file():
        if input_path.suffix.lower() not in VALID_SUFFIXES:
            raise ValueError(f"Not a GeoTIFF file: {input_path}")
        return [(input_path, default_output_path(input_path))]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")

    fire_root = default_output_path(input_path)
    pairs: list[tuple[Path, Path]] = []
    for rel in FIRE_LAYER_PATHS:
        src = input_path / rel
        if src.is_file():
            pairs.append((src, fire_root / rel))
    return pairs


def copy_tags(src: rasterio.io.DatasetReader, dst: rasterio.io.DatasetWriter) -> None:
    """Copy dataset tags when present."""
    tags = src.tags()
    if tags:
        dst.update_tags(**tags)


def compress_geotiff(
    src_path: Path,
    dst_path: Path,
    overwrite: bool,
) -> dict:
    """
    Compress one GeoTIFF band-by-band and verify the result.

    Returns one row dict for the report CSV.
    """
    src_path = src_path.resolve()
    dst_path = dst_path.resolve()

    if dst_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists (use --overwrite): {dst_path}"
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    original_size = src_path.stat().st_size

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

            # Band-by-band copy (avoids loading all bands at once).
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx)
                dst.write(data, band_idx)
                if src.descriptions and src.descriptions[band_idx - 1]:
                    dst.set_band_description(band_idx, src.descriptions[band_idx - 1])

                band_tags = src.tags(band_idx)
                if band_tags:
                    dst.update_tags(band_idx, **band_tags)

    compressed_size = dst_path.stat().st_size
    verify_status = verify_geotiff(dst_path, src_path)

    return {
        "file_name": src_path.name,
        "source_path": str(src_path),
        "output_path": str(dst_path),
        "original_bytes": original_size,
        "compressed_bytes": compressed_size,
        "original_size": format_size(original_size),
        "compressed_size": format_size(compressed_size),
        "percent_reduction": round(percent_reduction(original_size, compressed_size), 2),
        "verification_status": verify_status,
    }


def verify_geotiff(compressed_path: Path, reference_path: Path) -> str:
    """
    Open compressed file and check basic metadata against the original.

    Returns 'OK' or an error message string.
    """
    try:
        with rasterio.open(reference_path) as ref, rasterio.open(compressed_path) as comp:
            checks = [
                comp.crs == ref.crs,
                comp.transform == ref.transform,
                comp.width == ref.width,
                comp.height == ref.height,
                comp.count == ref.count,
                comp.dtypes == ref.dtypes,
            ]
            if not all(checks):
                return "FAIL: metadata mismatch (CRS/transform/shape/count/dtype)"
        return "OK"
    except Exception as exc:
        return f"FAIL: {type(exc).__name__}: {exc}"


def report_output_path(
    input_path: Path,
    output_path: Path | None,
    pairs: list[tuple[Path, Path]],
) -> Path:
    """Decide where to save compression_test_report.csv."""
    if output_path is not None and output_path.suffix.lower() in VALID_SUFFIXES:
        return output_path.with_name(REPORT_NAME)

    if input_path.is_file():
        first_dst = pairs[0][1]
        return first_dst.parent / REPORT_NAME

    if output_path is not None and output_path.is_dir():
        return output_path / REPORT_NAME

    return default_output_path(input_path) / REPORT_NAME


def resolve_destination(
    src: Path,
    input_path: Path,
    output_path: Path | None,
) -> Path:
    """Map each source file to its destination path."""
    if output_path is None:
        if input_path.is_file():
            return default_output_path(input_path)
        return default_output_path(input_path) / src.relative_to(input_path.resolve())

    output_path = output_path.resolve()
    if output_path.suffix.lower() in VALID_SUFFIXES:
        return output_path

    if input_path.is_file():
        return output_path

    return output_path / src.relative_to(input_path.resolve())


def print_table(rows: list[dict]) -> None:
    """Print a simple comparison table to the terminal."""
    print(f"\n{'=' * 90}")
    print("GeoTIFF compression test results")
    print(f"{'=' * 90}")
    print(f"{'File':<40} {'Original':>12} {'Compressed':>12} {'Saved':>8} {'Verify':>8}")
    print(f"{'-' * 90}")
    for row in rows:
        print(
            f"{row['file_name']:<40} "
            f"{row['original_size']:>12} "
            f"{row['compressed_size']:>12} "
            f"{row['percent_reduction']:>7.1f}% "
            f"{row['verification_status']:>8}"
        )
    print(f"{'=' * 90}\n")


def write_report_csv(report_path: Path, rows: list[dict]) -> None:
    """Save compression_test_report.csv."""
    fieldnames = [
        "file_name",
        "source_path",
        "output_path",
        "original_bytes",
        "compressed_bytes",
        "original_size",
        "compressed_size",
        "percent_reduction",
        "verification_status",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_compression(
    input_path: Path,
    output_path: Path | None,
    overwrite: bool,
) -> int:
    """Main compression loop."""
    input_path = input_path.resolve()

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return 1

    raw_pairs = collect_inputs(input_path)
    if not raw_pairs:
        print(f"[ERROR] No aligned GeoTIFFs found under: {input_path}")
        return 1

    pairs: list[tuple[Path, Path]] = []
    for src, _ in raw_pairs:
        dst = resolve_destination(src, input_path, output_path)
        pairs.append((src, dst))

    print(f"\nInput : {input_path}")
    print(f"Files : {len(pairs)}")
    print(f"Overwrite: {overwrite}\n")

    rows: list[dict] = []
    failed = 0

    for src, dst in pairs:
        print(f"Compressing: {src.name}")
        print(f"  -> {dst}")
        try:
            row = compress_geotiff(src, dst, overwrite=overwrite)
            rows.append(row)
            print(
                f"  {row['original_size']} -> {row['compressed_size']} "
                f"({row['percent_reduction']:.1f}% smaller) [{row['verification_status']}]"
            )
        except Exception as exc:
            failed += 1
            print(f"  [ERROR] {type(exc).__name__}: {exc}")

    if not rows:
        print("[ERROR] No files were compressed.")
        return 1

    print_table(rows)

    report_path = report_output_path(input_path, output_path, pairs)
    write_report_csv(report_path, rows)
    print(f"Report saved: {report_path.resolve()}")

    if failed:
        print(f"Completed with {failed} error(s).")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test GeoTIFF deflate compression on one aligned fire event "
            "or a single GeoTIFF before uploading to Google Drive."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="GeoTIFF file or aligned fire folder (data/aligned/<fire_id>/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output file or folder. "
            "Default: *_compressed.tif beside file input, or "
            f"{DEFAULT_COMPRESSED_ROOT}/<fire_id>/ for folder input."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite compressed outputs if they already exist.",
    )
    args = parser.parse_args()
    sys.exit(run_compression(args.input, args.output, args.overwrite))


if __name__ == "__main__":
    main()
