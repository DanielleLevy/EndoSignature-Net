"""Prepare authoritative metadata and filtered matrices for GSE213216.

Example:
    python prepare_gse213216.py

The script extracts only one filtered Cell Ranger H5 file per available
single-cell sample. It deliberately excludes raw matrices, molecule-level files
and Visium spatial-transcriptomics objects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.validation_data import parse_gse213216_miniml, prepare_gse213216_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSE213216 validation data.")
    parser.add_argument("--archive", type=Path, default=Path("data/GSE213216_RAW.tar"))
    parser.add_argument(
        "--miniml",
        type=Path,
        default=Path("output/source_metadata/GSE213216/GSE213216_family.xml"),
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=Path("data/GSE213216_processed"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/reports/validation/GSE213216/sample_metadata.csv"),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Build the manifest without extracting nested H5 matrices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = parse_gse213216_miniml(args.miniml)
    prepared = prepare_gse213216_manifest(
        metadata=metadata,
        archive_path=args.archive,
        matrix_dir=args.matrix_dir,
        extract=not args.metadata_only,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.manifest, index=False)

    ready = int(prepared["include_in_sc_eda"].sum())
    spatial = int((prepared["technology"] == "spatial_transcriptomics").sum())
    print(f"Official GEO samples: {len(prepared)}")
    print(f"Single-cell filtered matrices ready: {ready}")
    print(f"Spatial samples excluded from scRNA-seq QC: {spatial}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
