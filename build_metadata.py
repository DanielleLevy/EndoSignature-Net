"""Command-line entry point for creating the project metadata reports.

Usage:
    python build_metadata.py
    python build_metadata.py --data-dir data --output-dir output/reports
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.metadata import archive_members, build_samples, scan_files, summary_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build documented metadata manifests from local data files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing downloaded data")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/reports"), help="Directory for generated CSV reports"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = scan_files(args.data_dir)
    if not files:
        raise SystemExit(f"No files found in {args.data_dir}")

    samples = build_samples(files)
    write_csv(args.output_dir / "file_inventory.csv", files)
    write_csv(args.output_dir / "sample_metadata.csv", samples)
    write_csv(args.output_dir / "sample_summary.csv", summary_rows(samples))
    write_csv(args.output_dir / "archive_inventory.csv", archive_members(files))

    sc_samples = sum(sample.include_in_sc_eda for sample in samples)
    needs_review = sum(sample.review_status == "needs_review" for sample in samples)
    print(f"Documented {len(files)} files as {len(samples)} logical records.")
    print(f"Single-cell samples ready for initial EDA: {sc_samples}")
    print(f"Records requiring manual review: {needs_review}")
    print(f"Reports written to: {args.output_dir}")


if __name__ == "__main__":
    main()
