"""Verify the primary cohort, run Scrublet and document inclusion decisions.

Run after ``build_metadata.py`` and ``run_sc_eda.py``. The script applies the
same cell-level QC thresholds used in the EDA before estimating doublets. Input
matrices are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

import pandas as pd

from src.cohort import (
    DoubletSettings,
    build_inclusion_decisions,
    build_verified_primary_metadata,
    plot_doublet_scores,
    run_scrublet,
    summarize_doublets,
    write_cell_doublets,
)
from src.eda import QCThresholds, calculate_qc, filter_sample, read_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Define the primary cohort and detect doublets per sample.")
    parser.add_argument("--metadata", type=Path, default=Path("output/reports/sample_metadata.csv"))
    parser.add_argument(
        "--qc-thresholds",
        type=Path,
        default=Path("output/reports/qc/GSE179640/qc_thresholds.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--expected-doublet-rate", type=float, default=0.06)
    parser.add_argument("--review-doublet-rate", type=float, default=15.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Process the first N primary samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata, keep_default_na=False)
    verified = build_verified_primary_metadata(metadata)
    if args.limit is not None:
        verified = verified.head(args.limit)

    with args.qc_thresholds.open(encoding="utf-8") as handle:
        qc_thresholds = QCThresholds(**json.load(handle))
    settings = DoubletSettings(args.expected_doublet_rate, args.random_state, args.review_doublet_rate)
    settings.validate()

    reports = args.output_dir / "reports" / "cohort"
    cell_reports = reports / "cell_doublets"
    plot_dir = args.output_dir / "eda_plots" / "doublets" / "GSE179640"
    reports.mkdir(parents=True, exist_ok=True)
    verified.to_csv(reports / "verified_sample_metadata.csv", index=False)

    summaries = []
    errors = []
    print(f"Running per-sample Scrublet for {len(verified)} primary candidates")
    for index, (_, row) in enumerate(verified.iterrows(), start=1):
        sample_id = row["sample_id"]
        print(f"[{index}/{len(verified)}] {sample_id}", flush=True)
        try:
            raw = calculate_qc(read_sample(row))
            qc_filtered = filter_sample(raw, qc_thresholds)
            result = run_scrublet(qc_filtered, settings)
            summaries.append(summarize_doublets(result, row, settings))
            write_cell_doublets(result, sample_id, cell_reports / f"{sample_id}_doublets.csv.gz")
            plot_doublet_scores(result, sample_id, plot_dir / f"{sample_id}_doublet_scores.png")
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error_type": type(exc).__name__, "message": str(exc)})

    summary = pd.DataFrame(summaries)
    error_frame = pd.DataFrame(errors, columns=["sample_id", "error_type", "message"])
    summary.to_csv(reports / "doublet_summary.csv", index=False)
    error_frame.to_csv(reports / "doublet_processing_errors.csv", index=False)

    decisions = build_inclusion_decisions(metadata, summary)
    decisions.to_csv(reports / "sample_inclusion_decisions.csv", index=False)
    decisions.loc[decisions["decision"] == "include"].to_csv(reports / "primary_cohort_samples.csv", index=False)
    with (reports / "doublet_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings.__dict__, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Completed: {len(summary)}; failed: {len(error_frame)}")
    if not summary.empty:
        print(f"Predicted doublets: {summary['n_predicted_doublets'].sum()}")
        print(f"Samples requiring doublet review: {(summary['doublet_qc_status'] == 'review').sum()}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
