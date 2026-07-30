"""Run per-sample Scrublet for the independent GSE213216 validation cohort.

The same secondary cell-QC policy used in the preceding validation EDA is
applied before Scrublet. Predictions are saved at cell level and are treated as
computational flags rather than ground-truth doublet labels.
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
    build_validation_decisions,
    plot_doublet_scores,
    run_scrublet,
    summarize_doublets,
    write_cell_doublets,
)
from src.eda import QCThresholds, calculate_qc, filter_sample, read_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect doublets in GSE213216.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("output/reports/validation/GSE213216/sample_metadata.csv"),
    )
    parser.add_argument(
        "--qc-summary",
        type=Path,
        default=Path("output/reports/qc/GSE213216/qc_sample_summary.csv"),
    )
    parser.add_argument(
        "--qc-thresholds",
        type=Path,
        default=Path("output/reports/qc/GSE213216/qc_thresholds.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--expected-doublet-rate", type=float, default=0.06)
    parser.add_argument("--review-doublet-rate", type=float, default=15.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata, keep_default_na=False)
    qc_summary = pd.read_csv(args.qc_summary, keep_default_na=False)
    eligible = metadata.loc[
        (metadata["technology"] == "scRNA-seq")
        & (metadata["include_in_sc_eda"].astype(str).str.lower() == "true")
    ].sort_values("sample_id")
    if args.limit is not None:
        eligible = eligible.head(args.limit)

    with args.qc_thresholds.open(encoding="utf-8") as handle:
        qc_thresholds = QCThresholds(**json.load(handle))
    settings = DoubletSettings(
        expected_doublet_rate=args.expected_doublet_rate,
        random_state=args.random_state,
        review_rate_pct=args.review_doublet_rate,
    )
    settings.validate()

    reports = args.output_dir / "reports" / "validation" / "GSE213216" / "doublets"
    cell_reports = reports / "cell_predictions"
    plot_dir = args.output_dir / "eda_plots" / "doublets" / "GSE213216"
    reports.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    print(f"Running per-sample Scrublet for {len(eligible)} GSE213216 samples")
    for index, (_, row) in enumerate(eligible.iterrows(), start=1):
        sample_id = row["sample_id"]
        print(f"[{index}/{len(eligible)}] {sample_id}", flush=True)
        try:
            raw = calculate_qc(read_sample(row))
            qc_filtered = filter_sample(raw, qc_thresholds)
            result = run_scrublet(qc_filtered, settings)
            summaries.append(summarize_doublets(result, row, settings))
            write_cell_doublets(
                result,
                sample_id,
                cell_reports / f"{sample_id}_doublets.csv.gz",
            )
            plot_doublet_scores(
                result,
                sample_id,
                plot_dir / f"{sample_id}_doublet_scores.png",
            )
        except Exception as exc:
            errors.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    summary = pd.DataFrame(summaries)
    error_frame = pd.DataFrame(errors, columns=["sample_id", "error_type", "message"])
    summary.to_csv(reports / "doublet_summary.csv", index=False)
    error_frame.to_csv(reports / "doublet_processing_errors.csv", index=False)

    decisions = build_validation_decisions(metadata, qc_summary, summary)
    decisions.to_csv(reports / "sample_decisions.csv", index=False)
    decisions.loc[decisions["decision"] == "include"].to_csv(
        reports / "mapping_candidates.csv", index=False
    )
    decisions.loc[decisions["decision"] == "review"].to_csv(
        reports / "samples_requiring_review.csv", index=False
    )
    with (reports / "doublet_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings.__dict__, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Completed: {len(summary)}; failed: {len(error_frame)}")
    if not summary.empty:
        print(f"Predicted doublets: {summary['n_predicted_doublets'].sum()}")
        print(
            "Samples requiring doublet review: "
            f"{(summary['doublet_qc_status'] == 'review').sum()}"
        )
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
