"""Verify metadata and run bulk RNA-seq QC/EDA for GSE135485."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.bulk_eda import (
    BulkEDASettings,
    align_metadata_to_counts,
    load_raw_count_matrix,
    normalized_expression_and_pca,
    parse_geo_soft_samples,
    plot_bulk_eda,
    sample_correlation_summary,
    sample_qc_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verified GSE135485 bulk EDA.")
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("data/GSE135485_Endometriosis_raw_counts.csv.gz"),
    )
    parser.add_argument(
        "--geo-soft",
        type=Path,
        default=Path("output/source_metadata/GSE135485/GSE135485_family.soft"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = BulkEDASettings()
    settings.validate()
    counts = load_raw_count_matrix(args.counts)
    metadata = align_metadata_to_counts(
        parse_geo_soft_samples(args.geo_soft), counts
    )
    qc = sample_qc_metrics(counts, metadata, settings)
    log_cpm, pca, variance = normalized_expression_and_pca(
        counts, metadata, settings
    )
    correlation = sample_correlation_summary(log_cpm, metadata)
    correlation_lookup = correlation.set_index("count_matrix_column")[
        "median_spearman_correlation_to_other_samples"
    ]
    qc["median_spearman_correlation_to_other_samples"] = qc[
        "count_matrix_column"
    ].map(correlation_lookup)
    low_correlation = (
        qc["median_spearman_correlation_to_other_samples"]
        < settings.min_median_sample_correlation
    )
    qc.loc[low_correlation, "qc_flags"] = qc.loc[low_correlation, "qc_flags"].map(
        lambda value: f"{value};low_sample_correlation".strip(";")
    )
    qc.loc[low_correlation, "qc_status"] = "review"

    reports = args.output_dir / "reports" / "bulk" / "GSE135485"
    plots = args.output_dir / "eda_plots" / "bulk" / "GSE135485"
    reports.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(reports / "verified_sample_metadata.csv", index=False)
    qc.to_csv(reports / "sample_qc_summary.csv", index=False)
    qc.loc[qc["qc_status"] == "review"].to_csv(
        reports / "samples_requiring_review.csv", index=False
    )
    pca.to_csv(reports / "pca_coordinates.csv", index=False)
    variance.to_csv(reports / "pca_explained_variance.csv", index=False)
    correlation.to_csv(reports / "sample_correlation_summary.csv", index=False)
    log_cpm.to_csv(reports / "filtered_log2_cpm.csv.gz")
    with (reports / "bulk_eda_settings.json").open("w", encoding="utf-8") as handle:
        payload = asdict(settings)
        payload.update(
            {
                "input_type": "non_negative_integer_raw_counts",
                "n_input_genes": counts.shape[0],
                "n_samples": counts.shape[1],
                "n_filtered_genes": log_cpm.shape[0],
                "comparison_warning": (
                    "endometriosis group combines endometrial samples and lesions"
                ),
            }
        )
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (reports / "analysis_suitability.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "primary_clean_control_vs_eutopic_validation": "not_supported_by_available_metadata",
                "secondary_broad_pathological_vs_healthy_analysis": "possible_with_strict_sensitivity_analysis",
                "reasons": [
                    "Only four healthy-endometrium controls are present in the local matrix.",
                    "The 54 endometriosis records combine endometrial samples and lesions without sample-level separation.",
                    "Three of four controls were sequenced in lane L001, creating partial condition-lane confounding.",
                    "GSM4012536 is a strong low-complexity and low-correlation outlier.",
                ],
                "recommended_next_actions": [
                    "Exclude GSM4012536 from the primary sensitivity model and rerun with it included as a robustness check.",
                    "Model sequencing lane as a covariate where estimable.",
                    "Treat results as broad pathological-versus-healthy validation, not cell-type or eutopic-specific replication.",
                    "Require effect-direction consistency rather than relying only on nominal significance.",
                ],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    plot_bulk_eda(qc, pca, variance, log_cpm, plots)

    print(f"Raw count matrix: {counts.shape[0]} genes x {counts.shape[1]} samples")
    print(f"Conditions: {metadata['condition'].value_counts().to_dict()}")
    print(f"Genes retained for EDA: {log_cpm.shape[0]}")
    print(f"Samples flagged for review: {(qc['qc_status'] == 'review').sum()}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
