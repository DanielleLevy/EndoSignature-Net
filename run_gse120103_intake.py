"""Run the outcome-blind GSE120103 external-cohort intake."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.external_cohort_intake import (
    FROZEN_GENES,
    ExternalIntakeSettings,
    label_blind_processed_qc,
    load_gpl6480_annotation,
    load_gse120103_series_matrix,
    map_frozen_signature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-blind mapping and processed QC for GSE120103."
    )
    parser.add_argument(
        "--series-matrix",
        type=Path,
        default=Path(
            "output/source_metadata/GSE120103/GSE120103_series_matrix.txt.gz"
        ),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("output/source_metadata/GSE120103/GPL6480.annot.gz"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("output/reports/external_validation/v1.0/GSE120103_intake"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a reproducibility checksum for an input file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    settings = ExternalIntakeSettings()
    expression, metadata, series = load_gse120103_series_matrix(args.series_matrix)
    annotation = load_gpl6480_annotation(args.annotation)
    signature, mapping = map_frozen_signature(expression, annotation)
    qc, correlation, pca, explained = label_blind_processed_qc(
        expression, metadata, settings
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.report_dir / "verified_sample_metadata.csv", index=False)
    qc.to_csv(args.report_dir / "label_blind_sample_qc.csv", index=False)
    qc.loc[qc["qc_status"].eq("review")].to_csv(
        args.report_dir / "samples_requiring_review.csv", index=False
    )
    mapping.to_csv(args.report_dir / "frozen_signature_probe_mapping.csv", index=False)
    signature.to_csv(args.report_dir / "frozen_signature_expression.csv.gz")
    correlation.to_csv(args.report_dir / "sample_spearman_correlation.csv.gz")
    pca.to_csv(args.report_dir / "unlabelled_pca_coordinates.csv", index=False)
    explained.to_csv(args.report_dir / "unlabelled_pca_explained_variance.csv", index=False)

    mapped_genes = set(signature.index)
    missing_genes = [gene for gene in FROZEN_GENES if gene not in mapped_genes]
    balanced_cells = (
        metadata.groupby(["condition", "fertility"], observed=True).size().tolist()
        == [9, 9, 9, 9]
    )
    review_n = int(qc["qc_status"].eq("review").sum())
    exact_duplicate_n = int(
        (qc["maximum_spearman_to_other_sample"] > 0.9999).sum()
    )
    eligibility = {
        "study_id": "GSE120103",
        "analysis_stage": "label_blind_intake_only",
        "outcome_performance_inspected": False,
        "all_12_frozen_genes_mapped": not missing_genes,
        "mapped_gene_count": len(mapped_genes),
        "missing_genes": missing_genes,
        "unique_geo_samples": metadata["sample_id"].nunique() == 36,
        "distinct_participants_supported_by_primary_publication": True,
        "patient_identifiers_available": False,
        "balanced_disease_by_fertility_design": balanced_cells,
        "cycle_phase": "Secretory",
        "cycle_phase_resolved_from_source_name_and_primary_publication": True,
        "label_blind_qc_review_sample_count": review_n,
        "possible_duplicate_profile_sample_count": exact_duplicate_n,
        "source_name_fertility_disagreement_count": int(
            (~metadata["source_name_fertility_concordant"]).sum()
        ),
        "metadata_disagreement_explanation": (
            "The nine Group 2B source-name fields say Fertile while their titles, "
            "sample-group characteristics, group code, and publication identify "
            "them as Infertile; fertility was resolved from the concordant fields."
        ),
        "eligibility_decision": (
            "pass_to_locked_outcome_analysis"
            if not missing_genes and exact_duplicate_n == 0
            else "hold_for_review"
        ),
        "input_sha256": {
            args.series_matrix.name: sha256(args.series_matrix),
            args.annotation.name: sha256(args.annotation),
        },
        "settings": asdict(settings),
        "series_title": series.get("title", ""),
        "expression_scale_assessment": (
            "deposited transformed normalized values include negatives; no additional "
            "log transform was applied"
        ),
        "probe_selection_rule": "highest_across_sample_median_label_blind",
    }
    with (args.report_dir / "intake_eligibility.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(eligibility, handle, indent=2, sort_keys=True)
        handle.write("\n")

    pd.DataFrame(
        {
            "check": [
                "all_12_frozen_genes_mapped",
                "36_unique_geo_samples",
                "distinct_participants_supported_by_publication",
                "no_exact_duplicate_expression_profiles",
                "processed_expression_qc_completed",
            ],
            "passed": [
                not missing_genes,
                metadata["sample_id"].nunique() == 36,
                True,
                exact_duplicate_n == 0,
                True,
            ],
        }
    ).to_csv(args.report_dir / "eligibility_checklist.csv", index=False)

    print(f"Processed matrix: {expression.shape[0]} probes x {expression.shape[1]} samples")
    print(f"Frozen genes mapped: {len(mapped_genes)}/{len(FROZEN_GENES)}")
    print(f"Samples requiring technical review: {review_n}")
    print(f"Eligibility: {eligibility['eligibility_decision']}")
    print(f"Reports: {args.report_dir}")


if __name__ == "__main__":
    main()
