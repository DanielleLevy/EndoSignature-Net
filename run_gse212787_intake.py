"""Run the outcome-blind GSE212787 external-cohort intake."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.external_cohort_intake import FROZEN_GENES
from src.rnaseq_external_intake import (
    RNASeqIntakeSettings,
    extract_frozen_expression,
    label_blind_count_qc,
    load_count_and_fpkm,
    load_gse212787_metadata,
    load_ncbi_ensembl_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-blind GSE212787 metadata, mapping, and count QC."
    )
    parser.add_argument(
        "--series-matrix",
        type=Path,
        default=Path(
            "output/source_metadata/GSE212787/GSE212787_series_matrix.txt.gz"
        ),
    )
    parser.add_argument(
        "--expression",
        type=Path,
        default=Path(
            "output/source_metadata/GSE212787/GSE212787_Allgene_info.txt.gz"
        ),
    )
    parser.add_argument(
        "--gene-info",
        type=Path,
        default=Path("output/source_metadata/GSE212787/Homo_sapiens.gene_info.gz"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE212787_intake"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    settings = RNASeqIntakeSettings()
    metadata, series = load_gse212787_metadata(args.series_matrix)
    counts, fpkm = load_count_and_fpkm(args.expression, metadata)
    mapping = load_ncbi_ensembl_mapping(args.gene_info)
    target_ids = metadata.loc[
        metadata["include_in_external_target"], "sample_id"
    ].tolist()
    signature_counts = extract_frozen_expression(counts[target_ids], mapping)
    signature_fpkm = extract_frozen_expression(fpkm[target_ids], mapping)
    qc, correlation, pca, explained, log_cpm = label_blind_count_qc(
        counts, metadata, settings
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.report_dir / "verified_sample_metadata.csv", index=False)
    mapping.to_csv(args.report_dir / "frozen_gene_ensembl_mapping.csv", index=False)
    signature_counts.to_csv(args.report_dir / "frozen_signature_counts.csv.gz")
    signature_fpkm.to_csv(args.report_dir / "frozen_signature_fpkm.csv.gz")
    log_cpm[target_ids].to_csv(args.report_dir / "target_log_cpm.csv.gz")
    qc.to_csv(args.report_dir / "label_blind_sample_qc.csv", index=False)
    qc.loc[qc["qc_status"].eq("review")].to_csv(
        args.report_dir / "samples_requiring_review.csv", index=False
    )
    correlation.to_csv(args.report_dir / "sample_spearman_correlation.csv.gz")
    pca.to_csv(args.report_dir / "unlabelled_pca_coordinates.csv", index=False)
    explained.to_csv(args.report_dir / "unlabelled_pca_explained_variance.csv", index=False)

    missing = [gene for gene in FROZEN_GENES if gene not in signature_counts.index]
    all_expressed = [
        gene
        for gene in signature_counts.index
        if int(signature_counts.loc[gene].sum()) > 0
    ]
    reviews = qc.loc[qc["qc_status"].eq("review"), "sample_id"].tolist()
    duplicate_profiles = qc.loc[
        qc["maximum_spearman_to_other_sample"].gt(0.9999), "sample_id"
    ].tolist()
    blocking_reviews = qc.loc[
        qc["qc_flags"].fillna("").str.contains(
            "low_global_sample_correlation|possible_duplicate_expression_profile",
            regex=True,
        ),
        "sample_id",
    ].tolist()
    decision = (
        "pass_to_predeclared_directional_replication"
        if not missing and len(all_expressed) == len(FROZEN_GENES)
        and not blocking_reviews and not duplicate_profiles
        else "hold_for_review"
    )
    payload = {
        "study_id": "GSE212787",
        "analysis_stage": "label_blind_intake_only",
        "outcome_performance_inspected": False,
        "total_samples": len(metadata),
        "target_samples": len(target_ids),
        "target_case_samples": int(
            (
                metadata["include_in_external_target"]
                & metadata["condition"].eq("Endometriosis")
            ).sum()
        ),
        "target_control_samples": int(
            (
                metadata["include_in_external_target"]
                & metadata["condition"].eq("Disease_free_control")
            ).sum()
        ),
        "excluded_ectopic_samples": int(
            metadata["tissue_class"].eq("endometriosis_ectopic").sum()
        ),
        "unique_target_patients": int(
            metadata.loc[
                metadata["include_in_external_target"], "patient_id"
            ].nunique()
        ),
        "all_12_frozen_genes_mapped": not missing,
        "missing_frozen_genes": missing,
        "frozen_genes_with_nonzero_total_counts": all_expressed,
        "label_blind_qc_review_samples": reviews,
        "blocking_qc_review_samples": blocking_reviews,
        "nonblocking_review_policy": (
            "retain isolated library-size outliers after library-size normalization "
            "and include a leave-one-patient-out sensitivity analysis"
        ),
        "possible_duplicate_profiles": duplicate_profiles,
        "eligibility_decision": decision,
        "expression_units": {
            "primary_qc": "deposited_featurecounts_transformed_to_log2_cpm_plus_1",
            "deposited_secondary": "FPKM",
        },
        "settings": asdict(settings),
        "series_title": series.get("title", ""),
        "input_sha256": {
            args.series_matrix.name: sha256(args.series_matrix),
            args.expression.name: sha256(args.expression),
            args.gene_info.name: sha256(args.gene_info),
        },
        "known_limitations": [
            "only_13_target_patients",
            "mixed_proliferative_and_secretory_cycle_phases",
            "no_other_pathology_controls",
            "processed_count_matrix_without_sample_level_raw_read_qc",
        ],
        "next_action": (
            "run_frozen_directional_replication_without_tuning"
            if decision == "pass_to_predeclared_directional_replication"
            else "resolve_intake_flags_before_outcome_analysis"
        ),
    }
    with (args.report_dir / "intake_eligibility.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"All samples: {len(metadata)}; target eutopic samples: {len(target_ids)}")
    print(f"Frozen genes mapped: {len(signature_counts)}/{len(FROZEN_GENES)}")
    print(f"QC review samples: {reviews}")
    print(f"Decision: {decision}")
    print(f"Reports: {args.report_dir}")


if __name__ == "__main__":
    main()
