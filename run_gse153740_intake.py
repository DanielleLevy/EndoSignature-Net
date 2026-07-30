"""Run the outcome-blind GSE153740 transcript-level intake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.external_cohort_intake import FROZEN_GENES
from src.gse153740_intake import (
    label_blind_fpkm_qc,
    load_and_collapse_fpkm,
    load_ensembl90_transcript_mapping,
    load_gse153740_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-blind GSE153740 intake.")
    parser.add_argument(
        "--series-matrix",
        type=Path,
        default=Path(
            "output/source_metadata/GSE153740/GSE153740_series_matrix.txt.gz"
        ),
    )
    parser.add_argument(
        "--expression",
        type=Path,
        default=Path(
            "output/source_metadata/GSE153740/"
            "GSE153740_transcript_expression_matrix.txt.gz"
        ),
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        default=Path(
            "output/source_metadata/GSE153740/Homo_sapiens.GRCh38.90.gtf.gz"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE153740_intake"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, series = load_gse153740_metadata(args.series_matrix)
    transcript_mapping = load_ensembl90_transcript_mapping(args.gtf)
    transcript, gene_fpkm, coverage = load_and_collapse_fpkm(
        args.expression, metadata, transcript_mapping
    )
    qc, correlation = label_blind_fpkm_qc(transcript, metadata)
    available = [gene for gene in FROZEN_GENES if gene in gene_fpkm.index]
    missing = [gene for gene in FROZEN_GENES if gene not in gene_fpkm.index]
    signature = gene_fpkm.loc[available]
    nonzero_all = signature.gt(0).all(axis=1)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.report_dir / "verified_sample_metadata.csv", index=False)
    coverage.loc[coverage["gene"].isin(FROZEN_GENES)].to_csv(
        args.report_dir / "frozen_gene_transcript_mapping.csv", index=False
    )
    signature.to_csv(args.report_dir / "frozen_signature_gene_fpkm.csv.gz")
    np.log2(signature + 1).to_csv(
        args.report_dir / "frozen_signature_log2_fpkm.csv.gz"
    )
    qc.to_csv(args.report_dir / "label_blind_sample_qc.csv", index=False)
    qc.loc[qc["qc_status"].eq("review")].to_csv(
        args.report_dir / "samples_requiring_review.csv", index=False
    )
    correlation.to_csv(args.report_dir / "sample_spearman_correlation.csv.gz")

    reviews = qc.loc[qc["qc_status"].eq("review"), "sample_id"].tolist()
    blocking = qc.loc[
        qc["qc_flags"].fillna("").str.contains("low_global_sample_correlation"),
        "sample_id",
    ].tolist()
    decision = (
        "pass_to_locked_external_replication"
        if not missing and bool(nonzero_all.all()) and not blocking
        else "hold_for_review"
    )
    payload = {
        "study_id": "GSE153740",
        "analysis_stage": "label_blind_intake_only",
        "outcome_performance_inspected": False,
        "samples": len(metadata),
        "cases": int(metadata["condition"].eq("Endometriosis").sum()),
        "controls": int(metadata["condition"].eq("Disease_free_control").sum()),
        "unique_patients": metadata["patient_id"].nunique(),
        "cycle_phase": "Mid_secretory",
        "all_12_frozen_genes_mapped": not missing,
        "missing_genes": missing,
        "genes_nonzero_in_every_sample": nonzero_all[nonzero_all].index.tolist(),
        "label_blind_qc_review_samples": reviews,
        "blocking_qc_samples": blocking,
        "eligibility_decision": decision,
        "expression_representation": (
            "sum of Ensembl-90 transcript FPKM per gene; log2(FPKM+1)"
        ),
        "series_title": series.get("title", ""),
        "limitations": [
            "four_cases_and_four_controls_only",
            "FPKM_without_deposited_raw_count_matrix",
            "one_peritoneal_and_three_ovarian_endometriosis_cases",
            "no_other_pathology_controls",
        ],
    }
    with (args.report_dir / "intake_eligibility.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Samples: {len(metadata)}")
    print(f"Frozen genes: {len(available)}/{len(FROZEN_GENES)}")
    print(f"QC review samples: {reviews}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
