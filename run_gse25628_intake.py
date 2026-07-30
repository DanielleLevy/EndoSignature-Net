"""Run label-blind target definition, mapping, and QC for GSE25628."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.external_cohort_intake import FROZEN_GENES
from src.gse25628_intake import load_gse25628_series_matrix
from src.microarray_eda import (
    MicroarrayEDASettings,
    collapse_probes_to_genes,
    load_gpl570_annotation,
    processed_expression_qc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-blind GSE25628 intake.")
    parser.add_argument(
        "--series-matrix",
        type=Path,
        default=Path(
            "output/source_metadata/GSE25628/GSE25628_series_matrix.txt.gz"
        ),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("output/source_metadata/GSE25628/GPL571.annot.gz"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE25628_intake"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression, metadata, series = load_gse25628_series_matrix(args.series_matrix)
    annotation = load_gpl570_annotation(args.annotation)
    gene_expression, mapping = collapse_probes_to_genes(expression, annotation)
    target_metadata = metadata.loc[metadata["include_in_external_target"]].copy()
    target_ids = target_metadata["sample_id"].tolist()
    target_expression = gene_expression[target_ids]
    qc, correlation = processed_expression_qc(
        target_expression, target_metadata, MicroarrayEDASettings()
    )

    available = [gene for gene in FROZEN_GENES if gene in target_expression.index]
    missing = [gene for gene in FROZEN_GENES if gene not in target_expression.index]
    signature_mapping = mapping.loc[mapping["gene"].isin(FROZEN_GENES)].copy()
    signature_mapping["frozen_gene_available"] = True
    missing_rows = pd.DataFrame(
        {
            "gene": missing,
            "n_probes": 0,
            "probe_ids": "",
            "frozen_gene_available": False,
        }
    )
    signature_mapping = pd.concat(
        [signature_mapping, missing_rows], ignore_index=True
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.report_dir / "verified_sample_metadata.csv", index=False)
    signature_mapping.to_csv(
        args.report_dir / "frozen_signature_platform_coverage.csv", index=False
    )
    target_expression.loc[available].to_csv(
        args.report_dir / "available_frozen_gene_expression.csv.gz"
    )
    qc.to_csv(args.report_dir / "label_blind_sample_qc.csv", index=False)
    qc.loc[qc["qc_status"].eq("review")].to_csv(
        args.report_dir / "samples_requiring_review.csv", index=False
    )
    correlation.to_csv(args.report_dir / "sample_spearman_correlation.csv.gz")

    review_samples = qc.loc[qc["qc_status"].eq("review"), "sample_id"].tolist()
    payload = {
        "study_id": "GSE25628",
        "analysis_stage": "label_blind_intake_only",
        "outcome_performance_inspected": False,
        "total_samples": len(metadata),
        "target_samples": len(target_metadata),
        "target_cases": int(target_metadata["condition"].eq("Endometriosis").sum()),
        "target_controls": int(
            target_metadata["condition"].eq("Disease_free_control").sum()
        ),
        "excluded_ectopic_samples": int(
            metadata["tissue_class"].eq("endometriosis_ectopic").sum()
        ),
        "cycle_phase": "Proliferative",
        "frozen_genes_available": available,
        "frozen_genes_missing": missing,
        "frozen_gene_coverage": f"{len(available)}/{len(FROZEN_GENES)}",
        "label_blind_qc_review_samples": review_samples,
        "patient_identifiers_available": False,
        "eligibility_decision": (
            "ineligible_for_locked_12_gene_model"
            if missing
            else "pass_to_locked_external_replication"
        ),
        "claim_boundary": (
            "The locked 12-gene model cannot be applied when platform genes are "
            "missing. A reduced-panel analysis would be exploratory."
        ),
        "series_title": series.get("title", ""),
    }
    with (args.report_dir / "intake_eligibility.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Target: {len(target_metadata)} samples")
    print(f"Frozen-gene coverage: {len(available)}/{len(FROZEN_GENES)}")
    print(f"Missing genes: {missing}")
    print(f"QC review samples: {review_samples}")
    print(f"Decision: {payload['eligibility_decision']}")


if __name__ == "__main__":
    main()
