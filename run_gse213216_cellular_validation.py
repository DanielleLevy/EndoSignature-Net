"""Test replicated signatures for cell-family localization in GSE213216."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.cellular_validation import (
    CellularValidationSettings,
    aggregate_supported_expression,
    evaluate_cell_family_localization,
    plot_cellular_validation,
    select_candidate_gene_families,
    summarize_gene_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run marker-supported GSE213216 cellular validation."
    )
    parser.add_argument(
        "--bulk-gene-family-results",
        type=Path,
        default=Path(
            "output/reports/validation/GSE135485_signature_validation/"
            "gene_family_validation_results.csv.gz"
        ),
    )
    parser.add_argument(
        "--validated-artifacts",
        type=Path,
        default=Path("output/artifacts/GSE213216_marker_validated_samples"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = CellularValidationSettings()
    settings.validate()
    bulk_results = pd.read_csv(args.bulk_gene_family_results)
    candidates = select_candidate_gene_families(bulk_results)
    paths = sorted(args.validated_artifacts.glob("*_marker_validated.h5ad"))
    if not paths:
        raise SystemExit(f"No marker-validated artifacts found in {args.validated_artifacts}")

    patient_groups, availability = aggregate_supported_expression(
        paths, sorted(candidates["gene"].unique()), settings
    )
    results, paired = evaluate_cell_family_localization(
        patient_groups, candidates, settings
    )
    gene_summary = summarize_gene_evidence(results)
    gene_availability = (
        availability.groupby("gene", observed=True)["available"]
        .agg(n_artifacts_available="sum", n_artifacts_checked="size")
        .reset_index()
    )

    reports = (
        args.output_dir
        / "reports"
        / "validation"
        / "GSE213216"
        / "candidate_localization"
    )
    plots = (
        args.output_dir
        / "eda_plots"
        / "validation"
        / "GSE213216"
        / "candidate_localization"
    )
    reports.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(reports / "predeclared_candidate_gene_families.csv", index=False)
    patient_groups.to_csv(
        reports / "patient_tissue_family_expression.csv.gz", index=False
    )
    paired.to_csv(reports / "paired_patient_localization_metrics.csv.gz", index=False)
    results.to_csv(reports / "cell_family_localization_results.csv", index=False)
    gene_summary.to_csv(reports / "gene_localization_summary.csv", index=False)
    gene_availability.to_csv(
        reports / "candidate_gene_availability.csv", index=False
    )

    high_priority = results["priority_tier"].eq("high_priority_fdr_supported")
    summary = {
        **asdict(settings),
        "analysis_scope": "marker_supported_cell_family_localization_not_disease_direction",
        "n_marker_validated_artifacts": len(paths),
        "n_candidate_genes": candidates["gene"].nunique(),
        "n_candidate_gene_family_pairs": len(candidates),
        "n_genes_available_in_all_artifacts": int(
            gene_availability["n_artifacts_available"]
            .eq(len(paths))
            .sum()
        ),
        "minimum_artifacts_with_a_candidate_gene": int(
            gene_availability["n_artifacts_available"].min()
        ),
        "n_localization_tests": len(results),
        "n_eligible_localization_tests": int(
            results["eligible_for_localization_test"].sum()
        ),
        "n_supported_localization_tests": int(results["cell_family_localized"].sum()),
        "n_directionally_consistent_localization_tests": int(
            results["directionally_consistent_localization"].sum()
        ),
        "n_high_priority_gene_family_tests_supported": int(
            (high_priority & results["cell_family_localized"]).sum()
        ),
        "interpretation_limits": [
            "GSE213216 has no comparable healthy-control group, so disease direction is not tested.",
            "Transferred families are marker-concordant labels, not independent ground truth.",
            "Candidate localization does not establish diagnostic performance.",
            "Ovary and endometrioma tissues are excluded from the primary cohort.",
        ],
    }
    with (reports / "cellular_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_cellular_validation(results, plots)

    print(
        f"Candidates: {candidates['gene'].nunique()} genes in "
        f"{len(candidates)} discovery gene-family pairs"
    )
    print(
        f"Localization tests: {len(results)}; eligible: "
        f"{summary['n_eligible_localization_tests']}; supported: "
        f"{summary['n_supported_localization_tests']}"
    )
    print(
        "High-priority supported tests: "
        f"{summary['n_high_priority_gene_family_tests_supported']}"
    )
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
