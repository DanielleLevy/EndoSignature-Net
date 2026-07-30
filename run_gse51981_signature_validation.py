"""Validate predeclared signatures in cycle-adjusted GSE51981 microarrays."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.microarray_validation import (
    MicroarrayValidationSettings,
    plot_microarray_validation,
    select_predeclared_candidates,
    track_predefined_microarray_genes,
    validate_microarray_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GSE51981 candidate signature validation."
    )
    parser.add_argument(
        "--bulk-validation",
        type=Path,
        default=Path(
            "output/reports/validation/GSE135485_signature_validation/"
            "gene_level_validation_results.csv"
        ),
    )
    parser.add_argument(
        "--expression",
        type=Path,
        default=Path(
            "output/reports/microarray/GSE51981/gcrma_gene_expression.csv.gz"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "output/reports/microarray/GSE51981/verified_sample_metadata.csv"
        ),
    )
    parser.add_argument(
        "--qc",
        type=Path,
        default=Path(
            "output/reports/microarray/GSE51981/processed_sample_qc.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = MicroarrayValidationSettings()
    bulk = pd.read_csv(args.bulk_validation)
    expression = pd.read_csv(args.expression, index_col=0)
    metadata = pd.read_csv(args.metadata)
    qc = pd.read_csv(args.qc)
    candidates = select_predeclared_candidates(bulk)
    results, unavailable = validate_microarray_candidates(
        candidates, expression, metadata, qc, settings
    )
    tracked = track_predefined_microarray_genes(
        settings.tracked_genes, expression, metadata, settings
    )

    reports = (
        args.output_dir
        / "reports"
        / "validation"
        / "GSE51981_signature_validation"
    )
    plots = (
        args.output_dir
        / "eda_plots"
        / "validation"
        / "GSE51981_signature_validation"
    )
    reports.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports / "gene_level_validation_results.csv", index=False)
    unavailable.to_csv(reports / "unavailable_candidate_genes.csv", index=False)
    tracked.to_csv(reports / "predefined_gene_tracking.csv", index=False)

    high = results["prior_evidence_tier"].eq("gse135485_fdr_supported")
    summary = {
        **asdict(settings),
        "analysis_scope": (
            "cycle_phase_adjusted_external_tissue_level_candidate_validation"
        ),
        "primary_model": (
            "GCRMA_gene_expression ~ endometriosis + categorical_cycle_phase; "
            "HC3_robust_standard_errors"
        ),
        "n_predeclared_candidates": len(candidates),
        "n_available_candidates": len(results),
        "n_unavailable_candidates": len(unavailable),
        "n_robust_directional_replications": int(
            results["robust_primary_directional_replication"].sum()
        ),
        "n_primary_statistically_supported": int(
            results["primary_statistically_supported"].sum()
        ),
        "n_endometriosis_specific_supported": int(
            results["endometriosis_specific_support"].sum()
        ),
        "n_high_priority_available": int(high.sum()),
        "n_high_priority_primary_supported": int(
            (high & results["primary_statistically_supported"]).sum()
        ),
        "n_high_priority_specific_supported": int(
            (high & results["endometriosis_specific_support"]).sum()
        ),
        "interpretation_limits": [
            "Microarray tissue expression cannot establish cell-family origin.",
            "Cycle phase is adjusted categorically but two disease samples have unknown phase.",
            "Processed GCRMA data support distribution QC, not full raw-CEL diagnostics.",
            "Candidate-set evidence does not establish diagnostic generalization.",
        ],
    }
    with (reports / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_microarray_validation(results, plots)

    print(
        f"Candidates: {len(candidates)}; available: {len(results)}; "
        f"unavailable: {len(unavailable)}"
    )
    print(
        f"Robust directional: {summary['n_robust_directional_replications']}; "
        f"primary FDR-supported: {summary['n_primary_statistically_supported']}; "
        f"specificity-supported: {summary['n_endometriosis_specific_supported']}"
    )
    print(
        f"High-priority primary/specific: "
        f"{summary['n_high_priority_primary_supported']}/"
        f"{summary['n_high_priority_specific_supported']}"
    )
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
