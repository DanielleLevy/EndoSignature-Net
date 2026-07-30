"""Validate predeclared GSE179640 signatures in GSE135485 bulk RNA-seq."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.bulk_validation import (
    BulkValidationSettings,
    collapse_discovery_candidates,
    expand_gene_family_results,
    plot_validation_results,
    track_predefined_genes,
    validate_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conservative tissue-level signature validation."
    )
    parser.add_argument(
        "--discovery-results",
        type=Path,
        default=Path(
            "output/reports/signatures/GSE179640_pseudobulk/"
            "cell_family_differential_expression.csv.gz"
        ),
    )
    parser.add_argument(
        "--log-cpm",
        type=Path,
        default=Path(
            "output/reports/bulk/GSE135485/filtered_log2_cpm.csv.gz"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "output/reports/bulk/GSE135485/verified_sample_metadata.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = BulkValidationSettings()
    settings.validate()
    discovery = pd.read_csv(args.discovery_results)
    log_cpm = pd.read_csv(args.log_cpm, index_col=0)
    metadata = pd.read_csv(args.metadata)
    candidates, conflicts, gene_family_rows = collapse_discovery_candidates(
        discovery, settings.discovery_tier
    )
    validation, unavailable = validate_candidates(
        candidates, log_cpm, metadata, settings
    )
    expanded = expand_gene_family_results(validation, gene_family_rows)
    tracked = track_predefined_genes(
        settings.tracked_genes, discovery, log_cpm, metadata, settings
    )

    reports = (
        args.output_dir
        / "reports"
        / "validation"
        / "GSE135485_signature_validation"
    )
    plots = (
        args.output_dir
        / "eda_plots"
        / "validation"
        / "GSE135485_signature_validation"
    )
    reports.mkdir(parents=True, exist_ok=True)
    validation.to_csv(reports / "gene_level_validation_results.csv", index=False)
    expanded.to_csv(
        reports / "gene_family_validation_results.csv.gz", index=False
    )
    conflicts.to_csv(reports / "conflicting_discovery_directions.csv", index=False)
    unavailable.to_csv(reports / "unavailable_candidate_genes.csv", index=False)
    tracked.to_csv(reports / "predefined_gene_tracking.csv", index=False)

    summary = {
        **asdict(settings),
        "analysis_scope": (
            "secondary_broad_pathological_vs_healthy_tissue_level_validation"
        ),
        "model": "OLS_log2_CPM_with_condition_and_categorical_lane_HC3_SE",
        "multiplicity_scope": "available_predeclared_unique_candidate_genes",
        "n_discovery_gene_family_rows": len(gene_family_rows),
        "n_unique_consensus_candidate_genes": len(candidates),
        "n_conflicting_direction_genes": len(conflicts),
        "n_available_candidate_genes": len(validation),
        "n_unavailable_candidate_genes": len(unavailable),
        "n_directionally_replicated": int(
            validation["directionally_replicated"].sum()
        ),
        "n_statistically_supported": int(
            validation["statistically_supported"].sum()
        ),
        "interpretation_limits": [
            "Bulk tissue cannot establish which cell family generated an association.",
            "The disease group combines endometrial samples and lesions.",
            "Only four healthy controls are available.",
            "Condition and sequencing lane are partially confounded.",
            "Results are secondary tissue-level support, not diagnostic validation.",
        ],
    }
    with (reports / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_validation_results(validation, plots)

    print(
        f"Discovery candidates: {len(gene_family_rows)} gene-family rows, "
        f"{len(candidates)} consensus genes"
    )
    print(
        f"Available in filtered GSE135485: {len(validation)}; "
        f"unavailable: {len(unavailable)}"
    )
    print(
        f"Directionally replicated: {summary['n_directionally_replicated']}; "
        f"statistically supported: {summary['n_statistically_supported']}"
    )
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
