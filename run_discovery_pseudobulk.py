"""Discover exploratory patient-level cell-family signatures in GSE179640."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

import anndata as ad
import pandas as pd

from src.signatures import (
    PseudobulkSettings,
    aggregate_pseudobulk,
    differential_expression_by_family,
    plot_pseudobulk_results,
    select_exploratory_signatures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSE179640 patient-level pseudobulk.")
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path("output/artifacts/GSE179640_discovery_annotated_provisional.h5ad"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--min-control-patients", type=int, default=3)
    parser.add_argument("--min-endometriosis-patients", type=int, default=6)
    parser.add_argument("--top-genes", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = PseudobulkSettings(
        min_cells_per_pseudobulk=args.min_cells,
        min_control_patients=args.min_control_patients,
        min_endometriosis_patients=args.min_endometriosis_patients,
        top_genes_per_family=args.top_genes,
    )
    settings.validate()
    print(f"Loading discovery atlas: {args.atlas}", flush=True)
    atlas = ad.read_h5ad(args.atlas)
    matrix, metadata, eligibility = aggregate_pseudobulk(atlas, settings)
    print(
        f"Created {len(metadata)} patient-family pseudobulks; "
        f"eligible families: {eligibility['eligible_for_pseudobulk_de'].sum()}",
        flush=True,
    )
    differential_expression = differential_expression_by_family(
        matrix,
        metadata,
        atlas.var_names,
        eligibility,
        settings,
    )
    signatures = select_exploratory_signatures(differential_expression, settings)

    reports = args.output_dir / "reports" / "signatures" / "GSE179640_pseudobulk"
    plots = args.output_dir / "eda_plots" / "signatures" / "GSE179640_pseudobulk"
    reports.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(reports / "pseudobulk_sample_metadata.csv", index=False)
    eligibility.to_csv(reports / "cell_family_eligibility.csv", index=False)
    differential_expression.to_csv(
        reports / "cell_family_differential_expression.csv.gz", index=False
    )
    signatures.to_csv(reports / "exploratory_signature_candidates.csv", index=False)
    with (reports / "pseudobulk_settings.json").open("w", encoding="utf-8") as handle:
        payload = asdict(settings)
        payload.update(
            {
                "comparison": "endometriosis_eutopic_vs_control_endometrium",
                "replicate_unit": "patient",
                "statistical_scope": "exploratory_small_control_cohort",
            }
        )
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_pseudobulk_results(differential_expression, eligibility, plots, settings)

    supported = (
        differential_expression["statistical_tier"]
        == "fdr_effect_direction_stable"
    ).sum()
    effect_candidates = (
        differential_expression["statistical_tier"] == "effect_size_candidate"
    ).sum()
    print(f"Tested genes across families: {len(differential_expression)}")
    print(f"FDR + effect + leave-one-out stable rows: {supported}")
    print(f"Effect-size-only candidates: {effect_candidates}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
