"""Integrate all dataset evidence and freeze versioned signature tiers."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.evidence_integration import (
    EvidenceIntegrationSettings,
    frozen_signature_tables,
    integrate_evidence,
    modeling_readiness_decision,
    plot_evidence_integration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate evidence and freeze research signature tiers."
    )
    parser.add_argument(
        "--discovery",
        type=Path,
        default=Path(
            "output/reports/signatures/GSE179640_pseudobulk/"
            "cell_family_differential_expression.csv.gz"
        ),
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
        "--cellular-validation",
        type=Path,
        default=Path(
            "output/reports/validation/GSE213216/candidate_localization/"
            "cell_family_localization_results.csv"
        ),
    )
    parser.add_argument(
        "--microarray-validation",
        type=Path,
        default=Path(
            "output/reports/validation/GSE51981_signature_validation/"
            "gene_level_validation_results.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = EvidenceIntegrationSettings()
    evidence = integrate_evidence(
        pd.read_csv(args.discovery),
        pd.read_csv(args.bulk_validation),
        pd.read_csv(args.cellular_validation),
        pd.read_csv(args.microarray_validation),
        settings,
    )
    core, extended, watchlist = frozen_signature_tables(evidence)
    readiness = modeling_readiness_decision(evidence, settings)

    reports = args.output_dir / "reports" / "evidence_integration" / settings.signature_version
    plots = args.output_dir / "eda_plots" / "evidence_integration" / settings.signature_version
    reports.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(reports / "integrated_gene_evidence.csv", index=False)
    core.to_csv(reports / "frozen_core_pathology_signature.csv", index=False)
    extended.to_csv(
        reports / "frozen_extended_clean_control_signature.csv", index=False
    )
    watchlist.to_csv(reports / "specificity_watchlist.csv", index=False)
    evidence.loc[evidence["prior_gse135485_high_priority"]].to_csv(
        reports / "prior_high_priority_reassessment.csv", index=False
    )
    with (reports / "evidence_integration_settings.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(settings), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (reports / "modeling_readiness.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(readiness, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_evidence_integration(evidence, plots)

    print(f"Stable discovery genes integrated: {len(evidence)}")
    print(f"Frozen core pathology signature: {core['gene'].tolist()}")
    print(f"Frozen extended clean-control signature ({len(extended)}): {extended['gene'].tolist()}")
    print(f"Specificity watchlist ({len(watchlist)}): {watchlist['gene'].tolist()}")
    print(
        "Formally endometriosis-specific genes: "
        f"{readiness['n_formally_endometriosis_specific_genes']}"
    )
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
