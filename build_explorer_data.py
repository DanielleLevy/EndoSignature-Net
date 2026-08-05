"""Build a small, frozen result bundle for the Streamlit research explorer.

The exported files contain aggregate research results only. They do not include
patient-level expression or predictions and do not retrain any model.
"""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "explorer_data"


def read_json(path):
    return json.loads((ROOT / path).read_text())


def build_performance():
    internal = read_json(
        "output/reports/modeling/GSE51981_internal_baseline/v1.0/baseline_results_summary.json"
    )
    first = read_json(
        "output/reports/external_validation/v1.0/GSE212787_directional_replication/external_replication_summary.json"
    )
    second = read_json(
        "output/reports/external_validation/v1.0/GSE153740_directional_replication/external_replication_summary.json"
    )
    return pd.DataFrame(
        [
            {
                "cohort": "GSE51981",
                "analysis": "Internal repeated patient-level validation",
                "roc_auc": internal["mean_performance"]["frozen_signature_12"]["roc_auc"],
                "ci_lower": pd.NA,
                "ci_upper": pd.NA,
                "n_patients": internal["cohort"]["n_patients"],
                "status": "Promising internal discrimination",
                "is_external": False,
            },
            {
                "cohort": "GSE212787",
                "analysis": "Locked external application",
                "roc_auc": first["model_metrics"]["roc_auc"],
                "ci_lower": first["model_metrics"]["roc_auc_ci_lower"],
                "ci_upper": first["model_metrics"]["roc_auc_ci_upper"],
                "n_patients": 13,
                "status": "Promising but inconclusive",
                "is_external": True,
            },
            {
                "cohort": "GSE153740",
                "analysis": "Locked mid-secretory external application",
                "roc_auc": second["model_metrics"]["roc_auc"],
                "ci_lower": second["model_metrics"]["roc_auc_ci_lower"],
                "ci_upper": second["model_metrics"]["roc_auc_ci_upper"],
                "n_patients": second["n_cases"] + second["n_controls"],
                "status": "Not replicated",
                "is_external": True,
            },
        ]
    )


def main():
    OUTPUT.mkdir(exist_ok=True)
    build_performance().to_csv(OUTPUT / "cohort_performance.csv", index=False)

    effects = pd.read_csv(
        ROOT / "output/reports/model_interpretability/v1.0/cross_cohort_gene_effects.csv"
    ).rename(columns={"case_control_standardized_mean_difference": "standardized_effect"})
    effects.to_csv(OUTPUT / "gene_evidence.csv", index=False)

    importance = pd.read_csv(
        ROOT / "output/reports/model_interpretability/v1.0/gene_importance_and_stability.csv"
    )
    literature = pd.read_csv(
        ROOT / "output/reports/literature/v1.0/signature_literature_evidence.csv"
    )
    importance.merge(literature, on="gene", how="left", validate="one_to_one").to_csv(
        OUTPUT / "gene_summary.csv", index=False
    )

    cycle = pd.read_csv(
        ROOT / "output/reports/sensitivity/v1.0/gse51981_cycle_stratified_oof_metrics.csv"
    )
    cycle.loc[cycle.model.isin(["frozen_signature_12", "frozen_signature_12_plus_cycle"])].to_csv(
        OUTPUT / "cycle_metrics.csv", index=False
    )
    pd.read_csv(
        ROOT / "output/reports/sensitivity/v1.0/gse179640_cell_family_composition_tests.csv"
    ).to_csv(OUTPUT / "cell_composition.csv", index=False)
    pd.read_csv(
        ROOT / "output/reports/cross_cohort_metadata/v1.0/metadata_completeness.csv"
    ).to_csv(OUTPUT / "metadata_completeness.csv", index=False)

    summary = {
        "title": "EndoSignature Explorer",
        "signature_size": 12,
        "studies_audited": 8,
        "sample_records": 355,
        "patient_records_or_proxies": 324,
        "internal_mean_auc": 0.860,
        "cross_cohort_core": ["PDZD2", "ACSS2"],
        "composition_permanova_r2": 0.094,
        "composition_permanova_p": 0.414,
        "project_status": "Completed portfolio-scale computational research study",
        "claim_boundary": (
            "Research exploration only. The signature is not a validated diagnostic test, "
            "and gene associations do not establish causality."
        ),
        "data_version": "v1.0",
    }
    (OUTPUT / "project_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Explorer bundle written to {OUTPUT}")


if __name__ == "__main__":
    main()

