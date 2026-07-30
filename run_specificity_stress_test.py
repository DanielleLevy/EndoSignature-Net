"""Run an internal endometriosis-versus-other-pathology specificity stress test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.baseline_modeling import (
    BaselineSettings,
    plot_baseline_results,
    prepare_modeling_table,
    run_repeated_nested_cv,
    settings_dict,
    summarize_repeated_metrics,
)


SPECIFICITY_MODEL_ORDER = [
    "prevalence_only",
    "cycle_only",
    "pdzd2_only",
    "specificity_watchlist_3",
    "frozen_signature_12",
    "frozen_signature_12_plus_cycle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test the frozen signature against other uterine or pelvic "
            "pathology using repeated nested patient-level cross-validation."
        )
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
        "--signature",
        type=Path,
        default=Path(
            "output/reports/evidence_integration/v1.0/"
            "frozen_extended_clean_control_signature.csv"
        ),
    )
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=Path(
            "output/reports/evidence_integration/v1.0/specificity_watchlist.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--outer-repeats", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = BaselineSettings(
        outer_repeats=args.outer_repeats,
        primary_target="endometriosis_vs_other_uterine_pelvic_pathology",
        evaluation_status=(
            "internal_specificity_stress_test_GSE51981_previously_used_for_evidence"
        ),
    )
    signature_genes = pd.read_csv(args.signature)["gene"].tolist()
    watchlist_genes = pd.read_csv(args.watchlist)["gene"].tolist()
    table = prepare_modeling_table(
        pd.read_csv(args.expression),
        pd.read_csv(args.metadata),
        signature_genes,
        group_mapping={
            "Endometriosis": 1,
            "Non_endometriosis_other_pathology": 0,
        },
    )
    metrics, predictions, tuning = run_repeated_nested_cv(
        table,
        signature_genes,
        settings,
        additional_gene_models={"specificity_watchlist_3": watchlist_genes},
        model_order=SPECIFICITY_MODEL_ORDER,
    )
    summary = summarize_repeated_metrics(metrics)
    reports = (
        args.output_dir
        / "reports"
        / "modeling"
        / "GSE51981_specificity_stress_test"
        / settings.signature_version
    )
    plots = (
        args.output_dir
        / "eda_plots"
        / "modeling"
        / "GSE51981_specificity_stress_test"
        / settings.signature_version
    )
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "modeling_cohort.csv", index=False)
    metrics.to_csv(reports / "repeat_level_metrics.csv", index=False)
    predictions.to_csv(reports / "repeated_oof_predictions.csv.gz", index=False)
    tuning.to_csv(reports / "nested_tuning_results.csv", index=False)
    summary.to_csv(reports / "performance_summary.csv", index=False)
    settings_payload = settings_dict(settings)
    settings_payload["watchlist_genes"] = watchlist_genes
    settings_payload["model_order"] = SPECIFICITY_MODEL_ORDER
    with (reports / "specificity_settings.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(settings_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    mean_results = {
        model: {
            row["metric"]: float(row["mean"])
            for _, row in summary.loc[summary["model"].eq(model)].iterrows()
        }
        for model in summary["model"].unique()
    }
    result_payload = {
        "cohort": {
            "n_patients": len(table),
            "n_endometriosis": int(table["target"].sum()),
            "n_other_pathology": int((1 - table["target"]).sum()),
        },
        "mean_performance": mean_results,
        "interpretation": {
            "full_signature_mean_roc_auc_above_0_70": bool(
                mean_results["frozen_signature_12"]["roc_auc"] > 0.70
            ),
            "full_signature_outperforms_cycle_only": bool(
                mean_results["frozen_signature_12"]["roc_auc"]
                > mean_results["cycle_only"]["roc_auc"]
            ),
            "claim_boundary": (
                "Internal stress test only. GSE51981 contributed to evidence "
                "integration and the three-gene watchlist was defined from "
                "this same contrast. Results are not independent specificity "
                "validation."
            ),
        },
    }
    with (reports / "specificity_results_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_baseline_results(
        metrics,
        predictions,
        plots,
        model_order=SPECIFICITY_MODEL_ORDER,
        model_labels={"specificity_watchlist_3": "Watchlist 3"},
        title="Endometriosis vs other pathology: internal stress test",
    )

    wide = summary.pivot(index="model", columns="metric", values="mean")
    print(f"Patients/samples: {len(table)}")
    print(f"Endometriosis: {int(table['target'].sum())}")
    print(f"Other pathology: {int((1 - table['target']).sum())}")
    print(wide[["roc_auc", "pr_auc", "balanced_accuracy"]].round(3))
    print(
        "Interpretation boundary: internal specificity stress test; "
        "not independent external validation."
    )
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
