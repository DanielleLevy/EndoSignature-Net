"""Run the leakage-controlled exploratory GSE51981 baseline benchmark."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the frozen v1.0 signature using repeated nested "
            "patient-level cross-validation."
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
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--outer-repeats", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = BaselineSettings(outer_repeats=args.outer_repeats)
    signature = pd.read_csv(args.signature)
    signature_genes = signature["gene"].tolist()
    table = prepare_modeling_table(
        pd.read_csv(args.expression),
        pd.read_csv(args.metadata),
        signature_genes,
    )
    metrics, predictions, tuning = run_repeated_nested_cv(
        table, signature_genes, settings
    )
    summary = summarize_repeated_metrics(metrics)

    reports = (
        args.output_dir
        / "reports"
        / "modeling"
        / "GSE51981_internal_baseline"
        / settings.signature_version
    )
    plots = (
        args.output_dir
        / "eda_plots"
        / "modeling"
        / "GSE51981_internal_baseline"
        / settings.signature_version
    )
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "modeling_cohort.csv", index=False)
    metrics.to_csv(reports / "repeat_level_metrics.csv", index=False)
    predictions.to_csv(reports / "repeated_oof_predictions.csv.gz", index=False)
    tuning.to_csv(reports / "nested_tuning_results.csv", index=False)
    summary.to_csv(reports / "performance_summary.csv", index=False)
    with (reports / "baseline_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings_dict(settings), handle, indent=2, sort_keys=True)
        handle.write("\n")
    result_summary = {
        "cohort": {
            "n_patients": len(table),
            "n_endometriosis": int(table["target"].sum()),
            "n_clean_controls": int((1 - table["target"]).sum()),
        },
        "mean_performance": {
            model: {
                row["metric"]: float(row["mean"])
                for _, row in summary.loc[summary["model"].eq(model)].iterrows()
            }
            for model in summary["model"].unique()
        },
        "interpretation": {
            "signature_adds_information_beyond_pdzd2": bool(
                summary.loc[
                    summary["model"].eq("frozen_signature_12")
                    & summary["metric"].eq("roc_auc"),
                    "mean",
                ].iloc[0]
                > summary.loc[
                    summary["model"].eq("pdzd2_only")
                    & summary["metric"].eq("roc_auc"),
                    "mean",
                ].iloc[0]
            ),
            "cycle_addition_improves_mean_signature_roc_auc": bool(
                summary.loc[
                    summary["model"].eq("frozen_signature_12_plus_cycle")
                    & summary["metric"].eq("roc_auc"),
                    "mean",
                ].iloc[0]
                > summary.loc[
                    summary["model"].eq("frozen_signature_12")
                    & summary["metric"].eq("roc_auc"),
                    "mean",
                ].iloc[0]
            ),
            "claim_boundary": (
                "Internal exploratory feasibility only. GSE51981 contributed "
                "to signature selection, so this is not untouched external "
                "validation or evidence of endometriosis specificity."
            ),
        },
    }
    with (reports / "baseline_results_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_baseline_results(metrics, predictions, plots)

    wide = summary.pivot(index="model", columns="metric", values="mean")
    print(f"Patients/samples: {len(table)}")
    print(f"Endometriosis: {int(table['target'].sum())}")
    print(f"Clean controls: {int((1 - table['target']).sum())}")
    print(wide[["roc_auc", "pr_auc", "balanced_accuracy"]].round(3))
    print(
        "Interpretation boundary: internal exploratory cross-validation; "
        "not an untouched external test."
    )
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
