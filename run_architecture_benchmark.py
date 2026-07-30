"""Benchmark compact architectures on both frozen-signature tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.architecture_benchmark import (
    ArchitectureBenchmarkSettings,
    paired_differences,
    plot_architecture_benchmark,
    run_architecture_benchmark,
    settings_dict,
    summarize_architectures,
)
from src.baseline_modeling import prepare_modeling_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare compact architectures using identical nested CV splits."
    )
    parser.add_argument(
        "--expression", type=Path,
        default=Path(
            "output/reports/microarray/GSE51981/gcrma_gene_expression.csv.gz"
        ),
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=Path(
            "output/reports/microarray/GSE51981/verified_sample_metadata.csv"
        ),
    )
    parser.add_argument(
        "--signature", type=Path,
        default=Path(
            "output/reports/evidence_integration/v1.0/"
            "frozen_extended_clean_control_signature.csv"
        ),
    )
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ArchitectureBenchmarkSettings(outer_repeats=args.outer_repeats)
    expression = pd.read_csv(args.expression)
    metadata = pd.read_csv(args.metadata)
    genes = pd.read_csv(args.signature)["gene"].tolist()
    tasks = {
        "endometriosis_vs_clean_control": {
            "Endometriosis": 1,
            "Healthy_control_no_pathology": 0,
        },
        "endometriosis_vs_other_pathology": {
            "Endometriosis": 1,
            "Non_endometriosis_other_pathology": 0,
        },
    }
    all_metrics, all_predictions, all_tuning, cohort_rows = [], [], [], []
    for task_name, group_mapping in tasks.items():
        table = prepare_modeling_table(
            expression, metadata, genes, group_mapping=group_mapping
        )
        metrics, predictions, tuning = run_architecture_benchmark(
            table, genes, settings, task_name
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_tuning.append(tuning)
        cohort_rows.append(
            {
                "task": task_name, "n_patients": len(table),
                "n_positive": int(table["target"].sum()),
                "n_negative": int((1 - table["target"]).sum()),
            }
        )
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    summary = summarize_architectures(metrics)
    paired = paired_differences(metrics)

    reports = (
        args.output_dir / "reports" / "modeling"
        / "architecture_benchmark" / settings.signature_version
    )
    plots = (
        args.output_dir / "eda_plots" / "modeling"
        / "architecture_benchmark" / settings.signature_version
    )
    reports.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(reports / "repeat_level_metrics.csv", index=False)
    predictions.to_csv(reports / "repeated_oof_predictions.csv.gz", index=False)
    tuning.to_csv(reports / "nested_tuning_results.csv", index=False)
    summary.to_csv(reports / "architecture_performance_summary.csv", index=False)
    paired.to_csv(reports / "paired_roc_auc_differences.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(reports / "task_cohorts.csv", index=False)
    with (reports / "architecture_benchmark_settings.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(settings_dict(settings), handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_architecture_benchmark(metrics, plots)

    roc = summary.loc[summary["metric"].eq("roc_auc")].sort_values(
        ["task", "roc_auc_rank"]
    )
    print(
        roc[["task", "architecture", "mean", "percentile_2_5", "percentile_97_5"]]
        .round(3).to_string(index=False)
    )
    print(
        "Claim boundary: internal architecture comparison on a dataset that "
        "contributed to signature evidence; not external validation."
    )
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
