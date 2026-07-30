"""Apply the frozen 12-gene reference model to GSE153740."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.external_directional_replication import (
    ExternalReplicationSettings,
    cohort_standardize,
    fit_frozen_logistic,
    gene_direction_results,
    leave_one_gene_model_auc,
    leave_one_sample_auc,
    metric_summary,
    plot_external_replication,
    signed_directional_score,
    stratified_auc_interval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen GSE153740 external replication."
    )
    parser.add_argument(
        "--training-expression",
        type=Path,
        default=Path(
            "output/reports/microarray/GSE51981/gcrma_gene_expression.csv.gz"
        ),
    )
    parser.add_argument(
        "--training-metadata",
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
        "--external-expression",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE153740_intake/"
            "frozen_signature_log2_fpkm.csv.gz"
        ),
    )
    parser.add_argument(
        "--external-metadata",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE153740_intake/"
            "verified_sample_metadata.csv"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/"
            "GSE153740_directional_replication"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(
            "output/eda_plots/external_validation/v1.0/"
            "GSE153740_directional_replication"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ExternalReplicationSettings(bootstrap_seed=153740)
    signature = pd.read_csv(args.signature)
    genes = signature["gene"].tolist()
    directions = signature.set_index("gene")["discovery_direction"].loc[genes]

    training_expression = pd.read_csv(args.training_expression, index_col=0)
    training_metadata = pd.read_csv(args.training_metadata)
    training_metadata = training_metadata.loc[
        training_metadata["clinical_group"].isin(
            ["Endometriosis", "Healthy_control_no_pathology"]
        )
    ]
    training_ids = training_metadata["sample_id"].tolist()
    training_target = training_metadata.set_index("sample_id")["clinical_group"].map(
        {"Healthy_control_no_pathology": 0, "Endometriosis": 1}
    )
    training_standardized = cohort_standardize(
        training_expression.loc[genes, training_ids]
    )

    external = pd.read_csv(args.external_expression, index_col=0).loc[genes]
    metadata = pd.read_csv(args.external_metadata)
    external_ids = metadata["sample_id"].tolist()
    external = external[external_ids]
    external_standardized = cohort_standardize(external)
    target = metadata.set_index("sample_id")["condition"].map(
        {"Disease_free_control": 0, "Endometriosis": 1}
    ).loc[external_ids]

    model = fit_frozen_logistic(training_standardized, training_target, settings)
    probability = pd.Series(
        model.predict_proba(external_standardized.T)[:, 1],
        index=external_ids,
        name="model_probability",
    )
    directional_score = signed_directional_score(
        external_standardized, directions
    ).loc[external_ids]
    model_metrics = metric_summary(target, probability, settings)
    directional_auc = float(roc_auc_score(target, directional_score))
    directional_interval = stratified_auc_interval(
        target,
        directional_score,
        settings.bootstrap_resamples,
        settings.bootstrap_seed + 1,
    )
    gene_results = gene_direction_results(
        external_standardized, target, directions
    )
    patient_sensitivity = leave_one_sample_auc(target, probability)
    gene_sensitivity = leave_one_gene_model_auc(
        training_standardized,
        training_target,
        external_standardized,
        target,
        settings,
    )
    predictions = metadata.set_index("sample_id").loc[
        external_ids,
        ["patient_id", "condition", "cycle_phase", "disease_state"],
    ].copy()
    predictions["target"] = target
    predictions["model_probability"] = probability
    predictions["directional_score"] = directional_score
    predictions = predictions.reset_index()
    patient_sensitivity = patient_sensitivity.merge(
        predictions[["sample_id", "condition", "disease_state"]],
        left_on="removed_sample_id",
        right_on="sample_id",
        how="left",
    ).drop(columns="sample_id")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.report_dir / "external_predictions.csv", index=False)
    gene_results.to_csv(args.report_dir / "gene_direction_replication.csv", index=False)
    patient_sensitivity.to_csv(
        args.report_dir / "leave_one_patient_out_sensitivity.csv", index=False
    )
    gene_sensitivity.to_csv(
        args.report_dir / "leave_one_gene_out_sensitivity.csv", index=False
    )
    pd.DataFrame(
        {
            "gene": genes,
            "training_model_coefficient": model.coef_[0],
            "frozen_discovery_direction": directions.to_numpy(),
        }
    ).to_csv(args.report_dir / "training_model_coefficients.csv", index=False)

    matched = int(gene_results["direction_match"].sum())
    interpretation = (
        "supportive_small_cohort_replication"
        if model_metrics["roc_auc"] > 0.70 and matched >= 8
        else "not_replicated_in_small_cohort"
    )
    payload = {
        "study_id": "GSE153740",
        "analysis_status": "locked_external_application",
        "n_cases": int(target.sum()),
        "n_controls": int((1 - target).sum()),
        "cycle_phase": "Mid_secretory",
        "model": {
            "architecture": "L2 logistic regression",
            "C": settings.logistic_c,
            "class_weight": settings.class_weight,
            "external_tuning": False,
            "features": genes,
        },
        "representation": (
            "within-cohort per-gene z scores; GSE51981 GCRMA and GSE153740 "
            "log2(gene-summed transcript FPKM+1) standardized separately"
        ),
        "model_metrics": model_metrics,
        "directional_score": {
            "roc_auc": directional_auc,
            "roc_auc_ci_lower": directional_interval[0],
            "roc_auc_ci_upper": directional_interval[1],
        },
        "gene_direction_matches": matched,
        "gene_direction_total": len(gene_results),
        "leave_one_patient_auc_min": float(patient_sensitivity["roc_auc"].min()),
        "leave_one_patient_auc_max": float(patient_sensitivity["roc_auc"].max()),
        "leave_one_gene_auc_min": float(gene_sensitivity["roc_auc"].min()),
        "leave_one_gene_auc_max": float(gene_sensitivity["roc_auc"].max()),
        "interpretation": interpretation,
        "claim_boundary": (
            "Directional technical replication only. Four cases and four controls "
            "cannot establish external performance, calibration, specificity, or "
            "clinical utility."
        ),
        "settings": asdict(settings),
    }
    with (args.report_dir / "external_replication_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_external_replication(
        target,
        probability,
        directional_score,
        gene_results,
        args.plot_dir / "external_replication_summary.png",
        "GSE153740",
    )

    print(f"Frozen model ROC-AUC: {model_metrics['roc_auc']:.3f}")
    print(
        f"Bootstrap 95% CI: {model_metrics['roc_auc_ci_lower']:.3f}-"
        f"{model_metrics['roc_auc_ci_upper']:.3f}"
    )
    print(f"Directional score ROC-AUC: {directional_auc:.3f}")
    print(f"Gene-direction matches: {matched}/{len(gene_results)}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
