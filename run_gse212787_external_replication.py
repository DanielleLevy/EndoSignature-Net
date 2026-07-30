"""Apply the frozen 12-gene reference model to GSE212787."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.external_directional_replication import (
    ExternalReplicationSettings,
    cohort_standardize,
    cycle_stratified_concordance,
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
        description="Run frozen cross-platform GSE212787 replication."
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
        "--external-log-cpm",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE212787_intake/"
            "target_log_cpm.csv.gz"
        ),
    )
    parser.add_argument(
        "--external-metadata",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE212787_intake/"
            "verified_sample_metadata.csv"
        ),
    )
    parser.add_argument(
        "--gene-mapping",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE212787_intake/"
            "frozen_gene_ensembl_mapping.csv"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/"
            "GSE212787_directional_replication"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(
            "output/eda_plots/external_validation/v1.0/"
            "GSE212787_directional_replication"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ExternalReplicationSettings()
    signature = pd.read_csv(args.signature)
    genes = signature["gene"].tolist()
    directions = signature.set_index("gene")["discovery_direction"].loc[genes]

    training_expression = pd.read_csv(args.training_expression, index_col=0)
    training_metadata = pd.read_csv(args.training_metadata)
    training_metadata = training_metadata.loc[
        training_metadata["clinical_group"].isin(
            ["Endometriosis", "Healthy_control_no_pathology"]
        )
    ].copy()
    training_ids = training_metadata["sample_id"].tolist()
    training_target = training_metadata.set_index("sample_id")["clinical_group"].map(
        {"Healthy_control_no_pathology": 0, "Endometriosis": 1}
    )
    training = training_expression.loc[genes, training_ids]
    training_standardized = cohort_standardize(training)

    external_all = pd.read_csv(args.external_log_cpm, index_col=0)
    external_metadata = pd.read_csv(args.external_metadata)
    external_metadata = external_metadata.loc[
        external_metadata["include_in_external_target"].astype(bool)
    ].copy()
    external_ids = external_metadata["sample_id"].tolist()
    mapping = pd.read_csv(args.gene_mapping).set_index("gene")
    external = pd.DataFrame(
        {
            gene: external_all.loc[mapping.loc[gene, "ensembl_gene_id"], external_ids]
            for gene in genes
        }
    ).T
    external.index.name = "gene"
    external_standardized = cohort_standardize(external)
    external_target = external_metadata.set_index("sample_id")["condition"].map(
        {"Disease_free_control": 0, "Endometriosis": 1}
    ).loc[external_ids]

    model = fit_frozen_logistic(
        training_standardized, training_target, settings
    )
    probability = pd.Series(
        model.predict_proba(external_standardized.T)[:, 1],
        index=external_ids,
        name="model_probability",
    )
    directional_score = signed_directional_score(
        external_standardized, directions
    ).loc[external_ids]
    gene_results = gene_direction_results(
        external_standardized, external_target, directions
    )
    model_metrics = metric_summary(external_target, probability, settings)
    direction_auc = float(roc_auc_score(external_target, directional_score))
    direction_interval = stratified_auc_interval(
        external_target,
        directional_score,
        settings.bootstrap_resamples,
        settings.bootstrap_seed + 1,
    )

    predictions = external_metadata.set_index("sample_id").loc[external_ids, [
        "patient_id",
        "condition",
        "cycle_phase",
        "matrix_sample_alias",
    ]].copy()
    predictions["target"] = external_target
    predictions["model_probability"] = probability
    predictions["directional_score"] = directional_score
    predictions["model_prediction_at_0_5"] = probability.ge(settings.threshold).astype(int)
    predictions = predictions.reset_index()

    sample_sensitivity = leave_one_sample_auc(external_target, probability)
    sample_sensitivity = sample_sensitivity.merge(
        predictions[["sample_id", "condition", "cycle_phase"]],
        left_on="removed_sample_id",
        right_on="sample_id",
        how="left",
    ).drop(columns="sample_id")
    gene_sensitivity = leave_one_gene_model_auc(
        training_standardized,
        training_target,
        external_standardized,
        external_target,
        settings,
    )

    proliferative_ids = predictions.loc[
        predictions["cycle_phase"].eq("Proliferative"), "sample_id"
    ]
    proliferative_target = external_target.loc[proliferative_ids]
    proliferative_auc = float(
        roc_auc_score(proliferative_target, probability.loc[proliferative_ids])
    )
    flagged_removed = external_target.index != "GSM6552374"
    flagged_removed_auc = float(
        roc_auc_score(
            external_target.loc[flagged_removed],
            probability.loc[flagged_removed],
        )
    )
    score_by_cycle = (
        predictions.groupby("cycle_phase", observed=True)
        .agg(
            n=("sample_id", "size"),
            mean_model_probability=("model_probability", "mean"),
            mean_directional_score=("directional_score", "mean"),
        )
        .reset_index()
    )
    cycle_series = predictions.set_index("sample_id")["cycle_phase"].loc[external_ids]
    model_cycle_concordance, model_cycle_details = cycle_stratified_concordance(
        external_target, probability, cycle_series
    )
    direction_cycle_concordance, direction_cycle_details = (
        cycle_stratified_concordance(
            external_target, directional_score, cycle_series
        )
    )
    cycle_concordance = model_cycle_details.rename(
        columns={"within_phase_concordance": "model_concordance"}
    ).merge(
        direction_cycle_details.rename(
            columns={"within_phase_concordance": "directional_concordance"}
        ),
        on=["cycle_phase", "case_control_pairs"],
        how="outer",
        validate="one_to_one",
    )

    coefficients = pd.DataFrame(
        {
            "gene": genes,
            "training_model_coefficient": model.coef_[0],
            "frozen_discovery_direction": directions.loc[genes].to_numpy(),
        }
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.report_dir / "external_predictions.csv", index=False)
    gene_results.to_csv(args.report_dir / "gene_direction_replication.csv", index=False)
    coefficients.to_csv(args.report_dir / "training_model_coefficients.csv", index=False)
    sample_sensitivity.to_csv(
        args.report_dir / "leave_one_patient_out_sensitivity.csv", index=False
    )
    gene_sensitivity.to_csv(
        args.report_dir / "leave_one_gene_out_sensitivity.csv", index=False
    )
    score_by_cycle.to_csv(args.report_dir / "score_by_cycle_phase.csv", index=False)
    cycle_concordance.to_csv(
        args.report_dir / "within_cycle_concordance.csv", index=False
    )

    matched = int(gene_results["direction_match"].sum())
    interpretation = (
        "supportive_external_replication"
        if model_metrics["roc_auc"] > 0.70
        and model_metrics["roc_auc_ci_lower"] > 0.50
        and matched >= 8
        else "inconclusive_external_replication"
    )
    payload = {
        "study_id": "GSE212787",
        "analysis_status": "first_locked_external_application",
        "model": {
            "architecture": "L2 logistic regression",
            "C": settings.logistic_c,
            "C_source": (
                "modal inner-CV choice from the completed GSE51981 architecture "
                "benchmark: C=0.01 in 29 of 50 outer folds"
            ),
            "class_weight": settings.class_weight,
            "features": genes,
            "external_tuning": False,
        },
        "representation": (
            "within-cohort per-gene z scores; GSE51981 GCRMA and GSE212787 "
            "log2(CPM+1) standardized separately without outcome labels"
        ),
        "cohort_level_transductive_standardization": True,
        "model_metrics": model_metrics,
        "directional_score": {
            "roc_auc": direction_auc,
            "roc_auc_ci_lower": direction_interval[0],
            "roc_auc_ci_upper": direction_interval[1],
        },
        "gene_direction_matches": matched,
        "gene_direction_total": len(gene_results),
        "proliferative_only_roc_auc": proliferative_auc,
        "cycle_stratified_model_concordance": model_cycle_concordance,
        "cycle_stratified_directional_concordance": direction_cycle_concordance,
        "roc_auc_without_GSM6552374": flagged_removed_auc,
        "leave_one_patient_auc_min": float(sample_sensitivity["roc_auc"].min()),
        "leave_one_patient_auc_max": float(sample_sensitivity["roc_auc"].max()),
        "leave_one_gene_auc_min": float(gene_sensitivity["roc_auc"].min()),
        "leave_one_gene_auc_max": float(gene_sensitivity["roc_auc"].max()),
        "interpretation": interpretation,
        "claim_boundary": (
            "Supporting cohort-level cross-platform replication only; n=13, "
            "mixed cycle phase, no other-pathology controls, and external "
            "cohort standardization prevent clinical or single-sample claims."
        ),
        "settings": asdict(settings),
    }
    with (args.report_dir / "external_replication_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    plot_external_replication(
        external_target,
        probability,
        directional_score,
        gene_results,
        args.plot_dir / "external_replication_summary.png",
        "GSE212787",
    )
    print(f"Frozen model ROC-AUC: {model_metrics['roc_auc']:.3f}")
    print(
        "Bootstrap 95% CI: "
        f"{model_metrics['roc_auc_ci_lower']:.3f}-"
        f"{model_metrics['roc_auc_ci_upper']:.3f}"
    )
    print(f"Directional score ROC-AUC: {direction_auc:.3f}")
    print(f"Gene-direction matches: {matched}/{len(gene_results)}")
    print(f"Proliferative-only ROC-AUC: {proliferative_auc:.3f}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
