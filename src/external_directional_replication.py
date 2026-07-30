"""Frozen cross-platform replication utilities for independent RNA-seq cohorts."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ExternalReplicationSettings:
    """Locked model and uncertainty settings."""

    logistic_c: float = 0.01
    class_weight: str = "balanced"
    threshold: float = 0.5
    bootstrap_resamples: int = 2_000
    bootstrap_seed: int = 212787


def cohort_standardize(expression: pd.DataFrame) -> pd.DataFrame:
    """Standardize each gene within one cohort without phenotype labels."""

    means = expression.mean(axis=1)
    standard_deviations = expression.std(axis=1, ddof=0)
    if standard_deviations.le(0).any():
        failed = standard_deviations.index[standard_deviations.le(0)].tolist()
        raise ValueError(f"Genes have zero cohort variance: {failed}")
    return expression.sub(means, axis=0).div(standard_deviations, axis=0)


def fit_frozen_logistic(
    training_expression: pd.DataFrame,
    training_target: pd.Series,
    settings: ExternalReplicationSettings,
) -> LogisticRegression:
    """Fit the locked architecture once on the complete training cohort."""

    model = LogisticRegression(
        penalty="l2",
        C=settings.logistic_c,
        solver="liblinear",
        class_weight=settings.class_weight,
        max_iter=2_000,
        random_state=settings.bootstrap_seed,
    )
    model.fit(training_expression.T, training_target.loc[training_expression.columns])
    return model


def stratified_auc_interval(
    target: pd.Series,
    score: pd.Series,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Calculate a patient-level stratified bootstrap ROC-AUC interval."""

    target = target.astype(int)
    case_ids = target.index[target.eq(1)].to_numpy()
    control_ids = target.index[target.eq(0)].to_numpy()
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = np.concatenate(
            [
                rng.choice(case_ids, len(case_ids), replace=True),
                rng.choice(control_ids, len(control_ids), replace=True),
            ]
        )
        sampled_target = target.loc[sampled].to_numpy()
        sampled_score = score.loc[sampled].to_numpy()
        estimates.append(roc_auc_score(sampled_target, sampled_score))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def metric_summary(
    target: pd.Series,
    probability: pd.Series,
    settings: ExternalReplicationSettings,
) -> dict[str, float]:
    """Calculate discrimination, threshold metrics, and bootstrap uncertainty."""

    prediction = probability.ge(settings.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    lower, upper = stratified_auc_interval(
        target,
        probability,
        settings.bootstrap_resamples,
        settings.bootstrap_seed,
    )
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "roc_auc_ci_lower": lower,
        "roc_auc_ci_upper": upper,
        "average_precision": float(average_precision_score(target, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
    }


def gene_direction_results(
    standardized_external: pd.DataFrame,
    target: pd.Series,
    discovery_directions: pd.Series,
) -> pd.DataFrame:
    """Compare external standardized mean differences with frozen directions."""

    rows: list[dict[str, object]] = []
    for gene in standardized_external.index:
        cases = standardized_external.loc[gene, target.index[target.eq(1)]]
        controls = standardized_external.loc[gene, target.index[target.eq(0)]]
        effect = float(cases.mean() - controls.mean())
        expected = 1 if discovery_directions.loc[gene] == "higher_in_endometriosis" else -1
        rows.append(
            {
                "gene": gene,
                "frozen_direction": discovery_directions.loc[gene],
                "external_standardized_mean_difference": effect,
                "external_direction": (
                    "higher_in_endometriosis" if effect > 0 else "lower_in_endometriosis"
                ),
                "direction_match": bool(np.sign(effect) == expected),
            }
        )
    return pd.DataFrame(rows)


def signed_directional_score(
    standardized_external: pd.DataFrame,
    discovery_directions: pd.Series,
) -> pd.Series:
    """Average signed standardized expression using frozen biological directions."""

    signs = discovery_directions.map(
        {"higher_in_endometriosis": 1.0, "lower_in_endometriosis": -1.0}
    )
    if signs.isna().any():
        raise ValueError("Unrecognized frozen discovery direction")
    return standardized_external.mul(signs, axis=0).mean(axis=0).rename(
        "directional_score"
    )


def leave_one_sample_auc(
    target: pd.Series, score: pd.Series
) -> pd.DataFrame:
    """Measure external AUC after removing each patient once."""

    rows = []
    for sample_id in target.index:
        keep = target.index != sample_id
        rows.append(
            {
                "removed_sample_id": sample_id,
                "roc_auc": float(roc_auc_score(target.loc[keep], score.loc[keep])),
            }
        )
    return pd.DataFrame(rows)


def cycle_stratified_concordance(
    target: pd.Series,
    score: pd.Series,
    cycle_phase: pd.Series,
) -> tuple[float, pd.DataFrame]:
    """Measure case-control concordance using only pairs within cycle phase."""

    comparisons: list[float] = []
    rows: list[dict[str, object]] = []
    for phase in sorted(cycle_phase.dropna().unique()):
        ids = cycle_phase.index[cycle_phase.eq(phase)]
        case_scores = score.loc[ids[target.loc[ids].eq(1)]]
        control_scores = score.loc[ids[target.loc[ids].eq(0)]]
        phase_comparisons = [
            float(case > control) + 0.5 * float(case == control)
            for case in case_scores
            for control in control_scores
        ]
        if not phase_comparisons:
            continue
        comparisons.extend(phase_comparisons)
        rows.append(
            {
                "cycle_phase": phase,
                "case_control_pairs": len(phase_comparisons),
                "within_phase_concordance": float(np.mean(phase_comparisons)),
            }
        )
    if not comparisons:
        raise ValueError("No within-cycle case-control pairs are available")
    return float(np.mean(comparisons)), pd.DataFrame(rows)


def leave_one_gene_model_auc(
    training_standardized: pd.DataFrame,
    training_target: pd.Series,
    external_standardized: pd.DataFrame,
    external_target: pd.Series,
    settings: ExternalReplicationSettings,
) -> pd.DataFrame:
    """Refit the locked architecture after removing each gene as sensitivity."""

    rows = []
    for removed_gene in training_standardized.index:
        genes = training_standardized.index.drop(removed_gene)
        model = fit_frozen_logistic(
            training_standardized.loc[genes], training_target, settings
        )
        probability = pd.Series(
            model.predict_proba(external_standardized.loc[genes].T)[:, 1],
            index=external_target.index,
        )
        rows.append(
            {
                "removed_gene": removed_gene,
                "roc_auc": float(roc_auc_score(external_target, probability)),
            }
        )
    return pd.DataFrame(rows)


def plot_external_replication(
    target: pd.Series,
    model_probability: pd.Series,
    directional_score: pd.Series,
    gene_results: pd.DataFrame,
    output_path,
    study_id: str,
) -> None:
    """Create a compact external-replication figure."""

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for score, label in [
        (model_probability, "Frozen L2 model"),
        (directional_score, "Frozen directional score"),
    ]:
        false_positive, true_positive, _ = roc_curve(target, score)
        auc = roc_auc_score(target, score)
        axes[0].plot(false_positive, true_positive, label=f"{label} (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0].legend(frameon=False)
    axes[0].set_title(f"{study_id} external discrimination")

    ordered = gene_results.sort_values("external_standardized_mean_difference")
    colors = ordered["direction_match"].map({True: "#2a9d8f", False: "#e76f51"})
    axes[1].barh(
        ordered["gene"],
        ordered["external_standardized_mean_difference"],
        color=colors,
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Case − control standardized mean")
    axes[1].set_title("Frozen gene-direction replication")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
