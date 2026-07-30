"""Patient-level interpretability utilities for the frozen 12-gene model."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class InterpretabilitySettings:
    """Reproducible settings for internal stability and importance estimates."""

    logistic_c: float = 0.01
    class_weight: str = "balanced"
    cv_splits: int = 5
    cv_repeats: int = 20
    permutation_repeats: int = 20
    coefficient_bootstraps: int = 1_000
    seed: int = 51981


def _make_model(settings: InterpretabilitySettings, seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=settings.logistic_c,
        solver="liblinear",
        class_weight=settings.class_weight,
        max_iter=2_000,
        random_state=seed,
    )


def coefficient_bootstrap_stability(
    expression: pd.DataFrame,
    target: pd.Series,
    settings: InterpretabilitySettings,
) -> pd.DataFrame:
    """Estimate coefficient sign and magnitude stability by stratified bootstrap."""

    sample_ids = expression.columns.to_numpy()
    aligned_target = target.loc[sample_ids].astype(int)
    case_ids = aligned_target.index[aligned_target.eq(1)].to_numpy()
    control_ids = aligned_target.index[aligned_target.eq(0)].to_numpy()
    rng = np.random.default_rng(settings.seed)
    coefficients = []
    for iteration in range(settings.coefficient_bootstraps):
        sampled_ids = np.concatenate(
            [
                rng.choice(case_ids, len(case_ids), replace=True),
                rng.choice(control_ids, len(control_ids), replace=True),
            ]
        )
        sampled_x = expression.loc[:, sampled_ids].T
        sampled_y = aligned_target.loc[sampled_ids].to_numpy()
        scaler = StandardScaler()
        standardized = scaler.fit_transform(sampled_x)
        model = _make_model(settings, settings.seed + iteration)
        model.fit(standardized, sampled_y)
        coefficients.append(model.coef_[0])

    matrix = np.asarray(coefficients)
    rows = []
    for position, gene in enumerate(expression.index):
        values = matrix[:, position]
        median = float(np.median(values))
        rows.append(
            {
                "gene": gene,
                "coefficient_median": median,
                "coefficient_ci_lower": float(np.quantile(values, 0.025)),
                "coefficient_ci_upper": float(np.quantile(values, 0.975)),
                "positive_sign_fraction": float(np.mean(values > 0)),
                "sign_stability": float(
                    max(np.mean(values > 0), np.mean(values < 0))
                ),
            }
        )
    return pd.DataFrame(rows)


def repeated_cv_permutation_importance(
    expression: pd.DataFrame,
    target: pd.Series,
    settings: InterpretabilitySettings,
) -> pd.DataFrame:
    """Measure held-out ROC-AUC decrease after permuting each gene."""

    x = expression.T
    y = target.loc[x.index].astype(int)
    splitter = RepeatedStratifiedKFold(
        n_splits=settings.cv_splits,
        n_repeats=settings.cv_repeats,
        random_state=settings.seed,
    )
    rows = []
    for fold, (train_positions, test_positions) in enumerate(splitter.split(x, y)):
        train_x = x.iloc[train_positions]
        test_x = x.iloc[test_positions]
        train_y = y.iloc[train_positions]
        test_y = y.iloc[test_positions]
        scaler = StandardScaler()
        train_standardized = scaler.fit_transform(train_x)
        test_standardized = scaler.transform(test_x)
        model = _make_model(settings, settings.seed + fold)
        model.fit(train_standardized, train_y)
        baseline_auc = roc_auc_score(
            test_y, model.predict_proba(test_standardized)[:, 1]
        )
        importance = permutation_importance(
            model,
            test_standardized,
            test_y,
            scoring="roc_auc",
            n_repeats=settings.permutation_repeats,
            random_state=settings.seed + fold,
        )
        for gene, mean, standard_deviation in zip(
            expression.index, importance.importances_mean, importance.importances_std
        ):
            rows.append(
                {
                    "fold": fold,
                    "gene": gene,
                    "baseline_fold_auc": float(baseline_auc),
                    "auc_decrease": float(mean),
                    "within_fold_permutation_sd": float(standard_deviation),
                }
            )
    fold_results = pd.DataFrame(rows)
    summary = (
        fold_results.groupby("gene", sort=False)
        .agg(
            permutation_auc_decrease_mean=("auc_decrease", "mean"),
            permutation_auc_decrease_sd=("auc_decrease", "std"),
            positive_importance_fraction=("auc_decrease", lambda values: (values > 0).mean()),
        )
        .reset_index()
    )
    return summary


def fit_full_model_and_contributions(
    training_expression: pd.DataFrame,
    training_target: pd.Series,
    cohort_expression: pd.DataFrame,
    settings: InterpretabilitySettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the frozen model and decompose cohort logits into gene contributions."""

    training_scaler = StandardScaler()
    training_standardized = training_scaler.fit_transform(training_expression.T)
    model = _make_model(settings, settings.seed)
    model.fit(
        training_standardized,
        training_target.loc[training_expression.columns].astype(int),
    )

    cohort_values = cohort_expression.T
    cohort_standardized = cohort_values.sub(cohort_values.mean(axis=0), axis=1).div(
        cohort_values.std(axis=0, ddof=0), axis=1
    )
    contributions = cohort_standardized.mul(model.coef_[0], axis=1)
    contribution_table = contributions.stack().rename("logit_contribution").reset_index()
    contribution_table.columns = ["sample_id", "gene", "logit_contribution"]

    coefficients = pd.DataFrame(
        {
            "gene": training_expression.index,
            "full_model_coefficient": model.coef_[0],
            "absolute_coefficient": np.abs(model.coef_[0]),
        }
    )
    return coefficients, contribution_table


def cohort_gene_effects(
    standardized_expression: pd.DataFrame,
    target: pd.Series,
    cohort: str,
) -> pd.DataFrame:
    """Calculate case-control standardized mean differences for one cohort."""

    target = target.loc[standardized_expression.columns].astype(int)
    cases = standardized_expression.loc[:, target.eq(1)]
    controls = standardized_expression.loc[:, target.eq(0)]
    return pd.DataFrame(
        {
            "gene": standardized_expression.index,
            "cohort": cohort,
            "case_control_standardized_mean_difference": (
                cases.mean(axis=1) - controls.mean(axis=1)
            ).to_numpy(),
        }
    )


def plot_interpretability_summary(
    gene_summary: pd.DataFrame,
    cohort_effects: pd.DataFrame,
    output_path,
) -> None:
    """Plot model importance and cross-cohort direction heterogeneity."""

    ordered = gene_summary.sort_values("absolute_coefficient")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].barh(
        ordered["gene"],
        ordered["full_model_coefficient"],
        color=np.where(ordered["full_model_coefficient"] >= 0, "#2a9d8f", "#e76f51"),
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Frozen standardized coefficient")
    axes[0].set_title("Global model direction")

    matrix = cohort_effects.pivot(
        index="gene",
        columns="cohort",
        values="case_control_standardized_mean_difference",
    ).loc[
        gene_summary.sort_values("full_model_coefficient", ascending=False)["gene"],
        ["GSE51981", "GSE212787", "GSE153740"],
    ]
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    image = axes[1].imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[1].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    axes[1].set_yticks(range(len(matrix.index)), matrix.index)
    axes[1].set_title("Case-control direction by cohort")
    figure.colorbar(image, ax=axes[1], label="Standardized mean difference")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
