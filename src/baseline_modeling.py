"""Leakage-controlled exploratory baselines for a frozen molecular signature."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class BaselineSettings:
    """Predeclared settings for the exploratory GSE51981 benchmark."""

    signature_version: str = "v1.0"
    outer_folds: int = 5
    outer_repeats: int = 20
    inner_folds: int = 4
    random_seed: int = 51981
    classification_threshold: float = 0.5
    class_weight: str = "balanced"
    regularization_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    primary_target: str = "endometriosis_vs_clean_control"
    evaluation_status: str = (
        "internal_exploratory_cv_signature_selected_with_GSE51981"
    )


MODEL_ORDER = [
    "prevalence_only",
    "cycle_only",
    "pdzd2_only",
    "frozen_signature_12",
    "frozen_signature_12_plus_cycle",
]


def prepare_modeling_table(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    signature_genes: list[str],
    group_mapping: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Create one modeling row per unique patient for a predeclared contrast."""

    required_metadata = {
        "sample_id",
        "patient_id",
        "clinical_group",
        "cycle_phase",
    }
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_metadata:
        raise ValueError(f"Missing metadata columns: {sorted(missing_metadata)}")
    if "gene" not in expression.columns:
        raise ValueError("Expression matrix must contain a gene column")
    if metadata["patient_id"].duplicated().any():
        raise ValueError("Patient-level modeling requires unique patient IDs")

    groups = group_mapping or {
        "Endometriosis": 1,
        "Healthy_control_no_pathology": 0,
    }
    cohort = metadata.loc[metadata["clinical_group"].isin(groups)].copy()
    cohort["target"] = cohort["clinical_group"].map(groups).astype(int)
    missing_genes = sorted(set(signature_genes) - set(expression["gene"]))
    if missing_genes:
        raise ValueError(f"Frozen signature genes unavailable: {missing_genes}")
    missing_samples = sorted(set(cohort["sample_id"]) - set(expression.columns))
    if missing_samples:
        raise ValueError(f"Expression unavailable for samples: {missing_samples}")

    matrix = (
        expression.set_index("gene")
        .loc[signature_genes, cohort["sample_id"]]
        .T.rename_axis("sample_id")
        .reset_index()
    )
    table = cohort[
        ["sample_id", "patient_id", "clinical_group", "cycle_phase", "target"]
    ].merge(matrix, on="sample_id", how="inner", validate="one_to_one")
    table["cycle_phase"] = table["cycle_phase"].replace("Unknown", np.nan)
    if table[signature_genes].isna().any().any():
        raise ValueError("Frozen signature expression contains missing values")
    return table


def _logistic_pipeline(
    numeric_features: list[str],
    include_cycle: bool,
    settings: BaselineSettings,
) -> Pipeline:
    """Build a pipeline whose preprocessing is fitted only on training folds."""

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "genes",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if include_cycle:
        transformers.append(
            (
                "cycle",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                ["cycle_phase"],
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight=settings.class_weight,
                    max_iter=2000,
                    random_state=settings.random_seed,
                ),
            ),
        ]
    )


def _metric_row(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute discrimination and threshold metrics for one complete CV repeat."""

    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "brier_score": brier_score_loss(y_true, probability),
    }


def run_repeated_nested_cv(
    table: pd.DataFrame,
    signature_genes: list[str],
    settings: BaselineSettings,
    additional_gene_models: dict[str, list[str]] | None = None,
    model_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen models with repeated outer CV and nested tuning."""

    y = table["target"].to_numpy()
    model_specs = {
        "cycle_only": ([], True),
        "pdzd2_only": (["PDZD2"], False),
        "frozen_signature_12": (signature_genes, False),
        "frozen_signature_12_plus_cycle": (signature_genes, True),
    }
    for name, genes in (additional_gene_models or {}).items():
        model_specs[name] = (genes, False)
    active_model_order = model_order or MODEL_ORDER
    unknown_models = set(active_model_order) - {
        "prevalence_only",
        *model_specs,
    }
    if unknown_models:
        raise ValueError(f"Unknown requested models: {sorted(unknown_models)}")
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    tuning_rows: list[dict] = []

    for repeat in range(settings.outer_repeats):
        outer = StratifiedKFold(
            n_splits=settings.outer_folds,
            shuffle=True,
            random_state=settings.random_seed + repeat,
        )
        repeat_probabilities = {
            model: np.full(len(table), np.nan) for model in active_model_order
        }
        for fold, (train_index, test_index) in enumerate(outer.split(table, y)):
            y_train = y[train_index]
            prevalence = float(y_train.mean())
            repeat_probabilities["prevalence_only"][test_index] = prevalence
            for model_name in active_model_order:
                if model_name == "prevalence_only":
                    continue
                genes, include_cycle = model_specs[model_name]
                pipeline = _logistic_pipeline(genes, include_cycle, settings)
                inner = StratifiedKFold(
                    n_splits=settings.inner_folds,
                    shuffle=True,
                    random_state=settings.random_seed + 1000 * repeat + fold,
                )
                search = GridSearchCV(
                    pipeline,
                    {"classifier__C": list(settings.regularization_grid)},
                    scoring="roc_auc",
                    cv=inner,
                    n_jobs=1,
                    refit=True,
                )
                search.fit(table.iloc[train_index], y_train)
                probability = search.predict_proba(table.iloc[test_index])[:, 1]
                repeat_probabilities[model_name][test_index] = probability
                tuning_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "model": model_name,
                        "best_C": search.best_params_["classifier__C"],
                        "inner_cv_roc_auc": search.best_score_,
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                    }
                )

        for model_name, probability in repeat_probabilities.items():
            if np.isnan(probability).any():
                raise RuntimeError(f"Incomplete outer predictions for {model_name}")
            metrics = _metric_row(
                y, probability, settings.classification_threshold
            )
            metric_rows.append(
                {"repeat": repeat, "model": model_name, **metrics}
            )
            for row_index, prob in enumerate(probability):
                prediction_rows.append(
                    {
                        "repeat": repeat,
                        "model": model_name,
                        "sample_id": table.iloc[row_index]["sample_id"],
                        "patient_id": table.iloc[row_index]["patient_id"],
                        "target": y[row_index],
                        "probability": prob,
                        "prediction": int(
                            prob >= settings.classification_threshold
                        ),
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
    )


def summarize_repeated_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize complete-repeat metrics without treating outer folds as independent."""

    rows = []
    metric_names = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "brier_score",
    ]
    for model_name in metrics["model"].drop_duplicates():
        group = metrics.loc[metrics["model"].eq(model_name)]
        for metric in metric_names:
            values = group[metric].dropna()
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "median": values.median(),
                    "percentile_2_5": values.quantile(0.025),
                    "percentile_97_5": values.quantile(0.975),
                    "n_repeats": len(values),
                }
            )
    return pd.DataFrame(rows)


def plot_baseline_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
    model_order: list[str] | None = None,
    model_labels: dict[str, str] | None = None,
    title: str = "Repeated nested CV: internal exploratory performance",
) -> None:
    """Write compact discrimination plots for the exploratory benchmark."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    labels = {
        "prevalence_only": "Prevalence",
        "cycle_only": "Cycle only",
        "pdzd2_only": "PDZD2",
        "frozen_signature_12": "12 genes",
        "frozen_signature_12_plus_cycle": "12 genes + cycle",
    }
    labels.update(model_labels or {})
    active_model_order = model_order or MODEL_ORDER
    plot_metrics = metrics.loc[
        metrics["model"].isin(active_model_order),
        ["model", "roc_auc", "pr_auc", "balanced_accuracy"],
    ].melt(id_vars="model", var_name="metric", value_name="score")
    plot_metrics["model"] = plot_metrics["model"].map(labels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis, metric in zip(
        axes, ["roc_auc", "pr_auc", "balanced_accuracy"]
    ):
        sns.boxplot(
            data=plot_metrics.loc[plot_metrics["metric"].eq(metric)],
            x="model",
            y="score",
            ax=axis,
            color="#4C78A8",
        )
        axis.set_title(metric.replace("_", " ").upper())
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "repeated_cv_metric_distributions.png", dpi=180)
    plt.close(fig)

    averaged = (
        predictions.groupby(["model", "sample_id", "target"], as_index=False)[
            "probability"
        ].mean()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name in active_model_order:
        group = averaged.loc[averaged["model"].eq(model_name)]
        fpr, tpr, _ = roc_curve(group["target"], group["probability"])
        precision, recall, _ = precision_recall_curve(
            group["target"], group["probability"]
        )
        axes[0].plot(fpr, tpr, label=labels[model_name])
        axes[1].plot(recall, precision, label=labels[model_name])
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    fig.suptitle(f"{title}: mean repeated out-of-fold predictions")
    fig.tight_layout()
    fig.savefig(output_dir / "mean_oof_discrimination_curves.png", dpi=180)
    plt.close(fig)


def settings_dict(settings: BaselineSettings) -> dict:
    """Return JSON-safe settings with immutable tuples converted to lists."""

    result = asdict(settings)
    result["regularization_grid"] = list(settings.regularization_grid)
    return result
