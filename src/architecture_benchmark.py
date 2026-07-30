"""Paired nested-CV benchmark of compact models for a frozen gene signature."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ArchitectureBenchmarkSettings:
    """Predeclared shared evaluation policy for every architecture."""

    signature_version: str = "v1.0"
    outer_folds: int = 5
    outer_repeats: int = 10
    inner_folds: int = 3
    random_seed: int = 271828
    classification_threshold: float = 0.5
    scoring: str = "roc_auc"
    class_weight: str = "balanced"
    neural_max_epochs: int = 250
    neural_patience: int = 20


ARCHITECTURE_ORDER = [
    "prevalence_only",
    "logistic_l2",
    "logistic_elastic_net",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "hist_gradient_boosting",
    "small_mlp",
    "gene_attention",
]


class CompactTorchClassifier(ClassifierMixin, BaseEstimator):
    """Small sklearn-compatible PyTorch classifier with internal early stopping."""

    def __init__(
        self,
        architecture: str = "mlp",
        n_features: int = 12,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        dropout: float = 0.2,
        max_epochs: int = 250,
        patience: int = 20,
        validation_fraction: float = 0.2,
        random_state: int = 0,
    ):
        self.architecture = architecture
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _build_model(self) -> torch.nn.Module:
        if self.architecture == "mlp":
            return torch.nn.Sequential(
                torch.nn.Linear(self.n_features, 8),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(8, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1),
            )
        if self.architecture == "attention":
            return _GeneAttentionNetwork(self.n_features, self.dropout)
        raise ValueError(f"Unknown neural architecture: {self.architecture}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        train_x, validation_x, train_y, validation_y = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        self.model_ = self._build_model()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        negatives = max(float((train_y == 0).sum()), 1.0)
        positives = max(float((train_y == 1).sum()), 1.0)
        loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
        )
        train_x_tensor = torch.tensor(train_x)
        train_y_tensor = torch.tensor(train_y).reshape(-1, 1)
        validation_x_tensor = torch.tensor(validation_x)
        validation_y_tensor = torch.tensor(validation_y).reshape(-1, 1)
        best_loss = np.inf
        best_state = None
        stale_epochs = 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            optimizer.zero_grad()
            loss = loss_function(self.model_(train_x_tensor), train_y_tensor)
            loss.backward()
            optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                validation_loss = loss_function(
                    self.model_(validation_x_tensor), validation_y_tensor
                ).item()
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = {
                    name: value.detach().clone()
                    for name, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_epochs_ = epoch + 1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(np.asarray(X, dtype=np.float32)))
            positive = torch.sigmoid(logits).numpy().reshape(-1)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _GeneAttentionNetwork(torch.nn.Module):
    """Feature-attention network; attention is not assumed to equal causality."""

    def __init__(self, n_features: int, dropout: float):
        super().__init__()
        self.attention_bias = torch.nn.Parameter(torch.zeros(n_features))
        self.attention_scale = torch.nn.Parameter(torch.ones(n_features))
        self.head = torch.nn.Sequential(
            torch.nn.Linear(n_features, 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(8, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(
            self.attention_bias + values * self.attention_scale, dim=1
        )
        return self.head(values * attention)


def _scaled_pipeline(estimator) -> Pipeline:
    """Nest imputation and scaling inside every training fit."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def architecture_search_spaces(
    n_features: int,
    settings: ArchitectureBenchmarkSettings,
) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Return compact search spaces suitable for approximately 100 patients."""

    seed = settings.random_seed
    neural_common = {
        "n_features": n_features,
        "max_epochs": settings.neural_max_epochs,
        "patience": settings.neural_patience,
        "random_state": seed,
    }
    return {
        "logistic_l2": (
            _scaled_pipeline(
                LogisticRegression(
                    penalty="l2", solver="liblinear",
                    class_weight=settings.class_weight, max_iter=2000,
                    random_state=seed,
                )
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "logistic_elastic_net": (
            _scaled_pipeline(
                LogisticRegression(
                    penalty="elasticnet", solver="saga",
                    class_weight=settings.class_weight, max_iter=5000,
                    random_state=seed,
                )
            ),
            {
                "model__C": [0.01, 0.1, 1.0],
                "model__l1_ratio": [0.25, 0.5, 0.75],
            },
        ),
        "linear_svm": (
            _scaled_pipeline(
                SVC(
                    kernel="linear", probability=True,
                    class_weight=settings.class_weight, random_state=seed,
                )
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "rbf_svm": (
            _scaled_pipeline(
                SVC(
                    kernel="rbf", probability=True,
                    class_weight=settings.class_weight, random_state=seed,
                )
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": [0.01, 0.1, 1.0],
            },
        ),
        "random_forest": (
            _scaled_pipeline(
                RandomForestClassifier(
                    n_estimators=300, class_weight=settings.class_weight,
                    random_state=seed, n_jobs=1,
                )
            ),
            {
                "model__max_depth": [2, 4, None],
                "model__min_samples_leaf": [2, 5, 10],
            },
        ),
        "hist_gradient_boosting": (
            _scaled_pipeline(
                HistGradientBoostingClassifier(
                    class_weight=settings.class_weight, max_iter=150,
                    early_stopping=True, random_state=seed,
                )
            ),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_leaf_nodes": [3, 7],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
        "small_mlp": (
            _scaled_pipeline(
                CompactTorchClassifier(architecture="mlp", **neural_common)
            ),
            {
                "model__dropout": [0.1, 0.3],
                "model__weight_decay": [0.001, 0.01],
            },
        ),
        "gene_attention": (
            _scaled_pipeline(
                CompactTorchClassifier(architecture="attention", **neural_common)
            ),
            {
                "model__dropout": [0.1, 0.3],
                "model__weight_decay": [0.001, 0.01],
            },
        ),
    }


def _metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
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


def run_architecture_benchmark(
    table: pd.DataFrame,
    genes: list[str],
    settings: ArchitectureBenchmarkSettings,
    task_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use identical outer splits and nested tuning for every architecture."""

    X = table[genes].to_numpy(dtype=float)
    y = table["target"].to_numpy()
    spaces = architecture_search_spaces(len(genes), settings)
    metric_rows, prediction_rows, tuning_rows = [], [], []
    for repeat in range(settings.outer_repeats):
        outer = StratifiedKFold(
            n_splits=settings.outer_folds, shuffle=True,
            random_state=settings.random_seed + repeat,
        )
        probabilities = {
            architecture: np.full(len(table), np.nan)
            for architecture in ARCHITECTURE_ORDER
        }
        for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
            probabilities["prevalence_only"][test_index] = y[train_index].mean()
            for architecture in ARCHITECTURE_ORDER[1:]:
                estimator, grid = spaces[architecture]
                inner = StratifiedKFold(
                    n_splits=settings.inner_folds, shuffle=True,
                    random_state=settings.random_seed + repeat * 1000 + fold,
                )
                started = time.perf_counter()
                search = GridSearchCV(
                    estimator, grid, scoring=settings.scoring, cv=inner,
                    refit=True, n_jobs=1, error_score="raise",
                )
                search.fit(X[train_index], y[train_index])
                probabilities[architecture][test_index] = search.predict_proba(
                    X[test_index]
                )[:, 1]
                tuning_rows.append(
                    {
                        "task": task_name, "repeat": repeat, "fold": fold,
                        "architecture": architecture,
                        "best_parameters": repr(search.best_params_),
                        "inner_cv_roc_auc": search.best_score_,
                        "fit_seconds": time.perf_counter() - started,
                    }
                )
        for architecture, probability in probabilities.items():
            if np.isnan(probability).any():
                raise RuntimeError(f"Incomplete predictions for {architecture}")
            metric_rows.append(
                {
                    "task": task_name, "repeat": repeat,
                    "architecture": architecture,
                    **_metrics(y, probability, settings.classification_threshold),
                }
            )
            for index, value in enumerate(probability):
                prediction_rows.append(
                    {
                        "task": task_name, "repeat": repeat,
                        "architecture": architecture,
                        "sample_id": table.iloc[index]["sample_id"],
                        "patient_id": table.iloc[index]["patient_id"],
                        "target": y[index], "probability": value,
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
    )


def summarize_architectures(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize complete-repeat metrics and rank by mean ROC-AUC."""

    metric_names = [
        "roc_auc", "pr_auc", "balanced_accuracy",
        "sensitivity", "specificity", "brier_score",
    ]
    rows = []
    for (task, architecture), group in metrics.groupby(
        ["task", "architecture"], sort=False
    ):
        for metric in metric_names:
            values = group[metric]
            rows.append(
                {
                    "task": task, "architecture": architecture, "metric": metric,
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "percentile_2_5": values.quantile(0.025),
                    "percentile_97_5": values.quantile(0.975),
                    "n_repeats": len(values),
                }
            )
    summary = pd.DataFrame(rows)
    roc = summary.loc[summary["metric"].eq("roc_auc")].copy()
    roc["roc_auc_rank"] = roc.groupby("task")["mean"].rank(
        ascending=False, method="min"
    )
    return summary.merge(
        roc[["task", "architecture", "roc_auc_rank"]],
        on=["task", "architecture"], how="left",
    )


def paired_differences(
    metrics: pd.DataFrame, reference: str = "logistic_l2"
) -> pd.DataFrame:
    """Compute paired repeat-level ROC-AUC differences from the reference."""

    rows = []
    for task, task_metrics in metrics.groupby("task"):
        wide = task_metrics.pivot(
            index="repeat", columns="architecture", values="roc_auc"
        )
        for architecture in ARCHITECTURE_ORDER:
            difference = wide[architecture] - wide[reference]
            rows.append(
                {
                    "task": task, "architecture": architecture,
                    "reference": reference,
                    "mean_roc_auc_difference": difference.mean(),
                    "percentile_2_5": difference.quantile(0.025),
                    "percentile_97_5": difference.quantile(0.975),
                    "wins": int((difference > 0).sum()),
                    "ties": int((difference == 0).sum()),
                    "n_repeats": len(difference),
                }
            )
    return pd.DataFrame(rows)


def plot_architecture_benchmark(
    metrics: pd.DataFrame, output_dir: Path
) -> None:
    """Visualize paired distributions without declaring a winner post hoc."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    labels = {
        "prevalence_only": "Prevalence", "logistic_l2": "Logistic L2",
        "logistic_elastic_net": "Elastic net", "linear_svm": "Linear SVM",
        "rbf_svm": "RBF SVM", "random_forest": "Random forest",
        "hist_gradient_boosting": "Hist boosting", "small_mlp": "Small MLP",
        "gene_attention": "Gene attention",
    }
    plot_data = metrics.copy()
    plot_data["architecture_label"] = plot_data["architecture"].map(labels)
    for task in plot_data["task"].unique():
        task_data = plot_data.loc[plot_data["task"].eq(task)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for axis, metric in zip(
            axes, ["roc_auc", "pr_auc", "balanced_accuracy"]
        ):
            sns.boxplot(
                data=task_data, x="architecture_label", y=metric,
                order=[labels[item] for item in ARCHITECTURE_ORDER],
                ax=axis, color="#4C78A8",
            )
            axis.set_ylim(0, 1)
            axis.set_xlabel("")
            axis.set_title(metric.replace("_", " ").upper())
            axis.tick_params(axis="x", rotation=45)
        fig.suptitle(f"Architecture benchmark: {task}")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{task}_architecture_metric_distributions.png", dpi=180
        )
        plt.close(fig)


def settings_dict(settings: ArchitectureBenchmarkSettings) -> dict:
    return asdict(settings)
