"""Tests for the compact paired architecture benchmark."""

import numpy as np
import pandas as pd

from src.architecture_benchmark import (
    ArchitectureBenchmarkSettings,
    CompactTorchClassifier,
    paired_differences,
    summarize_architectures,
)


def test_compact_torch_architectures_return_probabilities() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 4))
    y = np.array([0] * 15 + [1] * 15)
    X[y == 1, 0] += 2.0
    for architecture in ["mlp", "attention"]:
        model = CompactTorchClassifier(
            architecture=architecture, n_features=4,
            max_epochs=20, patience=4, random_state=2,
        ).fit(X, y)
        probability = model.predict_proba(X)
        assert probability.shape == (30, 2)
        assert np.allclose(probability.sum(axis=1), 1.0)


def test_summary_and_paired_difference_use_complete_repeats() -> None:
    values = {
        "prevalence_only": 0.50, "logistic_l2": 0.70,
        "logistic_elastic_net": 0.72, "linear_svm": 0.69,
        "rbf_svm": 0.68, "random_forest": 0.67,
        "hist_gradient_boosting": 0.66, "small_mlp": 0.65,
        "gene_attention": 0.64,
    }
    rows = []
    for repeat in range(3):
        for architecture, base in values.items():
            value = base + repeat * 0.01
            rows.append(
                {
                    "task": "task", "repeat": repeat,
                    "architecture": architecture, "roc_auc": value,
                    "pr_auc": value, "balanced_accuracy": value,
                    "sensitivity": value, "specificity": value,
                    "brier_score": 1 - value,
                }
            )
    metrics = pd.DataFrame(rows)
    summary = summarize_architectures(metrics)
    elastic_rank = summary.loc[
        summary["architecture"].eq("logistic_elastic_net")
        & summary["metric"].eq("roc_auc"), "roc_auc_rank"
    ].iloc[0]
    assert elastic_rank == 1
    paired = paired_differences(metrics)
    elastic = paired.loc[
        paired["architecture"].eq("logistic_elastic_net")
    ].iloc[0]
    assert elastic["wins"] == 3
    assert np.isclose(elastic["mean_roc_auc_difference"], 0.02)
