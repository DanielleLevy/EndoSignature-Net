"""Tests for frozen-model interpretability utilities."""

import pandas as pd

from src.model_interpretability import (
    InterpretabilitySettings,
    coefficient_bootstrap_stability,
    fit_full_model_and_contributions,
)


def _toy_data():
    expression = pd.DataFrame(
        {
            "c1": [0.0, 4.0],
            "c2": [0.5, 3.5],
            "c3": [0.2, 4.2],
            "e1": [4.0, 0.0],
            "e2": [3.5, 0.5],
            "e3": [4.2, 0.2],
        },
        index=["UP", "DOWN"],
    )
    target = pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=expression.columns,
    )
    return expression, target


def test_bootstrap_recovers_stable_coefficient_signs():
    expression, target = _toy_data()
    settings = InterpretabilitySettings(coefficient_bootstraps=30, cv_repeats=1)
    result = coefficient_bootstrap_stability(expression, target, settings).set_index(
        "gene"
    )
    assert result.loc["UP", "positive_sign_fraction"] == 1.0
    assert result.loc["DOWN", "positive_sign_fraction"] == 0.0
    assert (result["sign_stability"] == 1.0).all()


def test_contributions_reconstruct_sample_specific_gene_terms():
    expression, target = _toy_data()
    settings = InterpretabilitySettings(coefficient_bootstraps=10, cv_repeats=1)
    coefficients, contributions = fit_full_model_and_contributions(
        expression, target, expression, settings
    )
    assert set(coefficients["gene"]) == {"UP", "DOWN"}
    assert len(contributions) == expression.size
    case_up = contributions.loc[
        (contributions["sample_id"] == "e1") & (contributions["gene"] == "UP"),
        "logit_contribution",
    ].item()
    assert case_up > 0

