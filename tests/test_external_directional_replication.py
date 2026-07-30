"""Tests for frozen external-replication helpers."""

import numpy as np
import pandas as pd

from src.external_directional_replication import (
    cohort_standardize,
    cycle_stratified_concordance,
    signed_directional_score,
    stratified_auc_interval,
)


def test_cohort_standardize_is_label_blind_per_gene() -> None:
    expression = pd.DataFrame(
        {"S1": [1.0, 10.0], "S2": [2.0, 12.0], "S3": [3.0, 14.0]},
        index=["A", "B"],
    )
    result = cohort_standardize(expression)
    np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.std(axis=1, ddof=0), 1.0)


def test_signed_directional_score_respects_lower_direction() -> None:
    expression = pd.DataFrame(
        {"S1": [1.0, 2.0], "S2": [3.0, -2.0]}, index=["UP", "DOWN"]
    )
    directions = pd.Series(
        {"UP": "higher_in_endometriosis", "DOWN": "lower_in_endometriosis"}
    )
    score = signed_directional_score(expression, directions)
    assert score["S2"] > score["S1"]


def test_stratified_auc_interval_is_bounded() -> None:
    target = pd.Series([0, 0, 1, 1], index=["a", "b", "c", "d"])
    score = pd.Series([0.1, 0.2, 0.8, 0.9], index=target.index)
    lower, upper = stratified_auc_interval(target, score, 100, 7)
    assert 0 <= lower <= upper <= 1
    assert lower == 1.0


def test_cycle_stratified_concordance_uses_within_phase_pairs() -> None:
    target = pd.Series([0, 1, 0, 1], index=["a", "b", "c", "d"])
    score = pd.Series([0.1, 0.9, 0.8, 0.2], index=target.index)
    phase = pd.Series(["P", "P", "S", "S"], index=target.index)
    concordance, details = cycle_stratified_concordance(target, score, phase)
    assert concordance == 0.5
    assert details["case_control_pairs"].sum() == 2
