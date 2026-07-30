"""Tests for label-blind raw Agilent rescue utilities."""

import numpy as np
import pandas as pd

from src.agilent_raw_rescue import quantile_normalize


def test_quantile_normalize_gives_identical_distributions() -> None:
    matrix = pd.DataFrame(
        {"sample_a": [5.0, 2.0, 3.0], "sample_b": [4.0, 1.0, 8.0]},
        index=["p1", "p2", "p3"],
    )

    normalized = quantile_normalize(matrix)

    np.testing.assert_allclose(
        np.sort(normalized["sample_a"]), np.sort(normalized["sample_b"])
    )
    assert normalized.loc["p1", "sample_a"] > normalized.loc["p3", "sample_a"]


def test_quantile_normalize_preserves_shape_and_labels() -> None:
    matrix = pd.DataFrame(
        {"S1": [1.0, 2.0], "S2": [3.0, 4.0]}, index=["A", "B"]
    )

    normalized = quantile_normalize(matrix)

    assert normalized.shape == matrix.shape
    assert list(normalized.index) == ["A", "B"]
    assert list(normalized.columns) == ["S1", "S2"]
