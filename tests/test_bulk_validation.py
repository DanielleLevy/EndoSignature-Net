"""Tests for conservative GSE135485 signature validation."""

import numpy as np
import pandas as pd

from src.bulk_validation import (
    BulkValidationSettings,
    collapse_discovery_candidates,
    fit_hc3_condition_effect,
    validate_candidates,
)


def test_candidate_collapse_separates_direction_conflicts() -> None:
    discovery = pd.DataFrame(
        {
            "gene": ["A", "A", "B", "B"],
            "provisional_cell_family": ["T", "NK", "T", "NK"],
            "direction": [
                "higher_in_endometriosis",
                "higher_in_endometriosis",
                "higher_in_endometriosis",
                "lower_in_endometriosis",
            ],
            "log2_fold_change": [2.0, 1.0, 1.5, -1.2],
            "fdr_bh": [0.01, 0.02, 0.01, 0.03],
            "statistical_tier": ["stable"] * 4,
        }
    )
    candidates, conflicts, rows = collapse_discovery_candidates(discovery, "stable")
    assert candidates["gene"].tolist() == ["A"]
    assert conflicts["gene"].tolist() == ["B"]
    assert len(rows) == 4
    assert candidates.loc[0, "n_discovery_cell_families"] == 2


def _synthetic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4", "OUT"],
            "count_matrix_column": ["c1", "c2", "c3", "c4", "d1", "d2", "d3", "d4", "out"],
            "condition": ["Control"] * 4 + ["Endometriosis"] * 5,
            "lane": ["L001", "L002", "L001", "L002", "L001", "L002", "L001", "L002", "L002"],
        }
    )


def test_hc3_model_recovers_lane_adjusted_condition_direction() -> None:
    metadata = _synthetic_metadata().iloc[:-1].copy()
    expression = pd.Series(
        [2.0, 3.0, 2.1, 3.1, 4.0, 5.0, 4.2, 5.2],
        index=metadata["count_matrix_column"],
    )
    fit = fit_hc3_condition_effect(expression, metadata)
    assert fit["bulk_adjusted_log2_cpm_difference"] > 1.5
    assert fit["design_rank"] == fit["n_design_columns"]


def test_validation_requires_all_sensitivity_directions() -> None:
    metadata = _synthetic_metadata()
    candidates = pd.DataFrame(
        {
            "gene": ["UP", "DOWN"],
            "discovery_direction": [
                "higher_in_endometriosis",
                "lower_in_endometriosis",
            ],
            "n_discovery_cell_families": [1, 1],
            "discovery_cell_families": ["T", "NK"],
            "median_discovery_log2_fold_change": [2.0, -2.0],
            "maximum_absolute_discovery_log2_fold_change": [2.0, 2.0],
            "minimum_discovery_fdr": [0.01, 0.01],
        }
    )
    log_cpm = pd.DataFrame(
        {
            "c1": [2.0, 5.0],
            "c2": [2.1, 5.1],
            "c3": [1.9, 4.9],
            "c4": [2.0, 5.0],
            "d1": [5.0, 2.0],
            "d2": [5.1, 2.1],
            "d3": [4.9, 1.9],
            "d4": [5.0, 2.0],
            "out": [5.0, 2.0],
        },
        index=["UP", "DOWN"],
    )
    settings = BulkValidationSettings(outlier_sample_id="OUT")
    result, unavailable = validate_candidates(candidates, log_cpm, metadata, settings)
    assert unavailable.empty
    assert result["directionally_replicated"].all()
    assert np.all(result["control_loo_direction_stability"] == 1.0)
