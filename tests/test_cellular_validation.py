"""Tests for marker-supported GSE213216 candidate localization."""

import pandas as pd

from src.cellular_validation import (
    CellularValidationSettings,
    evaluate_cell_family_localization,
    select_candidate_gene_families,
    summarize_gene_evidence,
)


def test_candidate_selection_keeps_only_directionally_replicated_pairs() -> None:
    bulk = pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "provisional_cell_family": ["T_cell", "NK_cell", "Epithelial"],
            "directionally_replicated": [True, True, False],
            "statistically_supported": [True, False, False],
        }
    )
    selected = select_candidate_gene_families(bulk)
    assert selected["gene"].tolist() == ["A", "B"]
    assert selected.set_index("gene").loc["A", "priority_tier"] == (
        "high_priority_fdr_supported"
    )


def test_patient_level_localization_detects_expected_family_enrichment() -> None:
    rows = []
    for patient in range(1, 7):
        for family, mean, detected in [
            ("T_cell", 2.0 + patient / 100, 80),
            ("NK_cell", 0.2, 10),
        ]:
            rows.append(
                {
                    "patient_id": str(patient),
                    "tissue_code": "EuE",
                    "cell_family": family,
                    "gene": "A",
                    "n_cells": 100,
                    "expression_sum": mean * 100,
                    "detected_cells": detected,
                    "mean_log1p_cp10k": mean,
                    "detection_fraction": detected / 100,
                }
            )
    candidates = pd.DataFrame(
        {
            "gene": ["A"],
            "provisional_cell_family": ["T_cell"],
            "priority_tier": ["high_priority_fdr_supported"],
            "statistically_supported": [True],
        }
    )
    settings = CellularValidationSettings(included_tissues=("EuE",))
    result, paired = evaluate_cell_family_localization(
        pd.DataFrame(rows), candidates, settings
    )
    assert len(paired) == 6
    assert result.loc[0, "eligible_for_localization_test"]
    assert result.loc[0, "directionally_consistent_localization"]
    assert result.loc[0, "cell_family_localized"]


def test_gene_summary_counts_supported_tissues() -> None:
    results = pd.DataFrame(
        {
            "gene": ["A", "A"],
            "priority_tier": ["high", "high"],
            "tissue_code": ["EuE", "EndoLesion"],
            "eligible_for_localization_test": [True, True],
            "cell_family_localized": [True, False],
            "fdr_bh": [0.05, 0.5],
            "positive_expression_difference_fraction": [0.8, 0.4],
        }
    )
    summary = summarize_gene_evidence(results)
    assert summary.loc[0, "n_localized_tests"] == 1
    assert summary.loc[0, "localized_tissues"] == "EuE"
