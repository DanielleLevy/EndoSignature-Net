"""Tests for transparent cross-dataset evidence integration."""

import pandas as pd

from src.evidence_integration import (
    EvidenceIntegrationSettings,
    frozen_signature_tables,
    integrate_evidence,
    modeling_readiness_decision,
)


def _discovery() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": ["CORE", "EXT", "WATCH"],
            "provisional_cell_family": ["Epithelial"] * 3,
            "direction": ["higher_in_endometriosis"] * 3,
            "log2_fold_change": [2.0, 1.5, 1.2],
            "fdr_bh": [0.01, 0.02, 0.03],
            "statistical_tier": ["fdr_effect_direction_stable"] * 3,
        }
    )


def test_evidence_rules_separate_core_extended_and_watchlist() -> None:
    bulk = pd.DataFrame(
        {
            "gene": ["CORE", "EXT", "WATCH"],
            "directionally_replicated": [True, True, True],
            "statistically_supported": [True, False, False],
            "bulk_adjusted_log2_cpm_difference": [1.0, 0.8, 0.7],
            "fdr_bh": [0.01, 0.2, 0.3],
            "validation_tier": ["a", "b", "b"],
        }
    )
    cellular = pd.DataFrame(
        {
            "gene": ["CORE", "EXT", "WATCH"],
            "directionally_consistent_localization": [True, False, False],
            "cell_family_localized": [False, False, False],
            "eligible_for_localization_test": [True, True, True],
            "fdr_bh": [0.2, 1.0, 1.0],
            "tissue_code": ["EuE"] * 3,
            "expected_cell_family": ["Epithelial"] * 3,
        }
    )
    micro = pd.DataFrame(
        {
            "gene": ["CORE", "EXT", "WATCH"],
            "robust_primary_directional_replication": [True, True, True],
            "primary_statistically_supported": [True, True, True],
            "endometriosis_specific_support": [False, False, False],
            "primary_adjusted_expression_difference": [1.0, 0.9, 0.8],
            "primary_fdr_bh": [0.01, 0.02, 0.03],
            "other_pathology_adjusted_difference": [-0.1, -0.1, 0.2],
            "other_pathology_fdr_bh": [0.8, 0.7, 0.2],
            "other_pathology_direction_matches_discovery": [False, False, True],
            "validation_tier": ["x", "x", "x"],
        }
    )
    evidence = integrate_evidence(
        _discovery(), bulk, cellular, micro, EvidenceIntegrationSettings()
    )
    core, extended, watchlist = frozen_signature_tables(evidence)
    assert core["gene"].tolist() == ["CORE"]
    assert set(extended["gene"]) == {"CORE", "EXT", "WATCH"}
    assert watchlist["gene"].tolist() == ["WATCH"]


def test_readiness_does_not_claim_specific_signature_without_support() -> None:
    evidence = pd.DataFrame(
        {
            "gene": ["A", "B"],
            "core_multistudy_pathology_candidate": [True, False],
            "extended_clean_control_signature": [True, True],
            "specificity_watchlist": [False, True],
            "endometriosis_specific_biomarker_supported": [False, False],
        }
    )
    decision = modeling_readiness_decision(
        evidence, EvidenceIntegrationSettings()
    )
    assert not decision["endometriosis_specific_signature_ready"]
    assert decision["pathology_vs_clean_control_exploratory_signature_ready"]
    assert not decision["deep_learning_model_ready"]
