"""Tests for cross-cohort patient metadata harmonization."""

import pandas as pd

from src.cross_cohort_metadata import (
    aggregate_patient_metadata,
    clean_value,
    metadata_completeness,
    normalize_cycle_phase,
)


def test_missing_tokens_are_not_treated_as_observed_metadata():
    assert pd.isna(clean_value("Unknown"))
    assert pd.isna(clean_value("not stated"))
    assert clean_value("Mid_Secretory") == "Mid_Secretory"
    assert pd.isna(clean_value("Not stated for controls in GEO"))


def test_cycle_phase_spelling_is_harmonized():
    assert normalize_cycle_phase("Mid-secretory") == "Mid_Secretory"
    assert normalize_cycle_phase("Secretory") == "Secretory_unspecified"


def test_patient_aggregation_preserves_multiple_tissues():
    samples = pd.DataFrame(
        {
            "study_id": ["GSE1", "GSE1"],
            "patient_id": ["P1", "P1"],
            "sample_id": ["S1", "S2"],
            "condition": ["Endometriosis", "Endometriosis"],
            "tissue": ["eutopic", "lesion"],
        }
    )
    result = aggregate_patient_metadata(samples)
    assert len(result) == 1
    assert result.loc[0, "n_samples"] == 2
    assert result.loc[0, "tissue"] == "eutopic;lesion"


def test_completeness_is_calculated_at_patient_level():
    patients = pd.DataFrame(
        {
            "study_id": ["GSE1", "GSE1"],
            "cycle_phase": ["Secretory", pd.NA],
        }
    )
    result = metadata_completeness(patients, ["cycle_phase"])
    assert result.loc[0, "n_with_metadata"] == 1
    assert result.loc[0, "completeness_fraction"] == 0.5
