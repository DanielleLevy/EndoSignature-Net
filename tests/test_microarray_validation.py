"""Tests for GSE51981 cycle-adjusted candidate validation."""

import pandas as pd

from src.microarray_validation import (
    MicroarrayValidationSettings,
    fit_phase_adjusted_effect,
    select_predeclared_candidates,
    validate_microarray_candidates,
)


def test_candidate_selection_uses_robust_bulk_tier() -> None:
    bulk = pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "discovery_direction": ["higher_in_endometriosis"] * 3,
            "directionally_replicated": [True, True, False],
            "statistically_supported": [True, False, False],
            "discovery_cell_families": ["T", "NK", "E"],
            "bulk_adjusted_log2_cpm_difference": [1.0, 0.5, 0.2],
        }
    )
    selected = select_predeclared_candidates(bulk)
    assert set(selected["gene"]) == {"A", "B"}
    assert selected.set_index("gene").loc["A", "prior_evidence_tier"] == (
        "gse135485_fdr_supported"
    )


def _metadata() -> pd.DataFrame:
    rows = []
    for group, prefix in [
        ("Endometriosis", "D"),
        ("Healthy_control_no_pathology", "C"),
        ("Non_endometriosis_other_pathology", "P"),
    ]:
        for index in range(4):
            rows.append(
                {
                    "sample_id": f"{prefix}{index}",
                    "clinical_group": group,
                    "cycle_phase": (
                        "Proliferative" if index % 2 == 0 else "Mid_Secretory"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_phase_adjusted_model_recovers_condition_effect() -> None:
    metadata = _metadata()
    expression = pd.Series(
        {
            row.sample_id: (
                5.0
                + (2.0 if row.clinical_group == "Endometriosis" else 0.0)
                + (1.0 if row.cycle_phase == "Mid_Secretory" else 0.0)
            )
            for row in metadata.itertuples(index=False)
        }
    )
    fit = fit_phase_adjusted_effect(
        expression,
        metadata,
        "Endometriosis",
        "Healthy_control_no_pathology",
    )
    assert abs(fit["adjusted_expression_difference"] - 2.0) < 1e-8
    assert fit["design_rank"] == fit["n_design_columns"]


def test_validation_requires_sensitivity_and_loo_direction() -> None:
    metadata = _metadata()
    values = {}
    for row in metadata.itertuples(index=False):
        disease = row.clinical_group == "Endometriosis"
        phase = row.cycle_phase == "Mid_Secretory"
        values[row.sample_id] = 4.0 + 3.0 * disease + 0.5 * phase
    expression = pd.DataFrame(values, index=["A"])
    candidates = pd.DataFrame(
        {
            "gene": ["A"],
            "discovery_direction": ["higher_in_endometriosis"],
            "discovery_cell_families": ["T_cell"],
            "prior_evidence_tier": ["gse135485_fdr_supported"],
            "bulk_adjusted_log2_cpm_difference": [1.0],
        }
    )
    qc = metadata[["sample_id"]].copy()
    qc["qc_status"] = "pass_processed_qc"
    settings = MicroarrayValidationSettings()
    results, unavailable = validate_microarray_candidates(
        candidates, expression, metadata, qc, settings
    )
    assert unavailable.empty
    assert results.loc[0, "robust_primary_directional_replication"]
    assert results.loc[0, "endometriosis_specific_support"]
