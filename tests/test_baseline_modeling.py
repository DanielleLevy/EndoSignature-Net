"""Tests for leakage-controlled frozen-signature baseline modeling."""

import pandas as pd

from src.baseline_modeling import (
    BaselineSettings,
    prepare_modeling_table,
    run_repeated_nested_cv,
    summarize_repeated_metrics,
)


def test_prepare_modeling_table_keeps_only_clean_control_contrast() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": ["E1", "E2", "C1", "O1"],
            "patient_id": ["P1", "P2", "P3", "P4"],
            "clinical_group": [
                "Endometriosis",
                "Endometriosis",
                "Healthy_control_no_pathology",
                "Non_endometriosis_other_pathology",
            ],
            "cycle_phase": ["A", "B", "A", "B"],
        }
    )
    expression = pd.DataFrame(
        {
            "gene": ["PDZD2", "OTHER"],
            "E1": [3.0, 1.0],
            "E2": [4.0, 1.0],
            "C1": [1.0, 1.0],
            "O1": [2.0, 1.0],
        }
    )
    table = prepare_modeling_table(
        expression, metadata, ["PDZD2", "OTHER"]
    )
    assert set(table["sample_id"]) == {"E1", "E2", "C1"}
    assert table["target"].sum() == 2


def test_prepare_modeling_table_supports_other_pathology_contrast() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": ["E1", "C1", "O1"],
            "patient_id": ["P1", "P2", "P3"],
            "clinical_group": [
                "Endometriosis",
                "Healthy_control_no_pathology",
                "Non_endometriosis_other_pathology",
            ],
            "cycle_phase": ["Unknown", "A", "B"],
        }
    )
    expression = pd.DataFrame(
        {
            "gene": ["PDZD2"],
            "E1": [3.0],
            "C1": [1.0],
            "O1": [2.0],
        }
    )
    table = prepare_modeling_table(
        expression,
        metadata,
        ["PDZD2"],
        group_mapping={
            "Endometriosis": 1,
            "Non_endometriosis_other_pathology": 0,
        },
    )
    assert set(table["sample_id"]) == {"E1", "O1"}
    assert table["target"].sum() == 1
    assert table.loc[table["sample_id"].eq("E1"), "cycle_phase"].isna().all()


def test_repeated_nested_cv_produces_complete_patient_predictions() -> None:
    rows = []
    for index in range(24):
        target = int(index >= 12)
        rows.append(
            {
                "sample_id": f"S{index}",
                "patient_id": f"P{index}",
                "clinical_group": (
                    "Endometriosis"
                    if target
                    else "Healthy_control_no_pathology"
                ),
                "cycle_phase": ["A", "B", "C"][index % 3],
                "target": target,
                "PDZD2": target * 2.0 + (index % 4) * 0.05,
                "GENE2": target * 1.0 + (index % 5) * 0.03,
            }
        )
    table = pd.DataFrame(rows)
    settings = BaselineSettings(
        outer_folds=3,
        outer_repeats=2,
        inner_folds=2,
        regularization_grid=(0.1, 1.0),
    )
    metrics, predictions, tuning = run_repeated_nested_cv(
        table, ["PDZD2", "GENE2"], settings
    )
    assert len(metrics) == 2 * 5
    assert len(predictions) == 2 * 5 * len(table)
    assert predictions.groupby(["repeat", "model"])["sample_id"].nunique().eq(
        len(table)
    ).all()
    assert not tuning.empty
    summary = summarize_repeated_metrics(metrics)
    assert set(summary["model"]) == {
        "prevalence_only",
        "cycle_only",
        "pdzd2_only",
        "frozen_signature_12",
        "frozen_signature_12_plus_cycle",
    }
