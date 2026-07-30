"""Tests for the frozen GSE120103 intake rules."""

import numpy as np
import pandas as pd

from src.external_cohort_intake import map_frozen_signature


def test_map_frozen_signature_uses_highest_label_blind_median() -> None:
    expression = pd.DataFrame(
        {
            "S1": [1.0, 4.0, 2.0],
            "S2": [2.0, 5.0, 3.0],
            "S3": [3.0, 6.0, 4.0],
        },
        index=["low_probe", "high_probe", "other_probe"],
    )
    annotation = pd.DataFrame(
        {
            "probe_id": ["low_probe", "high_probe", "other_probe"],
            "gene_symbol": ["PDZD2", "PDZD2", "DPF3"],
            "gene_id": ["23037", "23037", "8110"],
        }
    )

    signature, mapping = map_frozen_signature(
        expression, annotation, frozen_genes=("PDZD2", "DPF3")
    )

    assert list(signature.index) == ["PDZD2", "DPF3"]
    np.testing.assert_array_equal(signature.loc["PDZD2"], [4.0, 5.0, 6.0])
    selected = mapping.loc[mapping["selected_probe"]].set_index("gene_symbol")
    assert selected.loc["PDZD2", "probe_id"] == "high_probe"


def test_map_frozen_signature_does_not_impute_missing_gene() -> None:
    expression = pd.DataFrame({"S1": [1.0], "S2": [2.0]}, index=["probe"])
    annotation = pd.DataFrame(
        {"probe_id": ["probe"], "gene_symbol": ["PDZD2"], "gene_id": ["23037"]}
    )

    signature, _ = map_frozen_signature(
        expression, annotation, frozen_genes=("PDZD2", "DUOX1")
    )

    assert list(signature.index) == ["PDZD2"]
