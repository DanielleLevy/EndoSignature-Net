"""Tests for patient-level pseudobulk signature discovery."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.signatures import (
    PseudobulkSettings,
    aggregate_pseudobulk,
    differential_expression_by_family,
)


def test_pseudobulk_sums_cells_and_uses_patient_replication() -> None:
    counts = sparse.csr_matrix([[1, 0], [2, 1], [0, 3], [1, 2]])
    atlas = ad.AnnData(counts.astype(float))
    atlas.layers["counts"] = counts
    atlas.var_names = ["A", "B"]
    atlas.obs["sample_id"] = ["S1", "S1", "S2", "S2"]
    atlas.obs["patient_id"] = ["P1", "P1", "P2", "P2"]
    atlas.obs["condition"] = ["Control", "Control", "Endometriosis", "Endometriosis"]
    atlas.obs["provisional_cell_family"] = ["T_cell"] * 4
    settings = PseudobulkSettings(
        min_cells_per_pseudobulk=2,
        min_control_patients=2,
        min_endometriosis_patients=2,
    )
    matrix, metadata, eligibility = aggregate_pseudobulk(atlas, settings)
    assert matrix.tolist() == [[3, 1], [1, 5]]
    assert metadata["n_cells"].tolist() == [2, 2]
    assert not eligibility.loc[0, "eligible_for_pseudobulk_de"]


def test_differential_expression_reports_direction() -> None:
    matrix = np.array(
        [
            [100, 10],
            [110, 10],
            [400, 10],
            [420, 10],
        ],
        dtype=float,
    )
    metadata = pd.DataFrame(
        {
            "provisional_cell_family": ["T_cell"] * 4,
            "condition": ["Control", "Control", "Endometriosis", "Endometriosis"],
        }
    )
    eligibility = pd.DataFrame(
        {
            "provisional_cell_family": ["T_cell"],
            "eligible_for_pseudobulk_de": [True],
        }
    )
    settings = PseudobulkSettings(
        min_control_patients=2,
        min_endometriosis_patients=2,
        min_samples_expressing=2,
    )
    result = differential_expression_by_family(
        matrix, metadata, pd.Index(["A", "B"]), eligibility, settings
    ).set_index("gene")
    assert result.loc["A", "direction"] == "higher_in_endometriosis"
    assert result.loc["B", "direction"] == "lower_in_endometriosis"
