"""Tests for conservative external marker-concordance logic."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.marker_validation import (
    MarkerValidationSettings,
    add_marker_validation,
    validation_summaries,
)


def test_supported_and_uncertain_labels() -> None:
    genes = ["EPCAM", "KRT8", "KRT18", "CD3D", "CD3E", "TRAC"]
    counts = sparse.csr_matrix(
        [
            [5, 4, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 3, 5],
        ]
    )
    adata = ad.AnnData(counts.astype(float))
    adata.var_names = genes
    adata.layers["counts"] = counts.copy()
    adata.obs["transferred_cell_family"] = pd.Categorical(
        ["Epithelial", "Epithelial", "T_cell"]
    )
    adata.obs["mapping_confidence"] = [0.9, 0.9, 0.9]

    result = add_marker_validation(adata, MarkerValidationSettings())

    assert result.obs["marker_validation_status"].astype(str).iloc[0] == "marker_supported"
    assert result.obs["marker_validation_status"].astype(str).iloc[1] != "marker_supported"
    assert result.obs["conservative_cell_family"].astype(str).iloc[1] == "Uncertain"


def test_summary_reports_patient_support() -> None:
    cells = pd.DataFrame(
        {
            "cell_id": ["A", "B"],
            "sample_id": ["S1", "S2"],
            "patient_id": ["P1", "P2"],
            "condition": ["X", "X"],
            "tissue_code": ["T", "T"],
            "transferred_cell_family": ["Epithelial", "Epithelial"],
            "marker_validation_status": ["marker_supported", "uncertain"],
            "marker_panel_agreement": [True, False],
            "detected_predicted_markers": [3, 0],
        }
    )
    family, _, _ = validation_summaries(cells)
    assert family.loc[0, "n_patients"] == 2
    assert family.loc[0, "marker_supported_pct"] == 50.0
