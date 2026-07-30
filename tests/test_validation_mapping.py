"""Tests for patient-aware validation label transfer."""

import anndata as ad
import numpy as np
from scipy import sparse

from src.validation_mapping import (
    MappingSettings,
    add_transferred_labels,
    build_classifier,
    prepare_query_for_reference,
)


def test_query_projection_remains_sparse() -> None:
    query = ad.AnnData(sparse.csr_matrix([[1, 2, 0], [0, 1, 3]]))
    query.var_names = ["A", "B", "C"]
    settings = MappingSettings()
    projected = prepare_query_for_reference(
        query,
        ["A", "B", "C"],
        np.array([True, True, False]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        settings,
    )
    assert sparse.issparse(projected.X)
    assert projected.obsm["X_pca_reference"].shape == (2, 2)


def test_query_projection_allows_filtered_reference_genes() -> None:
    query = ad.AnnData(sparse.csr_matrix([[1, 2, 3]]))
    query.var_names = ["A", "C", "D"]
    projected = prepare_query_for_reference(
        query,
        ["A", "B", "C", "D"],
        np.array([True, True, True, True]),
        np.vstack([np.eye(3), np.ones(3)]),
        MappingSettings(),
    )
    assert projected.uns["reference_mapping"]["n_common_genes"] == 3
    assert projected.obsm["X_pca_reference"].shape == (1, 3)


def test_transferred_labels_include_confidence() -> None:
    settings = MappingSettings(confidence_threshold=0.6)
    model = build_classifier(settings)
    model.fit(
        np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.array(["A", "A", "B", "B"]),
    )
    query = ad.AnnData(sparse.csr_matrix(np.eye(2)))
    query.obsm["X_pca_reference"] = np.array([[-2.0, 0.0], [2.0, 0.0]])
    result = add_transferred_labels(query, model, settings)
    assert result.obs["transferred_cell_family"].astype(str).tolist() == ["A", "B"]
    assert np.all(result.obs["mapping_confidence"].to_numpy() > 0.5)
