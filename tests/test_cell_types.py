"""Tests for clustering names, marker availability and composition logic."""

import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.cell_types import (
    ClusteringSettings,
    add_cluster_annotations,
    available_marker_panels,
    cluster_composition_tables,
    leiden_key,
)


class CellTypeTests(unittest.TestCase):
    def test_leiden_key_is_stable(self):
        self.assertEqual(leiden_key(0.8), "leiden_0_8")

    def test_primary_resolution_must_be_requested(self):
        with self.assertRaises(ValueError):
            ClusteringSettings(resolutions=(0.5,), primary_resolution=0.8).validate()

    def test_missing_markers_are_omitted(self):
        panels = available_marker_panels(["EPCAM", "KRT8", "VWF"])
        self.assertEqual(panels["Epithelial"], ["EPCAM", "KRT8"])
        self.assertEqual(panels["Mast_cell"], [])

    def test_annotations_map_to_cells(self):
        adata = ad.AnnData(sparse.csr_matrix(np.eye(2)))
        adata.obs["leiden_0_8"] = pd.Categorical(["0", "1"])
        annotations = pd.DataFrame(
            {"cluster": ["0", "1"], "provisional_cell_family": ["Epithelial", "Myeloid"]}
        )
        result = add_cluster_annotations(adata, "leiden_0_8", annotations)
        self.assertEqual(result.obs["provisional_cell_family"].astype(str).tolist(), ["Epithelial", "Myeloid"])

    def test_composition_is_normalized_within_sample(self):
        adata = ad.AnnData(sparse.csr_matrix(np.eye(4)))
        adata.obs["sample_id"] = pd.Categorical(["S1", "S1", "S2", "S2"])
        adata.obs["patient_id"] = pd.Categorical(["P1", "P1", "P2", "P2"])
        adata.obs["condition"] = pd.Categorical(["Control", "Control", "Endometriosis", "Endometriosis"])
        adata.obs["cluster"] = pd.Categorical(["0", "1", "0", "0"])
        counts, _, _ = cluster_composition_tables(adata, "cluster")
        totals = counts.groupby("sample_id", observed=True)["fraction_within_sample"].sum()
        self.assertTrue(np.allclose(totals, 1.0))


if __name__ == "__main__":
    unittest.main()
