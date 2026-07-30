"""Tests for atlas metadata joins, doublet filtering and concatenation."""

import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.atlas import AtlasSettings, apply_doublet_predictions, attach_metadata, concatenate_samples, matrix_is_sparse


class AtlasTests(unittest.TestCase):
    def test_attach_metadata_makes_cell_ids_global(self):
        adata = ad.AnnData(sparse.csr_matrix(np.eye(2)))
        adata.obs_names = ["AA", "BB"]
        row = {"sample_id": "GSM1", "patient_id": "P1", "condition": "Control", "tissue_code": "Ctrl"}
        result = attach_metadata(adata, row)
        self.assertEqual(result.obs_names.tolist(), ["GSM1:AA", "GSM1:BB"])

    def test_doublet_predictions_are_joined_by_id(self):
        adata = ad.AnnData(sparse.csr_matrix(np.eye(2)))
        adata.obs_names = ["GSM1:AA", "GSM1:BB"]
        predictions = pd.DataFrame(
            {
                "cell_id": ["GSM1:BB", "GSM1:AA"],
                "doublet_score": [0.9, 0.1],
                "predicted_doublet": [True, False],
            }
        )
        singlets, report = apply_doublet_predictions(adata, predictions)
        self.assertEqual(singlets.obs_names.tolist(), ["GSM1:AA"])
        self.assertEqual(report["n_predicted_doublets_removed"], 1)

    def test_missing_predictions_fail(self):
        adata = ad.AnnData(sparse.csr_matrix(np.eye(2)))
        adata.obs_names = ["A", "B"]
        predictions = pd.DataFrame({"cell_id": ["A"], "doublet_score": [0.1], "predicted_doublet": [False]})
        with self.assertRaises(ValueError):
            apply_doublet_predictions(adata, predictions)

    def test_concatenation_uses_gene_intersection_and_stays_sparse(self):
        first = ad.AnnData(sparse.csr_matrix([[1, 2]]))
        first.var_names = ["A", "B"]
        second = ad.AnnData(sparse.csr_matrix([[3, 4]]))
        second.var_names = ["B", "C"]
        for index, item in enumerate((first, second), start=1):
            item.obs_names = [f"S{index}:cell"]
            item.obs["sample_id"] = f"S{index}"
            item.obs["patient_id"] = f"P{index}"
            item.obs["condition"] = "Control" if index == 1 else "Endometriosis"
            item.obs["tissue_code"] = "Ctrl" if index == 1 else "EuE"
        result = concatenate_samples([first, second])
        self.assertEqual(result.var_names.tolist(), ["B"])
        self.assertTrue(matrix_is_sparse(result))

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            AtlasSettings(n_pcs=1).validate()


if __name__ == "__main__":
    unittest.main()
