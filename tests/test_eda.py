"""Tests for single-cell QC calculations and filtering."""

import unittest

import anndata as ad
import numpy as np

from src.eda import QCThresholds, calculate_qc, cell_filter_mask, filter_sample, summarize_sample


class EDATests(unittest.TestCase):
    def setUp(self):
        # Cell 0 passes; cell 1 has too few genes; cell 2 has high mt percentage.
        matrix = np.array([[10, 5, 1], [0, 1, 0], [90, 5, 5]], dtype=np.float32)
        self.adata = ad.AnnData(matrix)
        self.adata.var_names = ["MT-CO1", "GENE1", "GENE2"]
        calculate_qc(self.adata)
        self.thresholds = QCThresholds(
            min_genes_per_cell=2,
            max_pct_mito=80.0,
            min_cells_per_gene=1,
            min_cells_after_qc=1,
            min_cell_retention_pct=20.0,
        )

    def test_qc_metrics_are_created(self):
        self.assertIn("pct_counts_mt", self.adata.obs)
        self.assertIn("n_genes_by_counts", self.adata.obs)

    def test_filter_mask_applies_gene_and_mito_thresholds(self):
        self.assertEqual(cell_filter_mask(self.adata, self.thresholds).tolist(), [True, False, False])

    def test_filter_returns_copy(self):
        filtered = filter_sample(self.adata, self.thresholds)
        self.assertEqual(filtered.n_obs, 1)
        self.assertEqual(self.adata.n_obs, 3)

    def test_summary_is_sample_level(self):
        filtered = filter_sample(self.adata, self.thresholds)
        metadata = {
            "sample_id": "GSM1",
            "study_id": "GSE1",
            "patient_id": "P1",
            "condition": "Control",
            "tissue": "control",
            "tissue_code": "Ctrl",
        }
        summary = summarize_sample(self.adata, filtered, metadata, self.thresholds)
        self.assertEqual(summary["n_cells_raw"], 3)
        self.assertAlmostEqual(summary["cell_retention_pct"], 100 / 3)

    def test_invalid_thresholds_fail_early(self):
        with self.assertRaises(ValueError):
            QCThresholds(max_pct_mito=120).validate()


if __name__ == "__main__":
    unittest.main()
