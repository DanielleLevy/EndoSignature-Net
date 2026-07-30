"""Unit tests for the conservative filename-based metadata parser."""

import gzip
import tempfile
import unittest
from pathlib import Path

from src.metadata import FileRecord, build_samples


def record(name: str, fmt: str = "10x_h5") -> FileRecord:
    return FileRecord(name, f"data/{name}", 1, fmt, "scRNA-seq", "GSE179640", name.split("_", 1)[0])


class MetadataTests(unittest.TestCase):
    def test_control_sample(self):
        sample = build_samples([record("GSM6102532_C01_Ctrl_filtered_feature_bc_matrix.h5")])[0]
        self.assertEqual((sample.patient_id, sample.condition), ("C01", "Control"))
        self.assertEqual(sample.tissue, "control")
        self.assertTrue(sample.include_in_sc_eda)

    def test_endometriosis_tissue(self):
        sample = build_samples([record("GSM6102552_E07_EcO_filtered_feature_bc_matrix.h5")])[0]
        self.assertEqual((sample.patient_id, sample.condition), ("E07", "Endometriosis"))
        self.assertEqual(sample.tissue, "ectopic_ovary")

    def test_matrix_market_files_are_one_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                Path(directory) / "GSM1_E01_EuE_barcodes.tsv.gz",
                Path(directory) / "GSM1_E01_EuE_features.tsv.gz",
                Path(directory) / "GSM1_E01_EuE_matrix.mtx.gz",
            ]
            with gzip.open(paths[1], "wt") as handle:
                handle.write("ENSG1\tGENE1\tGene Expression\n")
            members = [
                FileRecord(
                    path.name,
                    str(path),
                    1,
                    "10x_mtx" if "matrix" in path.name else "10x_mtx_companion",
                    "scRNA-seq",
                    "GSE179640",
                    "GSM1",
                )
                for path in paths
            ]
            sample = build_samples(members)[0]
            self.assertEqual(sample.sample_id, "GSM1")
            self.assertEqual(len(sample.source_files.split(";")), 3)
            self.assertTrue(sample.include_in_sc_eda)

    def test_unknown_name_is_not_assigned_a_condition(self):
        sample = build_samples([record("GSM9999999_filtered_feature_bc_matrix.h5")])[0]
        self.assertEqual(sample.condition, "")
        self.assertFalse(sample.include_in_sc_eda)
        self.assertEqual(sample.review_status, "needs_review")

    def test_hashtag_matrix_is_excluded_from_expression_eda(self):
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "GSM1_EOR01_features.tsv.gz"
            with gzip.open(feature_path, "wt") as handle:
                handle.write("HHTO1-AAAA\nunmapped\n")
            members = [
                FileRecord(
                    feature_path.name,
                    str(feature_path),
                    1,
                    "10x_mtx_companion",
                    "scRNA-seq",
                    "GSE179640",
                    "GSM1",
                )
            ]
            sample = build_samples(members)[0]
            self.assertEqual(sample.technology, "cell_hashing")
            self.assertFalse(sample.include_in_sc_eda)


if __name__ == "__main__":
    unittest.main()
