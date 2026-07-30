"""Tests for primary-cohort verification and inclusion decisions."""

import unittest

import pandas as pd

from src.cohort import (
    DoubletSettings,
    build_inclusion_decisions,
    build_validation_decisions,
    build_verified_primary_metadata,
)


class CohortTests(unittest.TestCase):
    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            DoubletSettings(expected_doublet_rate=0).validate()

    def test_verified_metadata_has_twelve_samples(self):
        samples = []
        from src.cohort import PRIMARY_SAMPLE_TITLES

        for sample_id in PRIMARY_SAMPLE_TITLES:
            control = sample_id in {"GSM6102532", "GSM6102533", "GSM6102534"}
            samples.append(
                {
                    "sample_id": sample_id,
                    "tissue_code": "Ctrl" if control else "EuE",
                    "condition": "Control" if control else "Endometriosis",
                }
            )
        verified = build_verified_primary_metadata(pd.DataFrame(samples))
        self.assertEqual(len(verified), 12)
        self.assertEqual(set(verified["geo_tissue"]), {"Endometrium"})

    def test_decisions_keep_other_tissues_secondary(self):
        metadata = pd.DataFrame(
            [
                {
                    "sample_id": "GSM6102532",
                    "study_id": "GSE179640",
                    "patient_id": "C01",
                    "condition": "Control",
                    "tissue_code": "Ctrl",
                    "technology": "scRNA-seq",
                },
                {
                    "sample_id": "GSM_OTHER",
                    "study_id": "GSE179640",
                    "patient_id": "E01",
                    "condition": "Endometriosis",
                    "tissue_code": "EcP",
                    "technology": "scRNA-seq",
                },
            ]
        )
        doublets = pd.DataFrame(
            [{"sample_id": "GSM6102532", "doublet_qc_status": "pass_initial_doublet_qc"}]
        )
        decisions = build_inclusion_decisions(metadata, doublets).set_index("sample_id")
        self.assertEqual(decisions.loc["GSM6102532", "decision"], "include")
        self.assertEqual(decisions.loc["GSM_OTHER", "decision"], "reserve")

    def test_validation_decisions_require_both_qc_passes(self):
        metadata = pd.DataFrame(
            [
                {
                    "sample_id": "GSM_PASS",
                    "study_id": "GSE213216",
                    "patient_id": "1",
                    "condition": "endometriosis_lesion",
                    "tissue_code": "EndoLesion",
                    "technology": "scRNA-seq",
                    "include_in_sc_eda": True,
                },
                {
                    "sample_id": "GSM_REVIEW",
                    "study_id": "GSE213216",
                    "patient_id": "2",
                    "condition": "endometriosis_lesion",
                    "tissue_code": "EndoLesion",
                    "technology": "scRNA-seq",
                    "include_in_sc_eda": True,
                },
            ]
        )
        cell_qc = pd.DataFrame(
            [
                {"sample_id": "GSM_PASS", "qc_status": "pass_initial_qc"},
                {"sample_id": "GSM_REVIEW", "qc_status": "review"},
            ]
        )
        doublets = pd.DataFrame(
            [
                {
                    "sample_id": sample_id,
                    "doublet_qc_status": "pass_initial_doublet_qc",
                }
                for sample_id in ("GSM_PASS", "GSM_REVIEW")
            ]
        )
        decisions = build_validation_decisions(
            metadata, cell_qc, doublets
        ).set_index("sample_id")
        self.assertEqual(decisions.loc["GSM_PASS", "decision"], "include")
        self.assertEqual(decisions.loc["GSM_REVIEW", "decision"], "review")


if __name__ == "__main__":
    unittest.main()
