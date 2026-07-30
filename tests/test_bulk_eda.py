"""Tests for verified GSE135485 bulk RNA-seq EDA."""

from pathlib import Path

import pandas as pd

from src.bulk_eda import (
    BulkEDASettings,
    align_metadata_to_counts,
    parse_geo_soft_samples,
    sample_qc_metrics,
)


SOFT = """^SAMPLE = GSM1
!Sample_title = E1_lane
!Sample_source_name_ch1 = lesion
!Sample_characteristics_ch1 = subject status: patient with endometriosis
!Sample_characteristics_ch1 = tissue: endometrial samples and lesions
^SAMPLE = GSM2
!Sample_title = EN1_lane
!Sample_source_name_ch1 = healthy endometrium
!Sample_characteristics_ch1 = subject status: healthy control
!Sample_characteristics_ch1 = tissue: healthy endometrium
"""


def test_soft_metadata_and_exact_alignment(tmp_path: Path) -> None:
    path = tmp_path / "family.soft"
    path.write_text(SOFT, encoding="utf-8")
    metadata = parse_geo_soft_samples(path)
    counts = pd.DataFrame({"E1_lane": [1, 2], "EN1_lane": [3, 4]}, index=["A", "B"])
    aligned = align_metadata_to_counts(metadata, counts)
    assert aligned["sample_id"].tolist() == ["GSM1", "GSM2"]
    assert aligned["condition"].tolist() == ["Endometriosis", "Control"]


def test_sample_qc_flags_extreme_library() -> None:
    metadata = pd.DataFrame(
        {
            "count_matrix_column": ["A", "B", "C", "D"],
            "condition": ["Control", "Control", "Endometriosis", "Endometriosis"],
        }
    )
    counts = pd.DataFrame(
        {
            "A": [100, 100, 100],
            "B": [100, 100, 100],
            "C": [100, 100, 100],
            "D": [1, 0, 0],
        },
        index=["MT-X", "RPS1", "GENE"],
    )
    qc = sample_qc_metrics(
        counts, metadata, BulkEDASettings(robust_outlier_mad=2.0)
    ).set_index("count_matrix_column")
    assert qc.loc["D", "qc_status"] == "review"
