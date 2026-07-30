"""Tests for verified GSE51981 processed microarray EDA."""

import gzip
from pathlib import Path

import pandas as pd

from src.microarray_eda import (
    MicroarrayEDASettings,
    collapse_probes_to_genes,
    pca_factor_associations,
    processed_expression_qc,
)


def test_probe_collapse_uses_median_and_excludes_no_rows() -> None:
    expression = pd.DataFrame(
        {"S1": [2.0, 4.0, 8.0], "S2": [3.0, 5.0, 9.0]},
        index=["P1", "P2", "P3"],
    )
    annotation = pd.DataFrame(
        {
            "probe_id": ["P1", "P2", "P3"],
            "gene_symbol": ["A", "A", "B"],
            "gene_id": ["1", "1", "2"],
        }
    )
    genes, mapping = collapse_probes_to_genes(expression, annotation)
    assert genes.loc["A", "S1"] == 3.0
    assert genes.loc["B", "S2"] == 9.0
    assert mapping.set_index("gene").loc["A", "n_probes"] == 2


def test_processed_qc_flags_low_correlation_sample() -> None:
    expression = pd.DataFrame(
        {
            "S1": [1, 2, 3, 4, 5, 6],
            "S2": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "S3": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
            "S4": [6, 1, 5, 2, 4, 3],
        },
        index=list("ABCDEF"),
    )
    metadata = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "clinical_group": ["Control", "Control", "Disease", "Disease"],
        }
    )
    qc, correlation = processed_expression_qc(
        expression,
        metadata,
        MicroarrayEDASettings(min_median_sample_correlation=0.7),
    )
    assert qc.set_index("sample_id").loc["S4", "qc_status"] == "review"
    assert correlation.shape == (4, 4)


def test_pca_factor_association_detects_group_separation() -> None:
    pca = pd.DataFrame(
        {
            "clinical_group": ["A", "A", "B", "B"],
            "cycle_phase": ["X", "Y", "X", "Y"],
            "PC1": [-2.0, -1.0, 1.0, 2.0],
            "PC2": [-1.0, 1.0, -1.0, 1.0],
        }
    )
    result = pca_factor_associations(pca, n_components=2)
    lookup = result.set_index(["factor", "component"])["eta_squared"]
    assert lookup.loc[("clinical_group", "PC1")] > 0.8
    assert lookup.loc[("cycle_phase", "PC2")] == 1.0
