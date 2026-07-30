"""Tests for GSE212787 frozen-gene extraction."""

import pandas as pd

from src.rnaseq_external_intake import extract_frozen_expression


def test_extract_frozen_expression_preserves_frozen_order() -> None:
    expression = pd.DataFrame(
        {"S1": [3, 7], "S2": [4, 8]},
        index=["ENSG_P", "ENSG_D"],
    )
    mapping = pd.DataFrame(
        {
            "gene": ["DPF3", "PDZD2"],
            "ncbi_gene_id": ["8110", "23037"],
            "ensembl_gene_id": ["ENSG_D", "ENSG_P"],
        }
    )

    result = extract_frozen_expression(
        expression, mapping, frozen_genes=("PDZD2", "DPF3")
    )

    assert list(result.index) == ["PDZD2", "DPF3"]
    assert result.loc["PDZD2", "S1"] == 3


def test_extract_frozen_expression_does_not_replace_missing_gene() -> None:
    expression = pd.DataFrame({"S1": [3]}, index=["ENSG_P"])
    mapping = pd.DataFrame(
        {
            "gene": ["PDZD2", "DUOX1"],
            "ncbi_gene_id": ["23037", "53905"],
            "ensembl_gene_id": ["ENSG_P", "ENSG_MISSING"],
        }
    )

    result = extract_frozen_expression(
        expression, mapping, frozen_genes=("PDZD2", "DUOX1")
    )

    assert list(result.index) == ["PDZD2"]
