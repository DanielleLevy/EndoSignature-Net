"""Tests for transcript-to-gene aggregation in GSE153740."""

import pandas as pd


def test_transcript_sum_is_gene_level_fpkm() -> None:
    expression = pd.DataFrame(
        {"S1": [1.0, 2.0, 5.0], "S2": [3.0, 4.0, 6.0]},
        index=["T1", "T2", "T3"],
    )
    mapping = pd.Series({"T1": "A", "T2": "A", "T3": "B"}, name="gene")
    aligned = expression.copy()
    aligned.insert(0, "gene", mapping.loc[expression.index].to_numpy())
    collapsed = aligned.groupby("gene").sum()
    assert collapsed.loc["A", "S1"] == 3.0
    assert collapsed.loc["B", "S2"] == 6.0
