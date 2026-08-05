import json

import pandas as pd
import pytest

from src.explorer import gene_view, load_explorer_data


def test_gene_view_returns_one_summary_row():
    effects = pd.DataFrame({"gene": ["A", "A"], "cohort": ["x", "y"], "standardized_effect": [1, -1]})
    summary = pd.DataFrame({"gene": ["A"], "sign_stability": [1.0], "permutation_auc_decrease_mean": [0.1]})
    rows, selected = gene_view("A", effects, summary)
    assert len(rows) == 2
    assert selected["sign_stability"] == 1.0


def test_loader_rejects_missing_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_explorer_data(tmp_path)

