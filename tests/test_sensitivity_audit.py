import numpy as np
import pandas as pd

from src.sensitivity_audit import (
    aggregate_repeated_predictions,
    benjamini_hochberg,
    composition_permanova,
    exact_family_composition_tests,
)


def test_bh_adjustment_preserves_order_and_monotonic_rank():
    adjusted = benjamini_hochberg([0.04, 0.001, 0.02])
    assert np.allclose(adjusted, [0.04, 0.003, 0.03])


def test_repeated_predictions_are_averaged_per_sample():
    frame = pd.DataFrame(
        {"model": ["m", "m"], "sample_id": ["s", "s"], "patient_id": ["p", "p"],
         "target": [1, 1], "probability": [0.2, 0.8]}
    )
    result = aggregate_repeated_predictions(frame)
    assert result.loc[0, "probability"] == 0.5


def test_exact_composition_test_uses_patient_label_assignments():
    index = pd.MultiIndex.from_tuples(
        [("c1", "Control"), ("c2", "Control"), ("e1", "Endometriosis"), ("e2", "Endometriosis")],
        names=["patient_id", "condition"],
    )
    matrix = pd.DataFrame({"A": [0.1, 0.1, 0.9, 0.9], "B": [0.9, 0.9, 0.1, 0.1]}, index=index)
    result = exact_family_composition_tests(matrix)
    assert set(result.cell_family) == {"A", "B"}
    assert result.n_label_assignments.eq(6).all()
    overall = composition_permanova(matrix)
    assert overall["n_label_assignments"] == 6
    assert 0 <= overall["exact_permutation_p"] <= 1
