"""Cycle-phase and cell-composition sensitivity utilities.

All inference is performed at patient/sample level. The functions do not tune
the frozen signature or use external outcomes for feature selection.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def aggregate_repeated_predictions(predictions):
    """Average repeated out-of-fold predictions for each patient and model."""
    keys = ["model", "sample_id", "patient_id", "target"]
    return predictions.groupby(keys, as_index=False)["probability"].mean()


def stratified_prediction_metrics(predictions, metadata):
    """Calculate overall and cycle-stratified metrics without refitting."""
    joined = predictions.merge(
        metadata[["sample_id", "cycle_phase"]], on="sample_id", how="left", validate="many_to_one"
    )
    rows = []
    for model, model_frame in joined.groupby("model", observed=True):
        strata = [("All", model_frame)] + list(model_frame.groupby("cycle_phase", observed=True))
        for phase, frame in strata:
            counts = frame["target"].value_counts()
            if len(counts) < 2:
                auc = np.nan
                ap = np.nan
            else:
                auc = roc_auc_score(frame["target"], frame["probability"])
                ap = average_precision_score(frame["target"], frame["probability"])
            rows.append(
                {
                    "model": model,
                    "cycle_phase": phase,
                    "n_patients": len(frame),
                    "n_controls": int(counts.get(0, 0)),
                    "n_endometriosis": int(counts.get(1, 0)),
                    "roc_auc": auc,
                    "average_precision": ap,
                    "mean_probability_control": frame.loc[frame.target.eq(0), "probability"].mean(),
                    "mean_probability_endometriosis": frame.loc[frame.target.eq(1), "probability"].mean(),
                }
            )
    return pd.DataFrame(rows)


def patient_composition_matrix(composition):
    """Create a complete patient-by-family fraction matrix and labels."""
    matrix = composition.pivot_table(
        index=["patient_id", "condition"],
        columns="provisional_cell_family",
        values="fraction_within_sample",
        aggfunc="sum",
        fill_value=0.0,
    )
    return matrix.sort_index()


def _condition_assignments(n, n_cases):
    for case_indices in combinations(range(n), n_cases):
        labels = np.zeros(n, dtype=bool)
        labels[list(case_indices)] = True
        yield labels


def exact_family_composition_tests(matrix):
    """Test family fractions with all patient-level label permutations."""
    conditions = np.asarray(
        matrix.index.get_level_values("condition") == "Endometriosis", dtype=bool
    )
    assignments = list(_condition_assignments(len(matrix), int(conditions.sum())))
    transformed = np.arcsin(np.sqrt(matrix.to_numpy(dtype=float)))
    rows = []
    for column_index, family in enumerate(matrix.columns):
        values = transformed[:, column_index]
        observed = values[conditions].mean() - values[~conditions].mean()
        null = np.array([values[a].mean() - values[~a].mean() for a in assignments])
        p_value = np.mean(np.abs(null) >= abs(observed) - 1e-12)
        raw = matrix.iloc[:, column_index].to_numpy()
        rows.append(
            {
                "cell_family": family,
                "mean_fraction_control": raw[~conditions].mean(),
                "mean_fraction_endometriosis": raw[conditions].mean(),
                "fraction_difference": raw[conditions].mean() - raw[~conditions].mean(),
                "arcsin_sqrt_effect": observed,
                "exact_permutation_p": p_value,
                "n_label_assignments": len(assignments),
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_bh"] = benjamini_hochberg(result["exact_permutation_p"])
    return result.sort_values("exact_permutation_p").reset_index(drop=True)


def composition_permanova(matrix, pseudocount=1e-4):
    """Exact two-group PERMANOVA on centered-log-ratio composition."""
    values = np.log(matrix.to_numpy(dtype=float) + pseudocount)
    values -= values.mean(axis=1, keepdims=True)
    observed_labels = np.asarray(
        matrix.index.get_level_values("condition") == "Endometriosis", dtype=bool
    )

    def r_squared(labels):
        grand = values.mean(axis=0)
        total = np.square(values - grand).sum()
        between = sum(
            np.sum(labels == group) * np.square(values[labels == group].mean(axis=0) - grand).sum()
            for group in (False, True)
        )
        return between / total

    observed = r_squared(observed_labels)
    null = np.array(
        [r_squared(a) for a in _condition_assignments(len(matrix), int(observed_labels.sum()))]
    )
    return {
        "n_patients": len(matrix),
        "n_controls": int((~observed_labels).sum()),
        "n_endometriosis": int(observed_labels.sum()),
        "clr_permanova_r_squared": observed,
        "exact_permutation_p": float(np.mean(null >= observed - 1e-12)),
        "n_label_assignments": len(null),
        "pseudocount": pseudocount,
    }
