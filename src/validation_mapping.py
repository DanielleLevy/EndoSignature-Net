"""Patient-aware reference mapping for the GSE213216 validation cohort.

The provisional GSE179640 broad cell-family labels are learned in its existing
30-dimensional PCA space. A class-balanced linear classifier limits dominance
by abundant cell families. Generalization is measured with patient-grouped
cross-validation before the model is fitted to the complete discovery atlas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MappingSettings:
    """Reproducible classifier, confidence and preprocessing settings."""

    target_sum: float = 10_000.0
    n_splits: int = 5
    max_iter: int = 1_000
    confidence_threshold: float = 0.60
    random_state: int = 0

    def validate(self) -> None:
        if self.target_sum <= 0 or self.n_splits < 2 or self.max_iter <= 0:
            raise ValueError("Invalid mapping preprocessing or model settings")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")


def build_classifier(settings: MappingSettings) -> Pipeline:
    """Return a class-balanced linear label-transfer classifier."""

    settings.validate()
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=settings.max_iter,
                    random_state=settings.random_state,
                ),
            ),
        ]
    )


def patient_grouped_reference_cv(
    coordinates: np.ndarray,
    labels: Sequence[str],
    patient_ids: Sequence[str],
    settings: MappingSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate label recovery without sharing patients across folds."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=str)
    groups = np.asarray(patient_ids, dtype=str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < settings.n_splits:
        raise ValueError("Not enough discovery patients for grouped cross-validation")

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    splitter = GroupKFold(n_splits=settings.n_splits)
    for fold, (train, test) in enumerate(
        splitter.split(coordinates, labels_array, groups), start=1
    ):
        model = build_classifier(settings)
        model.fit(coordinates[train], labels_array[train])
        predicted = model.predict(coordinates[test])
        probabilities = model.predict_proba(coordinates[test])
        fold_rows.append(
            {
                "fold": fold,
                "n_train_cells": len(train),
                "n_test_cells": len(test),
                "n_train_patients": len(np.unique(groups[train])),
                "n_test_patients": len(np.unique(groups[test])),
                "macro_f1": f1_score(labels_array[test], predicted, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(labels_array[test], predicted),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "patient_id": groups[test],
                    "true_family": labels_array[test],
                    "predicted_family": predicted,
                    "prediction_confidence": probabilities.max(axis=1),
                }
            )
        )
    return pd.DataFrame(fold_rows), pd.concat(prediction_rows, ignore_index=True)


def prepare_query_for_reference(
    query: ad.AnnData,
    reference_var_names: Sequence[str],
    reference_hvg_mask: np.ndarray,
    reference_loadings: np.ndarray,
    settings: MappingSettings,
) -> ad.AnnData:
    """Align, normalize and project one validation sample into reference PCA."""

    reference_index = pd.Index(reference_var_names)
    common = reference_index.intersection(query.var_names, sort=False)
    reference_positions = reference_index.get_indexer(common)
    common_hvg = reference_hvg_mask[reference_positions]
    if int(common_hvg.sum()) < reference_loadings.shape[1]:
        raise ValueError("Too few reference HVGs remain after query QC")

    result = query[:, common].copy()
    result.layers["counts"] = result.X.copy()
    sc.pp.normalize_total(result, target_sum=settings.target_sum)
    sc.pp.log1p(result)
    hvg_matrix = result[:, common_hvg].X
    hvg_reference_positions = reference_positions[common_hvg]
    coordinates = hvg_matrix @ reference_loadings[hvg_reference_positions]
    result.obsm["X_pca_reference"] = np.asarray(coordinates, dtype=np.float32)
    result.uns["reference_mapping"] = {
        "reference_study": "GSE179640",
        "reference_label": "provisional_cell_family",
        "projection": "GSE179640_HVG_PCA_loadings",
        "n_reference_genes": len(reference_index),
        "n_common_genes": len(common),
        "n_reference_hvgs": int(reference_hvg_mask.sum()),
        "n_common_hvgs": int(common_hvg.sum()),
    }
    return result


def add_transferred_labels(
    query: ad.AnnData,
    classifier: Pipeline,
    settings: MappingSettings,
) -> ad.AnnData:
    """Predict broad families and attach an uncalibrated model score."""

    result = query.copy()
    coordinates = np.asarray(result.obsm["X_pca_reference"], dtype=np.float64)
    probabilities = classifier.predict_proba(coordinates)
    predicted = classifier.classes_[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    result.obs["transferred_cell_family"] = pd.Categorical(
        predicted, categories=classifier.classes_
    )
    result.obs["mapping_confidence"] = confidence
    result.obs["mapping_confidence_status"] = pd.Categorical(
        np.where(
            confidence >= settings.confidence_threshold,
            "high_confidence",
            "low_confidence",
        )
    )
    result.obs["annotation_status"] = "transferred_from_provisional_GSE179640"
    return result


def mapping_sample_summary(adata: ad.AnnData) -> dict[str, object]:
    """Summarize transferred labels and confidence for one sample."""

    confidence = adata.obs["mapping_confidence"].to_numpy()
    high = adata.obs["mapping_confidence_status"].astype(str) == "high_confidence"
    return {
        "sample_id": str(adata.obs["sample_id"].iloc[0]),
        "patient_id": str(adata.obs["patient_id"].iloc[0]),
        "condition": str(adata.obs["condition"].iloc[0]),
        "tissue_code": str(adata.obs["tissue_code"].iloc[0]),
        "n_mapped_singlets": int(adata.n_obs),
        "median_mapping_confidence": float(np.median(confidence)),
        "high_confidence_cells": int(high.sum()),
        "high_confidence_pct": 100.0 * float(high.mean()),
        "n_common_reference_genes": int(
            adata.uns["reference_mapping"]["n_common_genes"]
        ),
        "n_common_reference_hvgs": int(
            adata.uns["reference_mapping"]["n_common_hvgs"]
        ),
    }


def family_mapping_summary(cell_predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize confidence by transferred family without pooling patients."""

    return (
        cell_predictions.groupby("transferred_cell_family", observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            n_samples=("sample_id", "nunique"),
            n_patients=("patient_id", "nunique"),
            median_confidence=("mapping_confidence", "median"),
            high_confidence_pct=("is_high_confidence", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )


def is_sparse(adata: ad.AnnData) -> bool:
    """Return whether the mapped expression matrix remains sparse."""

    return sparse.issparse(adata.X)
