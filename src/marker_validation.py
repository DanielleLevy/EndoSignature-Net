"""Marker-concordance checks for transferred GSE213216 cell-family labels.

This stage checks whether each transferred label is supported by expression of
curated core markers in the external dataset. It is a concordance analysis, not
independent ground truth, because related marker knowledge was also used to
define the provisional discovery labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


CORE_MARKER_PANELS: dict[str, tuple[str, ...]] = {
    "Epithelial": ("EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "KRT7"),
    "Stromal_fibroblast": ("COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1"),
    "Endothelial": ("PECAM1", "VWF", "KDR", "EMCN", "CLDN5", "RAMP2"),
    "Smooth_muscle_pericyte": ("ACTA2", "TAGLN", "MYL9", "RGS5", "MCAM", "CSPG4"),
    "T_cell": ("CD3D", "CD3E", "TRAC", "IL7R", "LTB"),
    "NK_cell": ("NKG7", "GNLY", "PRF1", "KLRD1", "GZMB", "CTSW"),
    "B_cell": ("MS4A1", "CD79A", "CD79B", "CD37", "CD74"),
    "Myeloid": ("LST1", "TYROBP", "FCER1G", "CTSS", "AIF1", "LYZ"),
    "Macrophage": ("CD68", "CD163", "C1QA", "C1QB", "APOE", "MRC1"),
    "Dendritic": ("FCER1A", "CD1C", "CLEC10A", "CST3", "HLA-DRA"),
    "Mast_cell": ("KIT", "CPA3", "MS4A2", "HDC", "GATA2"),
    "Cycling": ("MKI67", "TOP2A", "UBE2C", "PBK", "KIF20A", "STMN1", "TYMS"),
}


@dataclass(frozen=True)
class MarkerValidationSettings:
    """Predeclared rules for cell-level marker concordance."""

    mapping_score_threshold: float = 0.60
    minimum_detected_marker_fraction: float = 0.25
    minimum_detected_markers: int = 2
    minimum_panel_margin: float = 0.0

    def validate(self) -> None:
        if not 0 < self.mapping_score_threshold < 1:
            raise ValueError("mapping_score_threshold must be between 0 and 1")
        if not 0 < self.minimum_detected_marker_fraction <= 1:
            raise ValueError("minimum_detected_marker_fraction must be in (0, 1]")
        if self.minimum_detected_markers < 1:
            raise ValueError("minimum_detected_markers must be positive")


def _dense_column(matrix: object) -> np.ndarray:
    """Convert a one-gene sparse or dense slice to a flat float array."""

    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray()).ravel().astype(np.float64)
    return np.asarray(matrix).ravel().astype(np.float64)


def marker_panel_scores(
    adata: ad.AnnData,
    panels: dict[str, tuple[str, ...]] = CORE_MARKER_PANELS,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Calculate per-cell panel scores from within-sample standardized genes."""

    available = {
        family: [gene for gene in genes if gene in adata.var_names]
        for family, genes in panels.items()
    }
    scores: dict[str, np.ndarray] = {}
    for family, genes in available.items():
        standardized: list[np.ndarray] = []
        for gene in genes:
            values = _dense_column(adata[:, gene].X)
            standard_deviation = values.std(ddof=0)
            standardized.append(
                (values - values.mean()) / standard_deviation
                if standard_deviation > 0
                else np.zeros_like(values)
            )
        scores[family] = (
            np.mean(standardized, axis=0)
            if standardized
            else np.full(adata.n_obs, -np.inf)
        )
    return pd.DataFrame(scores, index=adata.obs_names), available


def add_marker_validation(
    adata: ad.AnnData,
    settings: MarkerValidationSettings,
    panels: dict[str, tuple[str, ...]] = CORE_MARKER_PANELS,
) -> ad.AnnData:
    """Attach marker agreement and conservative final-status fields."""

    settings.validate()
    result = adata.copy()
    panel_scores, available = marker_panel_scores(result, panels)
    predicted = result.obs["transferred_cell_family"].astype(str).to_numpy()
    score_array = panel_scores.to_numpy()
    top_indices = np.argmax(score_array, axis=1)
    ordered = np.sort(score_array, axis=1)
    top_families = panel_scores.columns.to_numpy()[top_indices]
    margins = ordered[:, -1] - ordered[:, -2]

    counts = result.layers["counts"]
    detected_counts = np.zeros(result.n_obs, dtype=int)
    required_counts = np.zeros(result.n_obs, dtype=int)
    predicted_panel_scores = np.zeros(result.n_obs, dtype=float)
    for family, genes in available.items():
        mask = predicted == family
        if not mask.any():
            continue
        required = max(
            settings.minimum_detected_markers,
            ceil(settings.minimum_detected_marker_fraction * len(genes)),
        )
        required_counts[mask] = min(required, len(genes))
        family_counts = counts[:, [result.var_names.get_loc(gene) for gene in genes]]
        detected = np.asarray((family_counts > 0).sum(axis=1)).ravel()
        detected_counts[mask] = detected[mask]
        predicted_panel_scores[mask] = panel_scores[family].to_numpy()[mask]

    has_markers = detected_counts >= required_counts
    panel_agreement = predicted == top_families
    adequate_margin = margins >= settings.minimum_panel_margin
    high_mapping = (
        result.obs["mapping_confidence"].to_numpy()
        >= settings.mapping_score_threshold
    )
    support_count = (
        has_markers.astype(int)
        + panel_agreement.astype(int)
        + high_mapping.astype(int)
    )
    status = np.where(
        has_markers & panel_agreement & adequate_margin & high_mapping,
        "marker_supported",
        np.where(support_count >= 2, "partial_support", "uncertain"),
    )
    final_family = np.where(status == "marker_supported", predicted, "Uncertain")

    result.obs["marker_top_family"] = pd.Categorical(top_families)
    result.obs["predicted_panel_score"] = predicted_panel_scores
    result.obs["marker_panel_margin"] = margins
    result.obs["detected_predicted_markers"] = detected_counts
    result.obs["required_predicted_markers"] = required_counts
    result.obs["marker_panel_agreement"] = panel_agreement
    result.obs["marker_validation_status"] = pd.Categorical(
        status, categories=["marker_supported", "partial_support", "uncertain"]
    )
    result.obs["conservative_cell_family"] = pd.Categorical(final_family)
    result.uns["marker_validation"] = {
        "interpretation": "external_marker_concordance_not_independent_ground_truth",
        "panels": {family: list(genes) for family, genes in panels.items()},
        "settings": settings.__dict__,
    }
    return result


def cell_validation_frame(adata: ad.AnnData) -> pd.DataFrame:
    """Return compact cell-level marker-validation results."""

    columns = [
        "sample_id",
        "patient_id",
        "condition",
        "tissue_code",
        "transferred_cell_family",
        "mapping_confidence",
        "marker_top_family",
        "predicted_panel_score",
        "marker_panel_margin",
        "detected_predicted_markers",
        "required_predicted_markers",
        "marker_panel_agreement",
        "marker_validation_status",
        "conservative_cell_family",
    ]
    result = adata.obs[columns].copy()
    result.insert(0, "cell_id", adata.obs_names)
    return result.reset_index(drop=True)


def validation_summaries(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize support by transferred family, tissue and sample."""

    cells = cells.copy()
    cells["is_marker_supported"] = (
        cells["marker_validation_status"] == "marker_supported"
    )
    family = (
        cells.groupby("transferred_cell_family", observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            n_samples=("sample_id", "nunique"),
            n_patients=("patient_id", "nunique"),
            marker_supported_pct=("is_marker_supported", lambda x: 100 * x.mean()),
            panel_agreement_pct=("marker_panel_agreement", lambda x: 100 * x.mean()),
            median_detected_markers=("detected_predicted_markers", "median"),
        )
        .reset_index()
    )
    tissue = (
        cells.groupby(["tissue_code", "transferred_cell_family"], observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            n_samples=("sample_id", "nunique"),
            marker_supported_pct=("is_marker_supported", lambda x: 100 * x.mean()),
            panel_agreement_pct=("marker_panel_agreement", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )
    sample = (
        cells.groupby(["sample_id", "patient_id", "condition", "tissue_code"], observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            marker_supported_pct=("is_marker_supported", lambda x: 100 * x.mean()),
            panel_agreement_pct=("marker_panel_agreement", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )
    return family, tissue, sample


def plot_validation_summaries(
    family: pd.DataFrame,
    tissue: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save family-level support and tissue-by-family heatmaps."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = family.sort_values("marker_supported_pct")
    fig, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=ordered,
        x="marker_supported_pct",
        y="transferred_cell_family",
        color="#4C78A8",
        ax=axis,
    )
    axis.set_title("External marker support for transferred cell families")
    axis.set_xlabel("Marker-supported cells (%)")
    axis.set_ylabel("Transferred family")
    fig.tight_layout()
    fig.savefig(output_dir / "marker_support_by_family.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot = tissue.pivot(
        index="transferred_cell_family",
        columns="tissue_code",
        values="marker_supported_pct",
    )
    fig, axis = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=100, annot=True, fmt=".1f", ax=axis)
    axis.set_title("Marker-supported transferred labels by tissue")
    axis.set_xlabel("Tissue")
    axis.set_ylabel("Transferred family")
    fig.tight_layout()
    fig.savefig(output_dir / "marker_support_by_tissue_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
