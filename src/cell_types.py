"""Discovery-cohort clustering and provisional broad cell-family annotation.

Annotations are intentionally labeled provisional. They are based on curated
marker panels and cluster-level normalized expression in GSE179640 only; they
must be reviewed against differential markers and validated in GSE213216.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


MARKER_PANELS: dict[str, tuple[str, ...]] = {
    "Epithelial": ("EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "KRT7"),
    "Stromal_fibroblast": ("COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1"),
    "Endothelial": ("PECAM1", "VWF", "KDR", "EMCN", "CLDN5", "RAMP2"),
    "Smooth_muscle_pericyte": ("ACTA2", "TAGLN", "MYL9", "RGS5", "MCAM", "CSPG4"),
    "T_cell": ("CD3D", "CD3E", "TRAC", "IL7R", "LTB", "MALAT1"),
    "NK_cell": ("NKG7", "GNLY", "PRF1", "KLRD1", "GZMB", "CTSW"),
    "B_cell": ("MS4A1", "CD79A", "CD79B", "CD37", "CD74", "HLA-DRA"),
    "Myeloid": ("LST1", "TYROBP", "FCER1G", "CTSS", "AIF1", "LYZ"),
    "Macrophage": ("CD68", "CD163", "C1QA", "C1QB", "APOE", "MRC1"),
    "Dendritic": ("FCER1A", "CD1C", "CLEC10A", "CST3", "HLA-DRA", "CD74"),
    "Mast_cell": ("KIT", "CPA3", "MS4A2", "HDC", "GATA2"),
    "Cycling": ("MKI67", "TOP2A", "UBE2C", "PBK", "KIF20A", "STMN1", "TYMS"),
}


@dataclass(frozen=True)
class ClusteringSettings:
    """Reproducible clustering, marker-ranking and annotation settings."""

    resolutions: tuple[float, ...] = (0.5, 0.8, 1.0)
    primary_resolution: float = 0.8
    random_state: int = 0
    top_markers_per_cluster: int = 50
    within_family_neighbors: int = 15

    def validate(self) -> None:
        if not self.resolutions or any(value <= 0 for value in self.resolutions):
            raise ValueError("Leiden resolutions must be positive")
        if self.primary_resolution not in self.resolutions:
            raise ValueError("primary_resolution must be included in resolutions")
        if self.top_markers_per_cluster <= 0 or self.within_family_neighbors <= 1:
            raise ValueError("Marker and neighbor settings must be positive")

    @property
    def primary_key(self) -> str:
        return leiden_key(self.primary_resolution)


def leiden_key(resolution: float) -> str:
    """Return a stable observation-column name for a Leiden resolution."""

    return f"leiden_{resolution:g}".replace(".", "_")


def run_leiden(adata: ad.AnnData, settings: ClusteringSettings) -> ad.AnnData:
    """Run Leiden at all requested resolutions on the existing neighbor graph."""

    settings.validate()
    result = adata.copy()
    for resolution in settings.resolutions:
        sc.tl.leiden(
            result,
            resolution=resolution,
            key_added=leiden_key(resolution),
            random_state=settings.random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    return result


def available_marker_panels(var_names: Sequence[str]) -> dict[str, list[str]]:
    """Return marker panels restricted to genes present in the atlas."""

    available = set(var_names)
    return {family: [gene for gene in markers if gene in available] for family, markers in MARKER_PANELS.items()}


def _cluster_gene_mean(adata: ad.AnnData, mask: np.ndarray, genes: Sequence[str]) -> float:
    matrix = adata[mask, list(genes)].X
    return float(matrix.mean()) if sparse.issparse(matrix) else float(np.mean(matrix))


def provisional_cluster_annotations(
    adata: ad.AnnData,
    cluster_key: str,
    panels: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Assign broad provisional labels from standardized marker-panel means."""

    clusters = sorted(adata.obs[cluster_key].astype(str).unique(), key=lambda value: int(value))
    expression_rows = []
    cluster_labels = adata.obs[cluster_key].astype(str).to_numpy()
    for cluster in clusters:
        mask = cluster_labels == cluster
        row: dict[str, object] = {"cluster": cluster, "n_cells": int(mask.sum())}
        for family, genes in panels.items():
            row[family] = _cluster_gene_mean(adata, mask, genes) if genes else np.nan
        expression_rows.append(row)

    expression = pd.DataFrame(expression_rows).set_index("cluster")
    score_columns = list(panels)
    standardized = expression[score_columns].apply(
        lambda column: (column - column.mean()) / column.std(ddof=0) if column.std(ddof=0) > 0 else 0.0
    )
    rows = []
    for cluster in expression.index:
        scores = standardized.loc[cluster].sort_values(ascending=False)
        top_family = str(scores.index[0])
        margin = float(scores.iloc[0] - scores.iloc[1]) if len(scores) > 1 else np.nan
        rows.append(
            {
                "cluster": cluster,
                "n_cells": int(expression.loc[cluster, "n_cells"]),
                "provisional_cell_family": top_family,
                "top_panel_zscore": float(scores.iloc[0]),
                "annotation_margin": margin,
                "review_priority": "high" if float(scores.iloc[0]) < 0.5 or margin < 0.25 else "standard",
                "annotation_status": "provisional_marker_panel_assignment",
                "validation_scope": "GSE179640_discovery_only",
            }
        )
    return pd.DataFrame(rows)


def add_cluster_annotations(adata: ad.AnnData, cluster_key: str, annotations: pd.DataFrame) -> ad.AnnData:
    """Map cluster-level provisional labels to cells."""

    result = adata.copy()
    mapping = annotations.set_index("cluster")["provisional_cell_family"].to_dict()
    labels = result.obs[cluster_key].astype(str).map(mapping)
    if labels.isna().any():
        raise ValueError("Every cluster must receive a provisional annotation")
    result.obs["provisional_cell_family"] = pd.Categorical(labels)
    result.obs["annotation_scope"] = "GSE179640_discovery_only"
    return result


def rank_cluster_markers(
    adata: ad.AnnData,
    cluster_key: str,
    settings: ClusteringSettings,
) -> pd.DataFrame:
    """Rank exploratory cluster markers using the full normalized gene space."""

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method="wilcoxon",
        n_genes=settings.top_markers_per_cluster,
        use_raw=False,
        pts=True,
        tie_correct=True,
    )
    result = sc.get.rank_genes_groups_df(adata, group=None)
    result.insert(0, "study_id", "GSE179640")
    result.insert(1, "analysis_scope", "discovery_only_exploratory_markers")
    return result


def cluster_composition_tables(
    adata: ad.AnnData,
    cluster_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw counts, within-sample fractions and patient-balanced summaries."""

    counts = (
        adata.obs.groupby(["sample_id", "patient_id", "condition", cluster_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = counts.groupby("sample_id", observed=True)["n_cells"].transform("sum")
    counts["fraction_within_sample"] = counts["n_cells"] / totals
    counts["study_id"] = "GSE179640"
    counts["analysis_scope"] = "discovery_only"

    patient_fractions = counts.rename(columns={cluster_key: "cluster"}).copy()
    condition_summary = (
        patient_fractions.groupby(["condition", "cluster"], observed=True)["fraction_within_sample"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_patient_fraction",
                "median": "median_patient_fraction",
                "std": "sd_patient_fraction",
                "count": "n_patients",
            }
        )
    )
    condition_summary["study_id"] = "GSE179640"
    condition_summary["analysis_scope"] = "descriptive_not_inferential"
    return counts, patient_fractions, condition_summary


def family_composition(adata: ad.AnnData) -> pd.DataFrame:
    """Compute provisional family fractions within each patient/sample."""

    result = (
        adata.obs.groupby(["sample_id", "patient_id", "condition", "provisional_cell_family"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    result["fraction_within_sample"] = result["n_cells"] / result.groupby("sample_id", observed=True)["n_cells"].transform("sum")
    result["study_id"] = "GSE179640"
    result["annotation_status"] = "provisional"
    return result


def within_family_mixing(adata: ad.AnnData, settings: ClusteringSettings) -> pd.DataFrame:
    """Measure sample mixing within each provisional cell family in PCA space."""

    rows = []
    for family in adata.obs["provisional_cell_family"].cat.categories:
        mask = adata.obs["provisional_cell_family"].to_numpy() == family
        indices = np.flatnonzero(mask)
        if len(indices) <= settings.within_family_neighbors:
            continue
        coordinates = adata.obsm["X_pca"][indices]
        labels = adata.obs["sample_id"].astype(str).to_numpy()[indices]
        n_neighbors = min(settings.within_family_neighbors + 1, len(indices))
        neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(coordinates).kneighbors(return_distance=False)
        neighbor_labels = labels[neighbors[:, 1:]]
        observed = float(np.mean(neighbor_labels == labels[:, None]))
        frequencies = pd.Series(labels).value_counts(normalize=True).to_numpy()
        rows.append(
            {
                "study_id": "GSE179640",
                "provisional_cell_family": family,
                "n_cells": len(indices),
                "n_samples": len(np.unique(labels)),
                "observed_same_sample_neighbor_fraction": observed,
                "random_mixing_expectation": float(np.square(frequencies).sum()),
                "mixing_excess": observed - float(np.square(frequencies).sum()),
                "interpretation": "descriptive_only_provisional_annotation",
            }
        )
    return pd.DataFrame(rows)


def plot_cluster_outputs(
    adata: ad.AnnData,
    cluster_key: str,
    panels: Mapping[str, Sequence[str]],
    family_counts: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save cluster, provisional annotation, marker and composition figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for color in (cluster_key, "provisional_cell_family", "condition", "sample_id"):
        sc.pl.umap(adata, color=color, show=False, frameon=False, title=f"UMAP by {color}")
        plt.gcf().savefig(output_dir / f"umap_by_{color}.png", dpi=180, bbox_inches="tight")
        plt.close(plt.gcf())

    nonempty_panels = {family: genes for family, genes in panels.items() if genes}
    sc.pl.dotplot(
        adata,
        var_names=nonempty_panels,
        groupby=cluster_key,
        standard_scale="var",
        show=False,
        title="Curated broad cell-family markers by cluster",
    )
    plt.gcf().savefig(output_dir / "marker_panel_dotplot.png", dpi=180, bbox_inches="tight")
    plt.close(plt.gcf())

    fig, axis = plt.subplots(figsize=(13, 7))
    plot_data = family_counts.copy()
    pivot = plot_data.pivot(index="sample_id", columns="provisional_cell_family", values="fraction_within_sample")
    pivot.plot(kind="bar", stacked=True, ax=axis, width=0.85)
    axis.tick_params(axis="x", rotation=75)
    axis.set_title("Provisional cell-family composition within each discovery sample")
    axis.set_xlabel("Sample")
    axis.set_ylabel("Fraction of sample cells")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "cell_family_composition_by_sample.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
