"""Primary-cohort atlas construction and exploratory batch diagnostics.

The atlas is built from independently QC-filtered samples. Raw counts are kept
in a layer, doublet predictions are joined by sample-qualified cell ID, and
highly variable genes are selected only after samples share one gene space.
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
from sklearn.metrics import silhouette_score

from src.eda import QCThresholds, calculate_qc, filter_sample, read_sample


@dataclass(frozen=True)
class AtlasSettings:
    """Reproducible preprocessing and embedding settings."""

    target_sum: float = 10_000.0
    n_top_genes: int = 2_000
    n_pcs: int = 30
    n_neighbors: int = 15
    random_state: int = 0
    silhouette_sample_size: int = 10_000

    def validate(self) -> None:
        if self.target_sum <= 0:
            raise ValueError("target_sum must be positive")
        if self.n_top_genes <= 0 or self.n_pcs <= 1 or self.n_neighbors <= 1:
            raise ValueError("Feature, PC and neighbor counts must be positive and meaningful")


def attach_metadata(adata: ad.AnnData, row: Mapping[str, str]) -> ad.AnnData:
    """Attach sample-level metadata and globally unique cell identifiers."""

    result = adata.copy()
    sample_id = row["sample_id"]
    result.obs_names = [f"{sample_id}:{barcode}" for barcode in result.obs_names]
    result.obs["sample_id"] = sample_id
    result.obs["patient_id"] = row["patient_id"]
    result.obs["condition"] = row["condition"]
    result.obs["tissue_code"] = row["tissue_code"]
    return result


def apply_doublet_predictions(adata: ad.AnnData, predictions: pd.DataFrame) -> tuple[ad.AnnData, dict[str, int]]:
    """Join Scrublet predictions exactly by cell ID and return predicted singlets."""

    required = {"cell_id", "doublet_score", "predicted_doublet"}
    missing_columns = required - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"Missing doublet columns: {sorted(missing_columns)}")
    if predictions["cell_id"].duplicated().any():
        raise ValueError("Doublet prediction IDs must be unique")

    indexed = predictions.set_index("cell_id")
    missing_cells = adata.obs_names.difference(indexed.index)
    if len(missing_cells):
        raise ValueError(f"Missing doublet predictions for {len(missing_cells)} QC-filtered cells")

    aligned = indexed.loc[adata.obs_names]
    adata.obs["doublet_score"] = aligned["doublet_score"].to_numpy(dtype=float)
    adata.obs["predicted_doublet"] = aligned["predicted_doublet"].astype(bool).to_numpy()
    keep = ~adata.obs["predicted_doublet"].to_numpy()
    report = {
        "n_cells_after_qc": int(adata.n_obs),
        "n_predicted_doublets_removed": int((~keep).sum()),
        "n_cells_in_atlas": int(keep.sum()),
    }
    return adata[keep].copy(), report


def load_primary_sample(
    row: Mapping[str, str],
    qc_thresholds: QCThresholds,
    prediction_path: Path,
) -> tuple[ad.AnnData, dict[str, object]]:
    """Load, QC, annotate and remove predicted doublets from one sample."""

    raw = calculate_qc(read_sample(row))
    filtered = attach_metadata(filter_sample(raw, qc_thresholds), row)
    predictions = pd.read_csv(prediction_path)
    singlets, counts = apply_doublet_predictions(filtered, predictions)
    return singlets, {"sample_id": row["sample_id"], **counts}


def concatenate_samples(samples: Sequence[ad.AnnData]) -> ad.AnnData:
    """Concatenate samples using genes present in every primary sample."""

    if not samples:
        raise ValueError("At least one sample is required")
    combined = ad.concat(samples, axis=0, join="inner", merge="same")
    combined.obs_names_make_unique()
    combined.var_names_make_unique()
    combined.obs["condition"] = pd.Categorical(combined.obs["condition"], categories=["Control", "Endometriosis"])
    for column in ("sample_id", "patient_id", "tissue_code"):
        combined.obs[column] = combined.obs[column].astype("category")
    return combined


def preprocess_atlas(adata: ad.AnnData, settings: AtlasSettings) -> ad.AnnData:
    """Normalize, select batch-aware HVGs, compute PCA, neighbors and UMAP."""

    settings.validate()
    result = adata.copy()
    result.layers["counts"] = result.X.copy()
    sc.pp.normalize_total(result, target_sum=settings.target_sum)
    sc.pp.log1p(result)
    sc.pp.highly_variable_genes(
        result,
        n_top_genes=min(settings.n_top_genes, result.n_vars),
        flavor="seurat",
        batch_key="sample_id",
        subset=False,
    )
    n_hvg = int(result.var["highly_variable"].sum())
    if n_hvg < settings.n_pcs:
        raise ValueError(f"Only {n_hvg} HVGs are available for {settings.n_pcs} PCs")

    sc.tl.pca(
        result,
        n_comps=settings.n_pcs,
        mask_var="highly_variable",
        zero_center=False,
        svd_solver="arpack",
        random_state=settings.random_state,
    )
    sc.pp.neighbors(result, n_neighbors=settings.n_neighbors, n_pcs=settings.n_pcs, random_state=settings.random_state)
    sc.tl.umap(result, random_state=settings.random_state)
    return result


def hvg_atlas(adata: ad.AnnData) -> ad.AnnData:
    """Return a compact HVG-only copy with count and normalized values."""

    return adata[:, adata.var["highly_variable"].to_numpy()].copy()


def _same_label_neighbor_fraction(adata: ad.AnnData, labels: np.ndarray) -> float:
    distances = adata.obsp["distances"].tocsr()
    fractions = []
    for cell_index in range(adata.n_obs):
        neighbors = distances.indices[distances.indptr[cell_index] : distances.indptr[cell_index + 1]]
        if len(neighbors):
            fractions.append(float(np.mean(labels[neighbors] == labels[cell_index])))
    return float(np.mean(fractions)) if fractions else float("nan")


def batch_diagnostics(adata: ad.AnnData, settings: AtlasSettings) -> pd.DataFrame:
    """Compute descriptive mixing metrics on the uncorrected atlas.

    These metrics quantify structure but cannot distinguish technical batch from
    real biological composition differences. They are not correction scores.
    """

    rng = np.random.default_rng(settings.random_state)
    sample_size = min(settings.silhouette_sample_size, adata.n_obs)
    indices = rng.choice(adata.n_obs, size=sample_size, replace=False)
    pcs = adata.obsm["X_pca"][indices]
    rows = []
    for column in ("sample_id", "patient_id", "condition"):
        labels = adata.obs[column].astype(str).to_numpy()
        frequencies = pd.Series(labels).value_counts(normalize=True)
        rows.append(
            {
                "label": column,
                "n_groups": int(len(frequencies)),
                "silhouette_pca": float(silhouette_score(pcs, labels[indices])) if len(frequencies) > 1 else np.nan,
                "observed_same_label_neighbor_fraction": _same_label_neighbor_fraction(adata, labels),
                "random_mixing_expectation": float(np.square(frequencies.to_numpy()).sum()),
                "interpretation": "descriptive_only_biology_and_batch_are_conflated",
            }
        )
    return pd.DataFrame(rows)


def sample_composition(adata: ad.AnnData) -> pd.DataFrame:
    """Summarize final atlas cell counts by sample and condition."""

    result = (
        adata.obs.groupby(["sample_id", "patient_id", "condition"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    total = result["n_cells"].sum()
    result["cell_fraction_pct"] = 100.0 * result["n_cells"] / total
    return result


def plot_embeddings(adata: ad.AnnData, output_dir: Path) -> None:
    """Save PCA and UMAP views colored by biological and technical metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for basis in ("pca", "umap"):
        for color in ("condition", "sample_id", "patient_id"):
            sc.pl.embedding(
                adata,
                basis=basis,
                color=color,
                show=False,
                frameon=False,
                title=f"{basis.upper()} by {color}",
            )
            plt.gcf().savefig(output_dir / f"{basis}_by_{color}.png", dpi=180, bbox_inches="tight")
            plt.close(plt.gcf())


def plot_atlas_composition(composition: pd.DataFrame, output_path: Path) -> None:
    """Save sample contributions to the final primary atlas."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=composition, x="sample_id", y="n_cells", hue="condition", dodge=False, ax=axis)
    axis.tick_params(axis="x", rotation=75)
    axis.set_title("Primary atlas cell composition")
    axis.set_xlabel("Sample")
    axis.set_ylabel("Singlet cells after QC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def matrix_is_sparse(adata: ad.AnnData) -> bool:
    """Expose a simple invariant used by tests and run-time reporting."""

    return sparse.issparse(adata.X)
