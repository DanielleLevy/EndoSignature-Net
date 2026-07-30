"""Quality-control and exploratory analysis utilities for scRNA-seq samples.

The functions in this module operate on one AnnData object at a time so the
complete cohort is never materialized as a dense matrix in memory.  Filtering
is performed on an in-memory copy and does not modify the downloaded inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


@dataclass(frozen=True)
class QCThresholds:
    """Explicit, serializable thresholds used for the initial QC pass."""

    min_genes_per_cell: int = 200
    max_genes_per_cell: int | None = None
    max_pct_mito: float = 20.0
    min_cells_per_gene: int = 3
    min_cells_after_qc: int = 200
    min_cell_retention_pct: float = 50.0

    def validate(self) -> None:
        """Raise ``ValueError`` when a threshold is internally inconsistent."""

        if self.min_genes_per_cell < 0 or self.min_cells_per_gene < 0:
            raise ValueError("Minimum-count thresholds cannot be negative")
        if self.max_genes_per_cell is not None and self.max_genes_per_cell <= self.min_genes_per_cell:
            raise ValueError("max_genes_per_cell must be greater than min_genes_per_cell")
        if not 0 <= self.max_pct_mito <= 100:
            raise ValueError("max_pct_mito must be between 0 and 100")
        if not 0 <= self.min_cell_retention_pct <= 100:
            raise ValueError("min_cell_retention_pct must be between 0 and 100")


def read_sample(row: Mapping[str, str]) -> ad.AnnData:
    """Read a 10x H5 or Matrix Market sample described by a metadata row."""

    file_format = row["file_format"]
    if file_format == "10x_h5":
        adata = sc.read_10x_h5(row["file_path"])
    elif "10x_mtx" in file_format:
        source_files = [Path(item) for item in row["source_files"].split(";")]
        matrix_files = [path for path in source_files if path.name.endswith("matrix.mtx.gz")]
        if len(matrix_files) != 1:
            raise ValueError(f"Expected one Matrix Market matrix for {row['sample_id']}")
        matrix_path = matrix_files[0]
        prefix = matrix_path.name[: -len("matrix.mtx.gz")]
        adata = sc.read_10x_mtx(matrix_path.parent, prefix=prefix, var_names="gene_symbols")
    else:
        raise ValueError(f"Unsupported single-cell format: {file_format}")

    adata.var_names_make_unique()
    return adata


def calculate_qc(adata: ad.AnnData) -> ad.AnnData:
    """Calculate standard per-cell and per-gene QC metrics in place."""

    # Case-insensitive matching supports common human mitochondrial gene naming.
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    return adata


def cell_filter_mask(adata: ad.AnnData, thresholds: QCThresholds) -> np.ndarray:
    """Return the cell-level QC mask without altering ``adata``."""

    thresholds.validate()
    mask = adata.obs["n_genes_by_counts"].to_numpy() >= thresholds.min_genes_per_cell
    if thresholds.max_genes_per_cell is not None:
        mask &= adata.obs["n_genes_by_counts"].to_numpy() <= thresholds.max_genes_per_cell
    mask &= adata.obs["pct_counts_mt"].to_numpy() <= thresholds.max_pct_mito
    return mask


def filter_sample(adata: ad.AnnData, thresholds: QCThresholds) -> ad.AnnData:
    """Return a filtered copy while preserving the original AnnData object."""

    filtered = adata[cell_filter_mask(adata, thresholds)].copy()
    sc.pp.filter_genes(filtered, min_cells=thresholds.min_cells_per_gene)
    return filtered


def summarize_sample(
    raw: ad.AnnData,
    filtered: ad.AnnData,
    metadata: Mapping[str, str],
    thresholds: QCThresholds,
) -> dict[str, object]:
    """Create one auditable sample-level summary row."""

    n_raw = int(raw.n_obs)
    n_filtered = int(filtered.n_obs)
    retention = 100.0 * n_filtered / n_raw if n_raw else 0.0
    flags = []
    if n_filtered < thresholds.min_cells_after_qc:
        flags.append("low_post_qc_cell_count")
    if retention < thresholds.min_cell_retention_pct:
        flags.append("low_cell_retention")
    if n_raw == 0:
        flags.append("empty_input")

    obs = raw.obs
    return {
        "sample_id": metadata["sample_id"],
        "study_id": metadata["study_id"],
        "patient_id": metadata["patient_id"],
        "condition": metadata["condition"],
        "tissue": metadata["tissue"],
        "tissue_code": metadata["tissue_code"],
        "n_cells_raw": n_raw,
        "n_cells_after_qc": n_filtered,
        "cell_retention_pct": retention,
        "n_genes_raw": int(raw.n_vars),
        "n_genes_after_qc": int(filtered.n_vars),
        "median_genes_per_cell": float(obs["n_genes_by_counts"].median()),
        "median_counts_per_cell": float(obs["total_counts"].median()),
        "median_pct_mito": float(obs["pct_counts_mt"].median()),
        "qc_status": "review" if flags else "pass_initial_qc",
        "qc_flags": ";".join(flags),
    }


def plot_sample_qc(adata: ad.AnnData, sample_id: str, output_path: Path) -> None:
    """Save a compact pre-filtering QC dashboard for one sample."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    metrics = (
        ("n_genes_by_counts", "Genes detected per cell"),
        ("total_counts", "Total counts per cell"),
        ("pct_counts_mt", "Mitochondrial counts (%)"),
    )
    for axis, (metric, title) in zip(axes[0], metrics):
        sns.violinplot(y=adata.obs[metric], ax=axis, color="#4C78A8", cut=0)
        axis.set_title(title)
        axis.set_xlabel("")

    sns.scatterplot(
        data=adata.obs,
        x="total_counts",
        y="n_genes_by_counts",
        hue="pct_counts_mt",
        palette="viridis",
        s=8,
        linewidth=0,
        ax=axes[1, 0],
        legend=False,
    )
    axes[1, 0].set_title("Library size vs detected genes")
    sns.scatterplot(
        data=adata.obs,
        x="n_genes_by_counts",
        y="pct_counts_mt",
        s=8,
        linewidth=0,
        color="#F58518",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Detected genes vs mitochondrial %")
    axes[1, 2].axis("off")
    fig.suptitle(f"Pre-filter QC: {sample_id}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_qc(summary: pd.DataFrame, output_dir: Path) -> None:
    """Save cohort-level sample size, retention and metric comparisons."""

    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = summary.sort_values("n_cells_raw", ascending=False)
    fig, axis = plt.subplots(figsize=(12, max(6, 0.28 * len(ordered))))
    sns.barplot(data=ordered, x="n_cells_raw", y="sample_id", hue="condition", dodge=False, ax=axis)
    axis.set_title("Cells per sample before secondary QC")
    axis.set_xlabel("Number of cells")
    axis.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(output_dir / "cells_per_sample.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=summary, x="sample_id", y="cell_retention_pct", hue="condition", dodge=False, ax=axis)
    axis.tick_params(axis="x", rotation=90)
    axis.set_title("Cell retention after secondary QC")
    axis.set_xlabel("Sample")
    axis.set_ylabel("Retained cells (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "cell_retention_per_sample.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    long_metrics = summary.melt(
        id_vars=["sample_id", "condition", "tissue_code"],
        value_vars=["median_genes_per_cell", "median_counts_per_cell", "median_pct_mito"],
        var_name="metric",
        value_name="value",
    )
    grid = sns.catplot(
        data=long_metrics,
        x="tissue_code",
        y="value",
        hue="condition",
        col="metric",
        kind="box",
        sharey=False,
        height=4,
        aspect=1.15,
    )
    grid.set_axis_labels("Tissue code", "Sample median")
    grid.fig.suptitle("Sample-level QC metrics by tissue", y=1.05)
    grid.savefig(output_dir / "qc_metrics_by_tissue.png", dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def write_thresholds(path: Path, thresholds: QCThresholds) -> None:
    """Persist exact QC settings for reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(thresholds), handle, indent=2, sort_keys=True)
        handle.write("\n")
