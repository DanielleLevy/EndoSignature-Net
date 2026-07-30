"""Metadata verification and exploratory QC for GSE135485 bulk RNA-seq."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class BulkEDASettings:
    """Predeclared filtering and outlier rules for bulk count EDA."""

    min_cpm: float = 1.0
    min_samples_expressing: int = 4
    robust_outlier_mad: float = 3.0
    min_median_sample_correlation: float = 0.60
    n_pcs: int = 10

    def validate(self) -> None:
        if self.min_cpm < 0 or self.min_samples_expressing < 1:
            raise ValueError("Expression filter settings are invalid")
        if self.robust_outlier_mad <= 0 or self.n_pcs < 2:
            raise ValueError("Outlier and PCA settings must be positive")
        if not -1 <= self.min_median_sample_correlation <= 1:
            raise ValueError("Correlation threshold must be between -1 and 1")


def parse_geo_soft_samples(path: Path) -> pd.DataFrame:
    """Parse sample titles and authoritative characteristics from GEO SOFT."""

    samples: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    characteristics: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current is not None:
                    current.update(characteristics)
                    samples.append(current)
                current = {"sample_id": line.split("=", 1)[1].strip()}
                characteristics = {}
            elif current is not None and line.startswith("!Sample_title = "):
                current["count_matrix_column"] = line.split("=", 1)[1].strip()
            elif current is not None and line.startswith("!Sample_source_name_ch1 = "):
                current["source_name"] = line.split("=", 1)[1].strip()
            elif current is not None and line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split("=", 1)[1].strip()
                if ":" in value:
                    key, item = value.split(":", 1)
                    characteristics[key.strip().lower().replace(" ", "_")] = item.strip()
    if current is not None:
        current.update(characteristics)
        samples.append(current)

    metadata = pd.DataFrame(samples)
    required = {"sample_id", "count_matrix_column", "subject_status", "tissue"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing GEO SOFT fields: {sorted(missing)}")
    metadata["study_id"] = "GSE135485"
    metadata["condition"] = metadata["subject_status"].map(
        {
            "patient with endometriosis": "Endometriosis",
            "healthy control": "Control",
        }
    )
    if metadata["condition"].isna().any():
        raise ValueError("Unrecognized subject status in GEO metadata")
    metadata["lane"] = metadata["count_matrix_column"].str.extract(
        r"_(L\d{3})_", expand=False
    ).fillna("not_encoded")
    metadata["metadata_source"] = "NCBI_GEO_SOFT"
    metadata["geo_sample_url"] = metadata["sample_id"].map(
        lambda value: f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={value}"
    )
    return metadata.sort_values("count_matrix_column").reset_index(drop=True)


def load_raw_count_matrix(path: Path) -> pd.DataFrame:
    """Load and validate the official non-negative integer count matrix."""

    counts = pd.read_csv(path, index_col=0)
    counts.index.name = "gene"
    if counts.index.duplicated().any() or counts.columns.duplicated().any():
        raise ValueError("Gene and sample identifiers must be unique")
    values = counts.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and non-negative")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("GSE135485 input is not an integer raw-count matrix")
    return counts


def align_metadata_to_counts(
    metadata: pd.DataFrame, counts: pd.DataFrame
) -> pd.DataFrame:
    """Require an exact one-to-one mapping between GEO samples and columns."""

    metadata_titles = set(metadata["count_matrix_column"])
    matrix_titles = set(counts.columns)
    if metadata_titles != matrix_titles:
        raise ValueError(
            "GEO/count mismatch: "
            f"{len(matrix_titles - metadata_titles)} matrix-only and "
            f"{len(metadata_titles - matrix_titles)} metadata-only titles"
        )
    return (
        metadata.set_index("count_matrix_column")
        .loc[counts.columns]
        .rename_axis("count_matrix_column")
        .reset_index()
    )


def _robust_z(values: pd.Series) -> pd.Series:
    """Return median/MAD robust z-scores with a zero-variance fallback."""

    median = values.median()
    mad = np.median(np.abs(values - median))
    if mad == 0:
        result = np.zeros(len(values), dtype=float)
        different = values.to_numpy() != median
        result[different] = np.sign(values.to_numpy()[different] - median) * np.inf
        return pd.Series(result, index=values.index)
    return 0.67448975 * (values - median) / mad


def sample_qc_metrics(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: BulkEDASettings,
) -> pd.DataFrame:
    """Calculate library, complexity and composition metrics per sample."""

    settings.validate()
    result = metadata.copy()
    result["library_size"] = counts.sum(axis=0).to_numpy()
    result["detected_genes"] = (counts > 0).sum(axis=0).to_numpy()
    result["zero_gene_fraction_pct"] = 100.0 * (counts == 0).mean(axis=0).to_numpy()

    upper_names = counts.index.str.upper()
    mito = upper_names.str.startswith("MT-")
    ribosomal = upper_names.str.match(r"^RP[SL]\d")
    library = result["library_size"].to_numpy()
    result["mitochondrial_count_pct"] = (
        100.0 * counts.loc[mito].sum(axis=0).to_numpy() / library
    )
    result["ribosomal_count_pct"] = (
        100.0 * counts.loc[ribosomal].sum(axis=0).to_numpy() / library
    )
    result["library_size_robust_z"] = _robust_z(
        np.log10(result["library_size"])
    ).to_numpy()
    result["detected_genes_robust_z"] = _robust_z(
        result["detected_genes"]
    ).to_numpy()
    flags: list[str] = []
    statuses: list[str] = []
    for _, row in result.iterrows():
        row_flags = []
        if abs(row["library_size_robust_z"]) > settings.robust_outlier_mad:
            row_flags.append("library_size_outlier")
        if abs(row["detected_genes_robust_z"]) > settings.robust_outlier_mad:
            row_flags.append("detected_gene_outlier")
        flags.append(";".join(row_flags))
        statuses.append("review" if row_flags else "pass_initial_qc")
    result["qc_flags"] = flags
    result["qc_status"] = statuses
    return result


def normalized_expression_and_pca(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: BulkEDASettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter genes, calculate log2-CPM and return PCA coordinates."""

    library = counts.sum(axis=0)
    cpm = counts.divide(library, axis=1) * 1_000_000.0
    keep = (cpm >= settings.min_cpm).sum(axis=1) >= settings.min_samples_expressing
    log_cpm = np.log2(cpm.loc[keep] + 0.5)
    sample_by_gene = log_cpm.T
    centered = sample_by_gene - sample_by_gene.mean(axis=0)
    n_components = min(settings.n_pcs, len(metadata) - 1, centered.shape[1])
    centered_values = centered.to_numpy(dtype=np.float64)
    if not np.isfinite(centered_values).all():
        raise ValueError("Non-finite values found before bulk PCA")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sample_gram = centered_values @ centered_values.T
    if not np.isfinite(sample_gram).all():
        raise ValueError("Non-finite sample Gram matrix produced before bulk PCA")
    eigenvalues, eigenvectors = np.linalg.eigh(sample_gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0, a_max=None)
    eigenvectors = eigenvectors[:, order]
    coordinates = eigenvectors[:, :n_components] * np.sqrt(
        eigenvalues[:n_components]
    )
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    pca = metadata.copy()
    for index in range(n_components):
        pca[f"PC{index + 1}"] = coordinates[:, index]
    variance = pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(n_components)],
            "explained_variance_ratio": explained_variance_ratio[:n_components],
        }
    )
    return log_cpm, pca, variance


def sample_correlation_summary(
    log_cpm: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Summarize each sample's correlation with the remaining cohort."""

    correlation = log_cpm.corr(method="spearman")
    rows = []
    for sample in correlation.columns:
        other = correlation.loc[sample].drop(sample)
        rows.append(
            {
                "count_matrix_column": sample,
                "median_spearman_correlation_to_other_samples": other.median(),
                "minimum_spearman_correlation_to_other_samples": other.min(),
            }
        )
    return metadata.merge(pd.DataFrame(rows), on="count_matrix_column", how="left")


def plot_bulk_eda(
    qc: pd.DataFrame,
    pca: pd.DataFrame,
    variance: pd.DataFrame,
    log_cpm: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save library, complexity, PCA and sample-correlation diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=qc, x="condition", y="library_size", ax=axes[0])
    sns.stripplot(data=qc, x="condition", y="library_size", color="black", size=4, ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_title("Raw library sizes")
    sns.boxplot(data=qc, x="condition", y="detected_genes", ax=axes[1])
    sns.stripplot(data=qc, x="condition", y="detected_genes", color="black", size=4, ax=axes[1])
    axes[1].set_title("Detected genes")
    fig.tight_layout()
    fig.savefig(output_dir / "library_size_and_complexity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for color in ("condition", "lane"):
        fig, axis = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=pca, x="PC1", y="PC2", hue=color, s=70, ax=axis)
        pc1 = 100 * variance.loc[0, "explained_variance_ratio"]
        pc2 = 100 * variance.loc[1, "explained_variance_ratio"]
        axis.set_xlabel(f"PC1 ({pc1:.1f}%)")
        axis.set_ylabel(f"PC2 ({pc2:.1f}%)")
        axis.set_title(f"GSE135485 log2-CPM PCA by {color}")
        fig.tight_layout()
        fig.savefig(output_dir / f"pca_by_{color}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    correlation = log_cpm.corr(method="spearman")
    annotation = qc.set_index("count_matrix_column").loc[correlation.index, "condition"]
    palette = {"Control": "#4C78A8", "Endometriosis": "#E45756"}
    colors = annotation.map(palette)
    grid = sns.clustermap(
        correlation,
        cmap="vlag",
        center=0,
        row_colors=colors,
        col_colors=colors,
        xticklabels=False,
        yticklabels=False,
        figsize=(10, 10),
    )
    grid.fig.suptitle("Spearman correlation of filtered log2-CPM samples", y=1.02)
    grid.savefig(output_dir / "sample_correlation_clustermap.png", dpi=180, bbox_inches="tight")
    plt.close(grid.fig)
