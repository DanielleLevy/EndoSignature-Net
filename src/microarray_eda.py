"""Verified metadata, probe mapping and processed-data EDA for GSE51981."""

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class MicroarrayEDASettings:
    """Predeclared processed-expression QC and PCA settings."""

    robust_outlier_mad: float = 3.0
    min_median_sample_correlation: float = 0.80
    top_variable_genes_for_pca: int = 5_000
    n_pcs: int = 10

    def validate(self) -> None:
        if self.robust_outlier_mad <= 0:
            raise ValueError("robust_outlier_mad must be positive")
        if not -1 <= self.min_median_sample_correlation <= 1:
            raise ValueError("Correlation threshold must be between -1 and one")
        if self.top_variable_genes_for_pca < 2 or self.n_pcs < 2:
            raise ValueError("PCA settings must be at least two")


def _strip_characteristic(values: list[str]) -> dict[str, list[str]]:
    """Split a GEO characteristic row, including mixed per-sample keys."""

    parsed = [value.split(":", 1) for value in values]
    if any(len(item) != 2 for item in parsed):
        raise ValueError("Malformed GEO sample characteristic")
    normalized = [
        item[0].strip().lower().replace(" ", "_").replace("/", "_")
        for item in parsed
    ]
    return {
        key: [
            item[1].strip() if item_key == key else ""
            for item_key, item in zip(normalized, parsed)
        ]
        for key in sorted(set(normalized))
    }


def load_series_matrix(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Read the official processed matrix and sample metadata in one file."""

    sample_fields: dict[str, list[str]] = {}
    characteristics: dict[str, list[str]] = {}
    series_fields: dict[str, str] = {}
    table_start: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("!Series_"):
                row = next(csv.reader([line], delimiter="\t"))
                series_fields[row[0].removeprefix("!Series_")] = row[1] if len(row) > 1 else ""
            elif line.startswith("!Sample_"):
                row = next(csv.reader([line], delimiter="\t"))
                key, values = row[0].removeprefix("!Sample_"), row[1:]
                if key == "characteristics_ch1":
                    for characteristic_key, parsed in _strip_characteristic(values).items():
                        characteristics[characteristic_key] = parsed
                else:
                    sample_fields[key] = values
            elif line.startswith("!series_matrix_table_begin"):
                table_start = line_number + 1
                break
    if table_start is None:
        raise ValueError("Series matrix table marker was not found")
    required = {"geo_accession", "title", "source_name_ch1", "platform_id"}
    missing = required - set(sample_fields)
    if missing:
        raise ValueError(f"Missing series-matrix sample fields: {sorted(missing)}")
    sample_ids = sample_fields["geo_accession"]
    if len(sample_ids) != 148 or len(set(sample_ids)) != 148:
        raise ValueError("GSE51981 must contain 148 unique GEO samples")

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient_id": sample_fields["title"],
            "sample_title": sample_fields["title"],
            "source_name": sample_fields["source_name_ch1"],
            "platform_id": sample_fields["platform_id"],
            **characteristics,
        }
    )
    metadata["study_id"] = "GSE51981"
    metadata["condition"] = metadata["endometriosis_no_endometriosis"].map(
        {"Endometriosis": "Endometriosis", "Non-Endometriosis": "Non_Endometriosis"}
    )
    if metadata["condition"].isna().any():
        raise ValueError("Unrecognized GSE51981 disease label")
    metadata["clinical_group"] = np.select(
        [
            metadata["condition"].eq("Endometriosis"),
            metadata["presence_or_absence_of_uterine_pelvic_pathology"].eq(
                "No Uterine Pelvic Pathology"
            ),
            metadata["presence_or_absence_of_uterine_pelvic_pathology"].eq(
                "Uterine Pelvic Pathology"
            ),
        ],
        [
            "Endometriosis",
            "Healthy_control_no_pathology",
            "Non_endometriosis_other_pathology",
        ],
        default="Unresolved",
    )
    metadata["cycle_phase"] = (
        metadata["tissue"]
        .str.replace(" Endometrial tissue", "", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    source_severity = metadata["source_name"].str.removeprefix("Endometriosis_")
    characteristic_severity = metadata["endometriosis_severity"]
    disease = metadata["condition"].eq("Endometriosis")
    metadata["severity_metadata_concordant"] = (
        ~disease | source_severity.eq(characteristic_severity)
    )
    metadata["metadata_source"] = "NCBI_GEO_series_matrix"
    metadata["geo_sample_url"] = metadata["sample_id"].map(
        lambda value: f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={value}"
    )

    expression = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        skiprows=table_start,
        comment="!",
        index_col=0,
    )
    expression.index.name = "probe_id"
    if list(expression.columns) != sample_ids:
        raise ValueError("Expression columns do not match GEO sample order")
    values = expression.to_numpy(dtype=np.float64)
    if expression.index.duplicated().any() or not np.isfinite(values).all():
        raise ValueError("Processed probe matrix must be unique and finite")
    return expression, metadata, series_fields


def load_gpl570_annotation(path: Path) -> pd.DataFrame:
    """Load official GEO probe annotation and retain unambiguous gene symbols."""

    header_row: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("ID\t"):
                header_row = line_number
                break
    if header_row is None:
        raise ValueError("GPL570 annotation header was not found")
    annotation = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        skiprows=header_row,
        low_memory=False,
    )
    annotation = annotation.rename(
        columns={"ID": "probe_id", "Gene symbol": "gene_symbol", "Gene ID": "gene_id"}
    )
    required = {"probe_id", "gene_symbol", "gene_id"}
    missing = required - set(annotation.columns)
    if missing:
        raise ValueError(f"Missing GPL570 annotation columns: {sorted(missing)}")
    symbols = annotation["gene_symbol"].fillna("").astype(str).str.strip()
    unambiguous = (
        symbols.ne("")
        & symbols.ne("---")
        & ~symbols.str.contains("///", regex=False)
        & symbols.str.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    )
    result = annotation.loc[unambiguous, ["probe_id", "gene_symbol", "gene_id"]].copy()
    return result.drop_duplicates("probe_id").reset_index(drop=True)


def collapse_probes_to_genes(
    probe_expression: pd.DataFrame,
    annotation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse unambiguous probe sets to genes using the within-gene median."""

    common = annotation.loc[annotation["probe_id"].isin(probe_expression.index)].copy()
    aligned = probe_expression.loc[common["probe_id"]].copy()
    aligned.insert(0, "gene_symbol", common["gene_symbol"].to_numpy())
    gene_expression = aligned.groupby("gene_symbol", observed=True).median()
    gene_expression.index.name = "gene"
    mapping = (
        common.groupby("gene_symbol", observed=True)
        .agg(
            n_probes=("probe_id", "nunique"),
            probe_ids=("probe_id", lambda values: ";".join(sorted(values))),
        )
        .reset_index()
        .rename(columns={"gene_symbol": "gene"})
    )
    return gene_expression, mapping


def _robust_z(values: pd.Series) -> pd.Series:
    """Calculate median/MAD z-scores with a zero-MAD fallback."""

    median = values.median()
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 0.67448975 * (values - median) / mad


def processed_expression_qc(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: MicroarrayEDASettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate processed-distribution and sample-correlation diagnostics."""

    settings.validate()
    qc = metadata.copy()
    qc["expression_median"] = expression.median(axis=0).to_numpy()
    qc["expression_iqr"] = (
        expression.quantile(0.75, axis=0) - expression.quantile(0.25, axis=0)
    ).to_numpy()
    qc["expression_mean"] = expression.mean(axis=0).to_numpy()
    qc["expression_sd"] = expression.std(axis=0).to_numpy()
    qc["expression_median_robust_z"] = _robust_z(qc["expression_median"]).to_numpy()
    qc["expression_iqr_robust_z"] = _robust_z(qc["expression_iqr"]).to_numpy()
    correlation = expression.corr(method="spearman")
    medians = []
    minima = []
    for sample in correlation.columns:
        other = correlation.loc[sample].drop(sample)
        medians.append(other.median())
        minima.append(other.min())
    qc["median_spearman_correlation_to_other_samples"] = medians
    qc["minimum_spearman_correlation_to_other_samples"] = minima
    flags = []
    for row in qc.itertuples(index=False):
        row_flags = []
        if abs(row.expression_median_robust_z) > settings.robust_outlier_mad:
            row_flags.append("processed_expression_median_outlier")
        if abs(row.expression_iqr_robust_z) > settings.robust_outlier_mad:
            row_flags.append("processed_expression_iqr_outlier")
        if (
            row.median_spearman_correlation_to_other_samples
            < settings.min_median_sample_correlation
        ):
            row_flags.append("low_sample_correlation")
        flags.append(";".join(row_flags))
    qc["qc_flags"] = flags
    qc["qc_status"] = np.where(qc["qc_flags"].eq(""), "pass_processed_qc", "review")
    correlation.index.name = "sample_id"
    return qc, correlation


def expression_pca(
    gene_expression: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: MicroarrayEDASettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate PCA from the most variable mapped genes."""

    variance = gene_expression.var(axis=1)
    n_genes = min(settings.top_variable_genes_for_pca, len(variance))
    selected = variance.nlargest(n_genes).index
    matrix = gene_expression.loc[selected].T.to_numpy(dtype=np.float64)
    matrix -= matrix.mean(axis=0)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    n_components = min(settings.n_pcs, len(metadata) - 1, matrix.shape[1])
    coordinates = u[:, :n_components] * singular_values[:n_components]
    explained = singular_values**2 / np.sum(singular_values**2)
    pca = metadata.copy()
    for index in range(n_components):
        pca[f"PC{index + 1}"] = coordinates[:, index]
    variance_frame = pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(n_components)],
            "explained_variance_ratio": explained[:n_components],
        }
    )
    return pca, variance_frame


def pca_factor_associations(
    pca: pd.DataFrame,
    factors: tuple[str, ...] = ("clinical_group", "cycle_phase"),
    n_components: int = 5,
) -> pd.DataFrame:
    """Report descriptive eta-squared associations between PCs and factors."""

    rows = []
    for factor in factors:
        if factor not in pca:
            raise ValueError(f"Missing PCA factor: {factor}")
        for component_index in range(1, n_components + 1):
            component = f"PC{component_index}"
            values = pca[component]
            grand_mean = values.mean()
            between = sum(
                len(group) * (group.mean() - grand_mean) ** 2
                for _, group in pca.groupby(factor, observed=True)[component]
            )
            total = ((values - grand_mean) ** 2).sum()
            rows.append(
                {
                    "factor": factor,
                    "component": component,
                    "eta_squared": between / total if total > 0 else np.nan,
                    "interpretation": "descriptive_not_hypothesis_test",
                }
            )
    return pd.DataFrame(rows)


def plot_microarray_eda(
    expression: pd.DataFrame,
    qc: pd.DataFrame,
    pca: pd.DataFrame,
    variance: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save processed-expression distributions, correlations and PCA."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(data=expression, color="#8FBBD9", fliersize=0, ax=ax)
    ax.set_xticks([])
    ax.set_xlabel("148 GSE51981 samples")
    ax.set_ylabel("Official GCRMA processed expression")
    ax.set_title("Processed probe-expression distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "processed_expression_boxplots.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=pca,
        x="PC1",
        y="PC2",
        hue="clinical_group",
        style="cycle_phase",
        ax=ax,
    )
    ax.set_xlabel(f"PC1 ({100 * variance.iloc[0]['explained_variance_ratio']:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * variance.iloc[1]['explained_variance_ratio']:.1f}%)")
    ax.set_title("GSE51981 PCA: clinical group and menstrual-cycle phase")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_clinical_group_cycle_phase.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ordered = qc.sort_values(
        "median_spearman_correlation_to_other_samples"
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(
        data=ordered,
        x=np.arange(len(ordered)),
        y="median_spearman_correlation_to_other_samples",
        hue="qc_status",
        ax=ax,
    )
    ax.axhline(0.80, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Samples ordered by median correlation")
    ax.set_ylabel("Median Spearman correlation")
    ax.set_title("Processed-expression sample correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_correlation_summary.png", dpi=180)
    plt.close(fig)
