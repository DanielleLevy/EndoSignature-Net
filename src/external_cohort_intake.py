"""Label-blind intake, frozen-gene mapping, and QC for GSE120103."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


FROZEN_GENES = (
    "PDZD2",
    "DPF3",
    "TMEM204",
    "RASSF4",
    "CA12",
    "ACSS2",
    "RNASE1",
    "ARG2",
    "SLC1A5",
    "STARD5",
    "ZSCAN30",
    "DUOX1",
)


@dataclass(frozen=True)
class ExternalIntakeSettings:
    """Predeclared, outcome-blind processed-expression QC settings."""

    robust_outlier_mad: float = 3.0
    correlation_outlier_mad: float = 3.0
    top_variable_probes_for_pca: int = 5_000
    n_pcs: int = 10


def _parse_characteristics(rows: list[list[str]]) -> dict[str, list[str]]:
    """Convert GEO characteristic rows to aligned named columns."""

    parsed: dict[str, list[str]] = {}
    for values in rows:
        split = [value.split(":", 1) for value in values]
        if any(len(item) != 2 for item in split):
            raise ValueError("Malformed GEO characteristic row")
        keys = [item[0].strip().lower().replace(" ", "_") for item in split]
        if len(set(keys)) != 1:
            raise ValueError("Mixed characteristic keys are not supported")
        parsed[keys[0]] = [item[1].strip() for item in split]
    return parsed


def load_gse120103_series_matrix(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load the official processed matrix and verify deposited metadata."""

    sample_fields: dict[str, list[str]] = {}
    characteristic_rows: list[list[str]] = []
    series_fields: dict[str, str] = {}
    table_start: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("!Series_"):
                row = next(csv.reader([line], delimiter="\t"))
                series_fields[row[0].removeprefix("!Series_")] = (
                    row[1] if len(row) > 1 else ""
                )
            elif line.startswith("!Sample_"):
                row = next(csv.reader([line], delimiter="\t"))
                key, values = row[0].removeprefix("!Sample_"), row[1:]
                if key == "characteristics_ch1":
                    characteristic_rows.append(values)
                else:
                    sample_fields[key] = values
            elif line.startswith("!series_matrix_table_begin"):
                table_start = line_number + 1
                break
    if table_start is None:
        raise ValueError("Series-matrix table marker was not found")

    required = {"geo_accession", "title", "source_name_ch1", "platform_id"}
    missing = required - set(sample_fields)
    if missing:
        raise ValueError(f"Missing sample fields: {sorted(missing)}")
    sample_ids = sample_fields["geo_accession"]
    if len(sample_ids) != 36 or len(set(sample_ids)) != 36:
        raise ValueError("GSE120103 must contain 36 unique GEO samples")

    characteristics = _parse_characteristics(characteristic_rows)
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_title": sample_fields["title"],
            "source_name": sample_fields["source_name_ch1"],
            "platform_id": sample_fields["platform_id"],
            **characteristics,
        }
    )
    metadata["study_id"] = "GSE120103"
    title_lower = metadata["sample_title"].str.lower()
    metadata["condition"] = np.where(
        title_lower.str.contains("disease free"),
        "Disease_free_control",
        "Stage_IV_ovarian_endometriosis",
    )
    metadata["fertility"] = np.where(
        title_lower.str.contains("infertile"), "Infertile", "Fertile"
    )
    metadata["cycle_phase"] = "Secretory"
    metadata["tissue_target"] = "Eutopic_endometrium"
    metadata["patient_id"] = metadata["sample_id"]
    metadata["source_name_fertility_concordant"] = [
        fertility.lower() in source.lower()
        for fertility, source in zip(metadata["fertility"], metadata["source_name"])
    ]
    metadata["metadata_source"] = "NCBI_GEO_series_matrix_and_primary_publication"
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


def load_gpl6480_annotation(path: Path) -> pd.DataFrame:
    """Read GPL6480 and retain probes with one valid gene symbol."""

    header_row: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("ID\t"):
                header_row = line_number
                break
    if header_row is None:
        raise ValueError("GPL6480 annotation header was not found")
    annotation = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        skiprows=header_row,
        low_memory=False,
    ).rename(
        columns={"ID": "probe_id", "Gene symbol": "gene_symbol", "Gene ID": "gene_id"}
    )
    required = {"probe_id", "gene_symbol", "gene_id"}
    missing = required - set(annotation)
    if missing:
        raise ValueError(f"Missing annotation fields: {sorted(missing)}")
    symbols = annotation["gene_symbol"].fillna("").astype(str).str.strip()
    valid = (
        symbols.ne("")
        & symbols.ne("---")
        & ~symbols.str.contains("///", regex=False)
        & symbols.str.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    )
    return (
        annotation.loc[valid, ["probe_id", "gene_symbol", "gene_id"]]
        .drop_duplicates("probe_id")
        .reset_index(drop=True)
    )


def map_frozen_signature(
    expression: pd.DataFrame,
    annotation: pd.DataFrame,
    frozen_genes: tuple[str, ...] = FROZEN_GENES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the highest-median valid probe per gene without using outcomes."""

    candidates = annotation.loc[
        annotation["gene_symbol"].isin(frozen_genes)
        & annotation["probe_id"].isin(expression.index)
    ].copy()
    medians = expression.median(axis=1).rename("across_sample_median")
    candidates = candidates.join(medians, on="probe_id")
    candidates = candidates.sort_values(
        ["gene_symbol", "across_sample_median", "probe_id"],
        ascending=[True, False, True],
    )
    candidates["selected_probe"] = ~candidates.duplicated("gene_symbol")
    candidates["selection_rule"] = "highest_across_sample_median_label_blind"

    selected = candidates.loc[candidates["selected_probe"]].set_index("gene_symbol")
    mapped = [
        expression.loc[selected.loc[gene, "probe_id"]].rename(gene)
        for gene in frozen_genes
        if gene in selected.index
    ]
    signature_expression = pd.DataFrame(mapped)
    signature_expression.index.name = "gene"
    return signature_expression, candidates.reset_index(drop=True)


def _robust_z(values: pd.Series) -> pd.Series:
    """Return median/MAD z-scores, with a zero-MAD fallback."""

    median = values.median()
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 0.67448975 * (values - median) / mad


def label_blind_processed_qc(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: ExternalIntakeSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run distribution, correlation, duplicate, and unlabelled PCA checks."""

    qc = metadata.copy()
    qc["expression_median"] = expression.median(axis=0).to_numpy()
    qc["expression_iqr"] = (
        expression.quantile(0.75, axis=0) - expression.quantile(0.25, axis=0)
    ).to_numpy()
    qc["expression_median_robust_z"] = _robust_z(qc["expression_median"]).to_numpy()
    qc["expression_iqr_robust_z"] = _robust_z(qc["expression_iqr"]).to_numpy()

    correlation = expression.corr(method="spearman")
    qc["median_spearman_to_other_samples"] = [
        correlation.loc[sample].drop(sample).median() for sample in correlation.columns
    ]
    qc["correlation_median_robust_z"] = _robust_z(
        qc["median_spearman_to_other_samples"]
    ).to_numpy()
    qc["maximum_spearman_to_other_sample"] = [
        correlation.loc[sample].drop(sample).max() for sample in correlation.columns
    ]

    flags: list[str] = []
    for row in qc.itertuples(index=False):
        current: list[str] = []
        if abs(row.expression_median_robust_z) > settings.robust_outlier_mad:
            current.append("processed_expression_median_outlier")
        if abs(row.expression_iqr_robust_z) > settings.robust_outlier_mad:
            current.append("processed_expression_iqr_outlier")
        if row.correlation_median_robust_z < -settings.correlation_outlier_mad:
            current.append("low_global_sample_correlation")
        if row.maximum_spearman_to_other_sample > 0.9999:
            current.append("possible_duplicate_expression_profile")
        flags.append(";".join(current))
    qc["qc_flags"] = flags
    qc["qc_status"] = np.where(qc["qc_flags"].eq(""), "pass", "review")

    variance = expression.var(axis=1).sort_values(ascending=False)
    selected = variance.head(min(settings.top_variable_probes_for_pca, len(variance)))
    matrix = expression.loc[selected.index].T
    standard_deviation = matrix.std(axis=0)
    stable = standard_deviation.gt(1e-8)
    matrix = matrix.loc[:, stable]
    scaled = (matrix - matrix.mean(axis=0)) / standard_deviation.loc[stable]
    n_components = min(settings.n_pcs, len(matrix) - 1, matrix.shape[1])
    model = PCA(n_components=n_components, svd_solver="full")
    coordinates = model.fit_transform(scaled)
    pca = pd.DataFrame(
        coordinates,
        columns=[f"PC{index + 1}" for index in range(n_components)],
    )
    pca.insert(0, "sample_id", expression.columns)
    explained = pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(n_components)],
            "explained_variance_ratio": model.explained_variance_ratio_,
        }
    )
    return qc, correlation, pca, explained
