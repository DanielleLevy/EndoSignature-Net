"""Outcome-blind metadata, gene mapping, and RNA-seq QC for GSE212787."""

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.external_cohort_intake import FROZEN_GENES


@dataclass(frozen=True)
class RNASeqIntakeSettings:
    """Predeclared processed-count QC settings."""

    minimum_total_count_for_qc: int = 10
    robust_outlier_mad: float = 3.0
    top_variable_genes_for_pca: int = 5_000
    n_pcs: int = 10


def _parse_characteristics(rows: list[list[str]]) -> dict[str, list[str]]:
    """Convert aligned GEO characteristic rows into named columns."""

    parsed: dict[str, list[str]] = {}
    for values in rows:
        split = [value.split(":", 1) for value in values]
        if any(len(item) != 2 for item in split):
            raise ValueError("Malformed GEO characteristic row")
        keys = [item[0].strip().lower().replace(" ", "_") for item in split]
        if len(set(keys)) != 1:
            raise ValueError("Mixed GEO characteristic keys are not supported")
        parsed[keys[0]] = [item[1].strip() for item in split]
    return parsed


def load_gse212787_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load all 20 GEO samples and establish explicit matrix-column mapping."""

    sample_fields: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    series_fields: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Series_"):
                row = next(csv.reader([line], delimiter="\t"))
                series_fields[row[0].removeprefix("!Series_")] = (
                    row[1] if len(row) > 1 else ""
                )
            elif line.startswith("!Sample_"):
                row = next(csv.reader([line], delimiter="\t"))
                key, values = row[0].removeprefix("!Sample_"), row[1:]
                if key == "characteristics_ch1":
                    characteristics.append(values)
                else:
                    sample_fields[key] = values
            elif line.startswith("!series_matrix_table_begin"):
                break
    required = {"geo_accession", "title", "platform_id"}
    missing = required - set(sample_fields)
    if missing:
        raise ValueError(f"Missing GEO metadata fields: {sorted(missing)}")
    sample_ids = sample_fields["geo_accession"]
    if len(sample_ids) != 20 or len(set(sample_ids)) != 20:
        raise ValueError("GSE212787 must contain 20 unique GEO samples")

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_title": sample_fields["title"],
            "platform_id": sample_fields["platform_id"],
            **_parse_characteristics(characteristics),
        }
    )
    patient = metadata["sample_title"].str.extract(r"patient\s+(\d+)", expand=False)
    if patient.isna().any():
        raise ValueError("A patient number is missing from a GSE212787 title")
    metadata["patient_id"] = "patient_" + patient
    title = metadata["sample_title"].str.lower()
    metadata["tissue_class"] = np.select(
        [
            title.str.startswith("control eutopic"),
            title.str.startswith("eutopic"),
            title.str.startswith("ecopic"),
        ],
        ["control_eutopic", "endometriosis_eutopic", "endometriosis_ectopic"],
        default="unresolved",
    )
    if metadata["tissue_class"].eq("unresolved").any():
        raise ValueError("Unresolved GSE212787 tissue title")
    metadata["condition"] = np.where(
        metadata["tissue_class"].eq("control_eutopic"),
        "Disease_free_control",
        "Endometriosis",
    )
    metadata["include_in_external_target"] = metadata["tissue_class"].isin(
        ["control_eutopic", "endometriosis_eutopic"]
    )

    control_aliases = ["NC5P", "NC6P", "NC7P", "NC8P", "NC9P", "NC10S"]
    aliases: list[str] = []
    control_index = 0
    for row in metadata.itertuples(index=False):
        number = row.patient_id.removeprefix("patient_")
        if row.tissue_class == "control_eutopic":
            aliases.append(control_aliases[control_index])
            control_index += 1
        elif row.tissue_class == "endometriosis_eutopic":
            suffix = "S" if number in {"1", "2", "3"} else "P"
            aliases.append(f"EU{number}{suffix}")
        else:
            suffix = "S" if number == "1" else "P"
            aliases.append(f"EC{number}{suffix}")
    metadata["matrix_sample_alias"] = aliases
    metadata["cycle_phase"] = metadata["matrix_sample_alias"].str[-1].map(
        {"P": "Proliferative", "S": "Secretory"}
    )
    metadata["study_id"] = "GSE212787"
    metadata["metadata_source"] = (
        "NCBI_GEO_series_matrix_titles_and_deposited_expression_column_names"
    )
    metadata["geo_sample_url"] = metadata["sample_id"].map(
        lambda value: f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={value}"
    )
    if metadata["matrix_sample_alias"].duplicated().any():
        raise ValueError("Expression aliases must be unique")
    return metadata, series_fields


def load_count_and_fpkm(
    path: Path, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read deposited counts/FPKM and rename columns to GEO accessions."""

    deposited = pd.read_csv(path, sep="\t", compression="gzip")
    if "gene_id" not in deposited:
        raise ValueError("Deposited expression file lacks gene_id")
    deposited = deposited.loc[
        deposited["gene_id"].fillna("").astype(str).str.match(r"^ENSG\d+$")
    ].copy()
    if deposited["gene_id"].duplicated().any():
        raise ValueError("Deposited Ensembl identifiers must be unique")
    alias_to_geo = metadata.set_index("matrix_sample_alias")["sample_id"].to_dict()
    count_columns = {
        column: alias_to_geo[column.removeprefix("Count_ ").strip()]
        for column in deposited
        if column.startswith("Count_ ")
    }
    fpkm_columns = {
        column: alias_to_geo[column.removeprefix("FPKM_ ").strip()]
        for column in deposited
        if column.startswith("FPKM_ ")
    }
    if len(count_columns) != 20 or len(fpkm_columns) != 20:
        raise ValueError("Expected 20 count and 20 FPKM columns")
    counts = deposited.set_index("gene_id")[list(count_columns)].rename(
        columns=count_columns
    )
    fpkm = deposited.set_index("gene_id")[list(fpkm_columns)].rename(
        columns=fpkm_columns
    )
    counts = counts.apply(pd.to_numeric, errors="raise")
    fpkm = fpkm.apply(pd.to_numeric, errors="raise")
    expected = set(metadata["sample_id"])
    if set(counts.columns) != expected or set(fpkm.columns) != expected:
        raise ValueError("Expression-to-GEO mapping is incomplete")
    if (counts < 0).any().any() or (fpkm < 0).any().any():
        raise ValueError("Counts and FPKM must be non-negative")
    return counts, fpkm


def load_ncbi_ensembl_mapping(
    path: Path, frozen_genes: tuple[str, ...] = FROZEN_GENES
) -> pd.DataFrame:
    """Map the frozen symbols to Ensembl IDs using NCBI gene_info."""

    gene_info = pd.read_csv(path, sep="\t", compression="gzip", dtype=str)
    selected = gene_info.loc[gene_info["Symbol"].isin(frozen_genes)].copy()
    selected["ensembl_gene_id"] = selected["dbXrefs"].str.extract(
        r"(?:^|\|)Ensembl:(ENSG\d+)"
    )
    mapping = selected.rename(
        columns={"Symbol": "gene", "GeneID": "ncbi_gene_id"}
    )[["gene", "ncbi_gene_id", "ensembl_gene_id"]]
    mapping = mapping.dropna(subset=["ensembl_gene_id"]).drop_duplicates("gene")
    order = {gene: index for index, gene in enumerate(frozen_genes)}
    mapping["frozen_order"] = mapping["gene"].map(order)
    return mapping.sort_values("frozen_order").reset_index(drop=True)


def extract_frozen_expression(
    expression: pd.DataFrame,
    mapping: pd.DataFrame,
    frozen_genes: tuple[str, ...] = FROZEN_GENES,
) -> pd.DataFrame:
    """Extract mapped frozen genes without replacement or imputation."""

    available = mapping.loc[
        mapping["ensembl_gene_id"].isin(expression.index)
    ].set_index("gene")
    rows = [
        expression.loc[available.loc[gene, "ensembl_gene_id"]].rename(gene)
        for gene in frozen_genes
        if gene in available.index
    ]
    result = pd.DataFrame(rows)
    result.index.name = "gene"
    return result


def _robust_z(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 0.67448975 * (values - median) / mad


def label_blind_count_qc(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: RNASeqIntakeSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run library, detection, correlation, and unlabelled PCA diagnostics."""

    target_ids = metadata.loc[
        metadata["include_in_external_target"], "sample_id"
    ].tolist()
    target_counts = counts[target_ids]
    library_sizes = target_counts.sum(axis=0)
    cpm = target_counts.divide(library_sizes, axis=1) * 1_000_000
    log_cpm = np.log2(cpm + 1)
    eligible_genes = target_counts.sum(axis=1).ge(
        settings.minimum_total_count_for_qc
    )
    qc_matrix = log_cpm.loc[eligible_genes]

    target_metadata = metadata.set_index("sample_id").loc[target_ids].reset_index()
    qc = target_metadata.copy()
    qc["library_size"] = library_sizes.loc[target_ids].to_numpy()
    qc["detected_genes_count_gt_zero"] = (
        target_counts.gt(0).sum(axis=0).loc[target_ids].to_numpy()
    )
    qc["library_size_robust_z"] = _robust_z(qc["library_size"]).to_numpy()
    qc["detected_genes_robust_z"] = _robust_z(
        qc["detected_genes_count_gt_zero"]
    ).to_numpy()
    correlation = qc_matrix.corr(method="spearman")
    qc["median_spearman_to_other_samples"] = [
        correlation.loc[sample].drop(sample).median() for sample in target_ids
    ]
    qc["correlation_robust_z"] = _robust_z(
        qc["median_spearman_to_other_samples"]
    ).to_numpy()
    qc["maximum_spearman_to_other_sample"] = [
        correlation.loc[sample].drop(sample).max() for sample in target_ids
    ]
    flags: list[str] = []
    for row in qc.itertuples(index=False):
        current: list[str] = []
        if abs(row.library_size_robust_z) > settings.robust_outlier_mad:
            current.append("library_size_outlier")
        if abs(row.detected_genes_robust_z) > settings.robust_outlier_mad:
            current.append("detected_gene_count_outlier")
        if row.correlation_robust_z < -settings.robust_outlier_mad:
            current.append("low_global_sample_correlation")
        if row.maximum_spearman_to_other_sample > 0.9999:
            current.append("possible_duplicate_expression_profile")
        flags.append(";".join(current))
    qc["qc_flags"] = flags
    qc["qc_status"] = np.where(qc["qc_flags"].eq(""), "pass", "review")

    variance = qc_matrix.var(axis=1).nlargest(
        min(settings.top_variable_genes_for_pca, len(qc_matrix))
    )
    matrix = qc_matrix.loc[variance.index].T
    n_components = min(settings.n_pcs, len(matrix) - 1, matrix.shape[1])
    model = PCA(n_components=n_components, svd_solver="full")
    coordinates = model.fit_transform(matrix)
    pca = pd.DataFrame(
        coordinates,
        columns=[f"PC{index + 1}" for index in range(n_components)],
    )
    pca.insert(0, "sample_id", target_ids)
    explained = pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(n_components)],
            "explained_variance_ratio": model.explained_variance_ratio_,
        }
    )
    return qc, correlation, pca, explained, log_cpm
