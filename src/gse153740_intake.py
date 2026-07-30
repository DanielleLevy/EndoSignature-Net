"""Outcome-blind transcript-to-gene intake and QC for GSE153740."""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_gse153740_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load eight biological samples and their deposited expression aliases."""

    sample_fields: dict[str, list[str]] = {}
    characteristic_rows: list[list[str]] = []
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
                    characteristic_rows.append(values)
                else:
                    sample_fields[key] = values
            elif line.startswith("!series_matrix_table_begin"):
                break
    sample_ids = sample_fields.get("geo_accession", [])
    if len(sample_ids) != 8 or len(set(sample_ids)) != 8:
        raise ValueError("GSE153740 must contain eight unique GEO samples")
    characteristics: dict[str, list[str]] = {}
    for values in characteristic_rows:
        split = [value.split(":", 1) for value in values]
        keys = [item[0].strip().lower().replace(" ", "_") for item in split]
        if any(len(item) != 2 for item in split) or len(set(keys)) != 1:
            raise ValueError("Malformed GEO characteristic row")
        characteristics[keys[0]] = [item[1].strip() for item in split]

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_title": sample_fields["title"],
            "matrix_sample_alias": sample_fields["description"],
            "platform_id": sample_fields["platform_id"],
            **characteristics,
        }
    )
    metadata["patient_id"] = metadata["sample_id"]
    metadata["condition"] = metadata["group"].map(
        {"Endometriosis": "Endometriosis", "Control": "Disease_free_control"}
    )
    if metadata["condition"].isna().any():
        raise ValueError("Unresolved disease group")
    metadata["cycle_phase"] = "Mid_secretory"
    metadata["tissue_class"] = "eutopic_endometrium"
    metadata["study_id"] = "GSE153740"
    metadata["metadata_source"] = "NCBI_GEO_series_matrix"
    if metadata["matrix_sample_alias"].duplicated().any():
        raise ValueError("Expression aliases must be unique")
    return metadata, series_fields


def load_ensembl90_transcript_mapping(path: Path) -> pd.DataFrame:
    """Extract transcript-to-gene mappings from the matching Ensembl 90 GTF."""

    rows: list[dict[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            attributes = dict(re.findall(r'(\S+) "([^"]+)";', fields[8]))
            transcript_id = attributes.get("transcript_id", "").split(".")[0]
            gene_id = attributes.get("gene_id", "").split(".")[0]
            gene = attributes.get("gene_name", "")
            if transcript_id and gene_id and gene:
                rows.append(
                    {
                        "transcript_id": transcript_id,
                        "ensembl_gene_id": gene_id,
                        "gene": gene,
                    }
                )
    mapping = pd.DataFrame(rows).drop_duplicates("transcript_id")
    if mapping.empty:
        raise ValueError("No transcript mappings parsed from Ensembl GTF")
    return mapping


def load_and_collapse_fpkm(
    expression_path: Path,
    metadata: pd.DataFrame,
    transcript_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map deposited transcript FPKM to genes by summing transcript estimates."""

    transcript = pd.read_csv(expression_path, sep="\t", compression="gzip")
    if "Transcript_ID" not in transcript:
        raise ValueError("Transcript expression lacks Transcript_ID")
    transcript["transcript_id"] = (
        transcript["Transcript_ID"].astype(str).str.split(".").str[0]
    )
    if transcript["transcript_id"].duplicated().any():
        raise ValueError("Deposited transcript identifiers must be unique")
    alias_to_geo = metadata.set_index("matrix_sample_alias")["sample_id"].to_dict()
    sample_columns = [column for column in transcript if column in alias_to_geo]
    if len(sample_columns) != 8:
        raise ValueError("Expected eight deposited expression columns")
    transcript_expression = transcript.set_index("transcript_id")[sample_columns].rename(
        columns=alias_to_geo
    )
    transcript_expression = transcript_expression.apply(pd.to_numeric, errors="raise")
    if (transcript_expression < 0).any().any():
        raise ValueError("FPKM values must be non-negative")

    common = transcript_mapping.loc[
        transcript_mapping["transcript_id"].isin(transcript_expression.index)
    ].copy()
    aligned = transcript_expression.loc[common["transcript_id"]].copy()
    aligned.insert(0, "gene", common["gene"].to_numpy())
    gene_expression = aligned.groupby("gene", observed=True).sum()
    gene_expression.index.name = "gene"
    coverage = (
        common.groupby("gene", observed=True)
        .agg(
            mapped_transcripts=("transcript_id", "nunique"),
            transcript_ids=(
                "transcript_id",
                lambda values: ";".join(sorted(values)),
            ),
        )
        .reset_index()
    )
    return transcript_expression, gene_expression, coverage


def label_blind_fpkm_qc(
    transcript_expression: pd.DataFrame,
    metadata: pd.DataFrame,
    robust_outlier_mad: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run detection, total-abundance, and correlation checks without labels."""

    log_expression = np.log2(transcript_expression + 1)
    qc = metadata.copy()
    qc["total_fpkm"] = transcript_expression.sum(axis=0).loc[
        qc["sample_id"]
    ].to_numpy()
    qc["detected_transcripts"] = transcript_expression.gt(0).sum(axis=0).loc[
        qc["sample_id"]
    ].to_numpy()

    def robust_z(values: pd.Series) -> pd.Series:
        median = values.median()
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return 0.67448975 * (values - median) / mad

    qc["total_fpkm_robust_z"] = robust_z(qc["total_fpkm"]).to_numpy()
    qc["detected_transcripts_robust_z"] = robust_z(
        qc["detected_transcripts"]
    ).to_numpy()
    informative = transcript_expression.sum(axis=1).gt(0)
    correlation = log_expression.loc[informative].corr(method="spearman")
    qc["median_spearman_to_other_samples"] = [
        correlation.loc[sample].drop(sample).median() for sample in qc["sample_id"]
    ]
    qc["correlation_robust_z"] = robust_z(
        qc["median_spearman_to_other_samples"]
    ).to_numpy()
    flags: list[str] = []
    for row in qc.itertuples(index=False):
        current: list[str] = []
        if abs(row.total_fpkm_robust_z) > robust_outlier_mad:
            current.append("total_fpkm_outlier")
        if abs(row.detected_transcripts_robust_z) > robust_outlier_mad:
            current.append("detected_transcript_count_outlier")
        if row.correlation_robust_z < -robust_outlier_mad:
            current.append("low_global_sample_correlation")
        flags.append(";".join(current))
    qc["qc_flags"] = flags
    qc["qc_status"] = np.where(qc["qc_flags"].eq(""), "pass", "review")
    return qc, correlation
