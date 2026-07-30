"""Outcome-blind metadata and processed-microarray intake for GSE25628."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def load_gse25628_series_matrix(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load the official RMA matrix and define the eutopic target from titles."""

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
        raise ValueError("GSE25628 expression table marker is missing")
    sample_ids = sample_fields.get("geo_accession", [])
    if len(sample_ids) != 22 or len(set(sample_ids)) != 22:
        raise ValueError("GSE25628 must contain 22 unique GEO samples")

    characteristics: dict[str, list[str]] = {}
    for values in characteristic_rows:
        split = [value.split(":", 1) for value in values]
        if any(len(item) != 2 for item in split):
            raise ValueError("Malformed GEO characteristic")
        keys = [item[0].strip().lower().replace(" ", "_") for item in split]
        if len(set(keys)) != 1:
            raise ValueError("Mixed characteristic keys are unsupported")
        characteristics[keys[0]] = [item[1].strip() for item in split]
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_title": sample_fields["title"],
            "platform_id": sample_fields["platform_id"],
            **characteristics,
        }
    )
    title = metadata["sample_title"].str.lower()
    metadata["tissue_class"] = np.select(
        [
            title.str.startswith("normal endometrium"),
            title.str.startswith("eutopic endometrium"),
            title.str.startswith("ectopic endometrium"),
        ],
        ["control_endometrium", "endometriosis_eutopic", "endometriosis_ectopic"],
        default="unresolved",
    )
    if metadata["tissue_class"].eq("unresolved").any():
        raise ValueError("Unresolved GSE25628 tissue title")
    metadata["condition"] = np.select(
        [
            metadata["tissue_class"].eq("control_endometrium"),
            metadata["tissue_class"].eq("endometriosis_eutopic"),
            metadata["tissue_class"].eq("endometriosis_ectopic"),
        ],
        ["Disease_free_control", "Endometriosis", "Endometriosis"],
        default="Unresolved",
    )
    metadata["include_in_external_target"] = metadata["tissue_class"].isin(
        ["control_endometrium", "endometriosis_eutopic"]
    )
    metadata["cycle_phase"] = "Proliferative"
    metadata["patient_id"] = metadata["sample_id"]
    metadata["study_id"] = "GSE25628"
    metadata["patient_identifier_available"] = False
    metadata["metadata_source"] = "NCBI_GEO_series_matrix"

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
        raise ValueError("Expression columns do not match the GEO sample order")
    if expression.index.duplicated().any():
        raise ValueError("GSE25628 probe identifiers must be unique")
    values = expression.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("GSE25628 expression must be finite")
    return expression, metadata, series_fields
