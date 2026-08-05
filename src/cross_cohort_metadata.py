"""Harmonize patient metadata across heterogeneous transcriptomic cohorts."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLINICAL_FIELDS = [
    "condition",
    "age",
    "cycle_phase",
    "disease_stage",
    "disease_subtype",
    "tissue",
    "hormonal_treatment",
    "fertility_status",
    "country",
]

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "unknown", "not stated", "not_available"}


def clean_value(value):
    """Return missing metadata as ``pd.NA`` without inventing information."""

    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    lowered = text.lower()
    return pd.NA if lowered in MISSING_TOKENS or lowered.startswith("not stated") else text


def clean_series(series: pd.Series) -> pd.Series:
    """Normalize explicit missing tokens in one metadata column."""

    return series.map(clean_value)


def normalize_cycle_phase(value):
    """Harmonize phase spelling while preserving unspecified secretory records."""

    value = clean_value(value)
    if pd.isna(value):
        return pd.NA
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "proliferative": "Proliferative",
        "early_secretory": "Early_Secretory",
        "mid_secretory": "Mid_Secretory",
        "late_secretory": "Late_Secretory",
        "secretory": "Secretory_unspecified",
    }
    return mapping.get(token, str(value))


def joined_categories(values: Iterable[object]) -> object:
    """Join categories, including values already combined at patient level."""

    categories = set()
    for value in values:
        value = clean_value(value)
        if pd.isna(value):
            continue
        categories.update(part.strip() for part in str(value).split(";") if part.strip())
    return ";".join(sorted(categories)) if categories else pd.NA


def joined_unique(values: Iterable[object]) -> object:
    """Join distinct observed values while preserving transparent conflicts."""

    cleaned = [clean_value(value) for value in values]
    unique = sorted({str(value) for value in cleaned if not pd.isna(value)})
    return ";".join(unique) if unique else pd.NA


def aggregate_patient_metadata(samples: pd.DataFrame) -> pd.DataFrame:
    """Collapse sample records to one auditable row per study-specific patient."""

    required = {"study_id", "patient_id", "sample_id"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing patient aggregation columns: {sorted(missing)}")
    rows = []
    for (study_id, patient_id), group in samples.groupby(
        ["study_id", "patient_id"], sort=True, dropna=False
    ):
        row = {
            "study_id": study_id,
            "patient_id": patient_id,
            "patient_uid": f"{study_id}:{patient_id}",
            "n_samples": int(len(group)),
            "sample_ids": ";".join(sorted(group["sample_id"].astype(str))),
        }
        for column in samples.columns:
            if column in {"study_id", "patient_id", "sample_id"}:
                continue
            row[column] = joined_unique(group[column])
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_completeness(
    patients: pd.DataFrame, fields: list[str] | None = None
) -> pd.DataFrame:
    """Calculate per-cohort patient-level completeness for predeclared fields."""

    fields = fields or CLINICAL_FIELDS
    rows = []
    for study_id, group in patients.groupby("study_id", sort=True):
        for field in fields:
            available = group[field].map(clean_value).notna()
            rows.append(
                {
                    "study_id": study_id,
                    "field": field,
                    "n_patients": int(len(group)),
                    "n_with_metadata": int(available.sum()),
                    "completeness_fraction": float(available.mean()),
                }
            )
    return pd.DataFrame(rows)


def cohort_summary(patients: pd.DataFrame) -> pd.DataFrame:
    """Summarize cohort size, design, and analysis eligibility."""

    rows = []
    for study_id, group in patients.groupby("study_id", sort=True):
        rows.append(
            {
                "study_id": study_id,
                "n_patients": int(len(group)),
                "n_samples": int(group["n_samples"].sum()),
                "conditions": joined_categories(group["condition"]),
                "cycle_phases": joined_categories(group["cycle_phase"]),
                "disease_stages": joined_categories(group["disease_stage"]),
                "disease_subtypes": joined_categories(group["disease_subtype"]),
                "tissues": joined_categories(group["tissue"]),
                "platforms": joined_categories(group["platform"]),
                "technologies": joined_categories(group["technology"]),
                "analysis_roles": joined_categories(group["analysis_role"]),
                "patient_id_quality": joined_categories(group["patient_id_quality"]),
            }
        )
    return pd.DataFrame(rows)


def pairwise_comparability(
    patients: pd.DataFrame, fields: list[str] | None = None
) -> pd.DataFrame:
    """Report shared metadata coverage and category overlap for each cohort pair."""

    fields = fields or CLINICAL_FIELDS
    studies = sorted(patients["study_id"].unique())
    rows = []
    for left_index, left_study in enumerate(studies):
        left = patients.loc[patients["study_id"].eq(left_study)]
        for right_study in studies[left_index + 1 :]:
            right = patients.loc[patients["study_id"].eq(right_study)]
            for field in fields:
                left_values = {
                    str(clean_value(value))
                    for value in left[field]
                    if not pd.isna(clean_value(value))
                }
                right_values = {
                    str(clean_value(value))
                    for value in right[field]
                    if not pd.isna(clean_value(value))
                }
                overlap = sorted(left_values.intersection(right_values))
                rows.append(
                    {
                        "left_study": left_study,
                        "right_study": right_study,
                        "field": field,
                        "left_completeness": float(left[field].map(clean_value).notna().mean()),
                        "right_completeness": float(right[field].map(clean_value).notna().mean()),
                        "shared_recorded_categories": ";".join(overlap) if overlap else pd.NA,
                        "has_recorded_category_overlap": bool(overlap),
                    }
                )
    return pd.DataFrame(rows)


def plot_metadata_audit(
    patients: pd.DataFrame,
    completeness: pd.DataFrame,
    output_path,
) -> None:
    """Create a compact dashboard of cohort scale, phase, and missingness."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    counts = patients.groupby("study_id").size().sort_values()
    axes[0, 0].barh(counts.index, counts.values, color="#3973B8")
    axes[0, 0].set_title("Patient records by cohort")
    axes[0, 0].set_xlabel("Patients / patient proxies")

    phase = (
        patients.assign(cycle_phase=patients["cycle_phase"].fillna("Missing"))
        .groupby(["study_id", "cycle_phase"])
        .size()
        .unstack(fill_value=0)
    )
    phase.plot(kind="bar", stacked=True, ax=axes[0, 1], colormap="tab20")
    axes[0, 1].set_title("Recorded menstrual-cycle phase")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Patients")
    axes[0, 1].legend(fontsize=8, frameon=False)

    matrix = completeness.pivot(
        index="study_id", columns="field", values="completeness_fraction"
    ).loc[sorted(patients["study_id"].unique()), CLINICAL_FIELDS]
    image = axes[1, 0].imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axes[1, 0].set_yticks(range(len(matrix.index)), matrix.index)
    axes[1, 0].set_title("Patient-level metadata completeness")
    figure.colorbar(image, ax=axes[1, 0], label="Fraction observed")

    technology = patients.groupby(["study_id", "technology"]).size().unstack(fill_value=0)
    technology_colors = {
        "bulk_RNA-seq": "#1B998B",
        "bulk_RNA-seq_transcript_FPKM": "#E9A23B",
        "microarray": "#6C77B5",
        "scRNA-seq": "#D95D5D",
    }
    technology.plot(
        kind="bar",
        stacked=True,
        ax=axes[1, 1],
        color=[technology_colors.get(column, "#888888") for column in technology.columns],
    )
    axes[1, 1].set_title("Measurement technology")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Patients / patient proxies")
    axes[1, 1].legend(fontsize=8, frameon=False)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
