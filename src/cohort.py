"""Primary-cohort definition and per-sample doublet detection.

The primary comparison is deliberately restricted to control endometrium versus
eutopic endometrium from patients with endometriosis.  GEO-verified annotations
are kept separate from filename-derived metadata to preserve provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


PRIMARY_SAMPLE_TITLES = {
    "GSM6102532": "Control Patient 1 - Control Endometrium",
    "GSM6102533": "Control Patient 2 - Control Endometrium",
    "GSM6102534": "Control Patient 3 - Control Endometrium",
    "GSM6102537": "Endometriosis Patient 1 - Eutopic",
    "GSM6102540": "Endometriosis Patient 2 - Eutopic",
    "GSM6102543": "Endometriosis Patient 3 - Eutopic",
    "GSM6102546": "Endometriosis Patient 4 - Eutopic",
    "GSM6102549": "Endometriosis Patient 5 - Eutopic",
    "GSM6102551": "Endometriosis Patient 6 - Eutopic",
    "GSM6102554": "Endometriosis Patient 7 - Eutopic",
    "GSM6102555": "Endometriosis Patient 8 - Eutopic",
    "GSM6102560": "Endometriosis Patient 9 - Eutopic",
}

GEO_SERIES_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE179640"


@dataclass(frozen=True)
class DoubletSettings:
    """Parameters and review rules for per-sample Scrublet runs."""

    expected_doublet_rate: float = 0.06
    random_state: int = 0
    review_rate_pct: float = 15.0

    def validate(self) -> None:
        if not 0 < self.expected_doublet_rate < 1:
            raise ValueError("expected_doublet_rate must be between 0 and 1")
        if not 0 <= self.review_rate_pct <= 100:
            raise ValueError("review_rate_pct must be between 0 and 100")


def build_verified_primary_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return the 12 primary candidates annotated from the NCBI GEO records."""

    selected = metadata.loc[metadata["sample_id"].isin(PRIMARY_SAMPLE_TITLES)].copy()
    if len(selected) != len(PRIMARY_SAMPLE_TITLES):
        found = set(selected["sample_id"])
        missing = sorted(set(PRIMARY_SAMPLE_TITLES) - found)
        raise ValueError(f"Missing primary samples from local metadata: {missing}")

    selected["geo_title"] = selected["sample_id"].map(PRIMARY_SAMPLE_TITLES)
    selected["geo_condition"] = np.where(selected["tissue_code"] == "Ctrl", "Control", "Endometriosis")
    selected["geo_sample_location"] = "Eutopic"
    selected["geo_tissue"] = "Endometrium"
    selected["geo_method"] = "scRNA-seq"
    selected["geo_library_type"] = "Gene Expression"
    selected["hormonal_treatment"] = np.where(
        selected["geo_condition"] == "Endometriosis",
        "Oral contraceptive treatment stated by study protocol",
        "Not stated for controls in the GEO sample record",
    )
    selected["metadata_verification"] = "verified_against_NCBI_GEO"
    selected["geo_series_url"] = GEO_SERIES_URL
    selected["geo_sample_url"] = selected["sample_id"].map(
        lambda sample: f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={sample}"
    )
    return selected.sort_values("sample_id")


def run_scrublet(adata: ad.AnnData, settings: DoubletSettings) -> ad.AnnData:
    """Run Scrublet on raw counts and return a copy with cell-level results."""

    settings.validate()
    result = adata.copy()
    sc.pp.scrublet(
        result,
        expected_doublet_rate=settings.expected_doublet_rate,
        random_state=settings.random_state,
        copy=False,
    )
    return result


def summarize_doublets(
    adata: ad.AnnData,
    metadata: Mapping[str, str],
    settings: DoubletSettings,
) -> dict[str, object]:
    """Summarize one sample's Scrublet predictions and review status."""

    scores = adata.obs["doublet_score"].to_numpy()
    threshold = adata.uns.get("scrublet", {}).get("threshold", np.nan)
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Scrublet produced non-finite scores or threshold")
    predicted = adata.obs["predicted_doublet"].astype(bool)
    rate = 100.0 * float(predicted.mean()) if adata.n_obs else 0.0
    return {
        "sample_id": metadata["sample_id"],
        "patient_id": metadata["patient_id"],
        "condition": metadata["condition"],
        "tissue_code": metadata["tissue_code"],
        "n_cells_entering_scrublet": int(adata.n_obs),
        "n_predicted_doublets": int(predicted.sum()),
        "predicted_doublet_rate_pct": rate,
        "scrublet_threshold": float(threshold),
        "expected_doublet_rate": settings.expected_doublet_rate,
        "doublet_qc_status": "review" if rate > settings.review_rate_pct else "pass_initial_doublet_qc",
    }


def write_cell_doublets(adata: ad.AnnData, sample_id: str, path: Path) -> None:
    """Persist cell-level scores using sample-qualified cell barcodes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(
        {
            "cell_id": [f"{sample_id}:{barcode}" for barcode in adata.obs_names],
            "sample_id": sample_id,
            "doublet_score": adata.obs["doublet_score"].to_numpy(),
            "predicted_doublet": adata.obs["predicted_doublet"].astype(bool).to_numpy(),
        }
    )
    result.to_csv(path, index=False)


def plot_doublet_scores(adata: ad.AnnData, sample_id: str, path: Path) -> None:
    """Save the observed score distribution and Scrublet threshold."""

    path.parent.mkdir(parents=True, exist_ok=True)
    scores = adata.obs["doublet_score"].to_numpy()
    threshold = adata.uns.get("scrublet", {}).get("threshold", np.nan)
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(scores, bins=50, ax=axis, color="#4C78A8")
    if np.isfinite(threshold):
        axis.axvline(threshold, color="#E45756", linestyle="--", label=f"Threshold = {threshold:.3f}")
        axis.legend()
    axis.set_title(f"Scrublet scores: {sample_id}")
    axis.set_xlabel("Doublet score")
    axis.set_ylabel("Cells")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_inclusion_decisions(
    metadata: pd.DataFrame,
    doublet_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Assign an explicit analysis role and reason to every logical record."""

    doublet_status = doublet_summary.set_index("sample_id")["doublet_qc_status"].to_dict()
    rows = []
    for _, item in metadata.iterrows():
        sample_id = item["sample_id"]
        if sample_id in PRIMARY_SAMPLE_TITLES:
            status = doublet_status.get(sample_id, "doublet_result_missing")
            if status == "pass_initial_doublet_qc":
                role, decision = "primary_cohort", "include"
                reason = "GEO-verified control or eutopic endometrium; passed initial cell and doublet QC"
            else:
                role, decision = "primary_cohort", "review"
                reason = f"Primary comparison candidate requires review: {status}"
        elif item["technology"] == "scRNA-seq":
            role, decision = "secondary_single_cell", "reserve"
            reason = "Different lesion site or organoid; reserved for secondary analysis"
        elif item["technology"] == "cell_hashing":
            role, decision = "demultiplexing_support", "exclude_from_expression"
            reason = "HTO-only matrix without gene-expression features"
        elif item["technology"] in {"bulk_RNA-seq", "microarray"}:
            role, decision = "external_validation", "reserve"
            reason = "Different transcriptomic measurement level; reserved for signature validation"
        else:
            role, decision = "provenance", "exclude_from_analysis"
            reason = "Archive or unsupported logical record"
        rows.append(
            {
                "sample_id": sample_id,
                "study_id": item["study_id"],
                "patient_id": item["patient_id"],
                "condition": item["condition"],
                "tissue_code": item["tissue_code"],
                "technology": item["technology"],
                "analysis_role": role,
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_role", "sample_id"])


def build_validation_decisions(
    metadata: pd.DataFrame,
    qc_summary: pd.DataFrame,
    doublet_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine independent QC signals without silently excluding samples."""

    qc_status = qc_summary.set_index("sample_id")["qc_status"].to_dict()
    doublet_status = doublet_summary.set_index("sample_id")["doublet_qc_status"].to_dict()
    rows = []
    for _, item in metadata.iterrows():
        sample_id = item["sample_id"]
        available = bool(item["include_in_sc_eda"])
        cell_status = qc_status.get(sample_id, "not_run")
        dbl_status = doublet_status.get(sample_id, "not_run")

        if item["technology"] == "spatial_transcriptomics":
            role, decision = "spatial_secondary", "exclude_from_scRNA"
            reason = "Visium spatial object; not a single-cell expression matrix"
        elif not available:
            role, decision = "provenance", "not_available"
            reason = "No local filtered single-cell matrix available"
        elif cell_status == "pass_initial_qc" and dbl_status == "pass_initial_doublet_qc":
            role, decision = "single_cell_validation", "include"
            reason = "Passed initial cell QC and per-sample doublet review rules"
        else:
            role, decision = "single_cell_validation", "review"
            reason = f"Requires review before atlas mapping: cell_qc={cell_status}; doublet_qc={dbl_status}"

        rows.append(
            {
                "sample_id": sample_id,
                "study_id": item["study_id"],
                "patient_id": item["patient_id"],
                "condition": item["condition"],
                "tissue_code": item["tissue_code"],
                "technology": item["technology"],
                "analysis_role": role,
                "cell_qc_status": cell_status,
                "doublet_qc_status": dbl_status,
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_role", "sample_id"])
