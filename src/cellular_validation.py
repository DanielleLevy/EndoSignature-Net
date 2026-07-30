"""Cell-family localization of replicated signatures in GSE213216.

This analysis does not test disease direction because GSE213216 lacks a
comparable healthy-control group. It asks a narrower question: among cells
with marker-supported transferred labels, are replicated genes preferentially
expressed in the cell families where they were discovered in GSE179640?
Biological replication is evaluated at the patient-by-tissue level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class CellularValidationSettings:
    """Predeclared cohort, expression and localization rules."""

    included_tissues: tuple[str, ...] = ("EuE", "EndoLesion")
    marker_status: str = "marker_supported"
    min_expected_family_cells: int = 20
    min_comparator_cells: int = 100
    min_paired_patients: int = 5
    min_expected_detection_fraction: float = 0.01
    min_positive_patient_fraction: float = 0.70
    fdr_threshold: float = 0.10
    normalization_target: float = 10_000.0

    def validate(self) -> None:
        if self.min_expected_family_cells < 1 or self.min_comparator_cells < 1:
            raise ValueError("Minimum cell counts must be positive")
        if self.min_paired_patients < 2:
            raise ValueError("At least two paired patients are required")
        if not 0 <= self.min_expected_detection_fraction <= 1:
            raise ValueError("Detection threshold must be between zero and one")
        if not 0.5 <= self.min_positive_patient_fraction <= 1:
            raise ValueError("Positive-patient fraction must be in [0.5, 1]")
        if not 0 < self.fdr_threshold < 1:
            raise ValueError("FDR threshold must be between zero and one")


def select_candidate_gene_families(
    bulk_gene_family_results: pd.DataFrame,
) -> pd.DataFrame:
    """Select robust bulk-replicated genes with their discovery families."""

    required = {
        "gene",
        "provisional_cell_family",
        "directionally_replicated",
        "statistically_supported",
    }
    missing = required - set(bulk_gene_family_results.columns)
    if missing:
        raise ValueError(f"Missing bulk-validation columns: {sorted(missing)}")
    selected = bulk_gene_family_results.loc[
        bulk_gene_family_results["directionally_replicated"].eq(True)
    ].copy()
    selected["priority_tier"] = np.where(
        selected["statistically_supported"].eq(True),
        "high_priority_fdr_supported",
        "secondary_directional_replication",
    )
    return selected[
        [
            "gene",
            "provisional_cell_family",
            "priority_tier",
            "statistically_supported",
        ]
    ].drop_duplicates().sort_values(["priority_tier", "gene", "provisional_cell_family"])


def _candidate_expression(
    adata: ad.AnnData,
    genes: list[str],
    settings: CellularValidationSettings,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return log-normalized expression and detection for available candidates."""

    available = [gene for gene in genes if gene in adata.var_names]
    if not available:
        return np.empty((adata.n_obs, 0)), np.empty((adata.n_obs, 0)), available
    indices = [adata.var_names.get_loc(gene) for gene in available]
    counts = adata.layers["counts"][:, indices]
    total = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    if (total <= 0).any():
        raise ValueError("Zero-library cells found in marker-validated artifact")
    if sparse.issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float64)
    normalized = np.log1p(
        counts / total[:, None] * settings.normalization_target
    )
    return normalized, counts > 0, available


def aggregate_supported_expression(
    artifact_paths: list[Path],
    genes: list[str],
    settings: CellularValidationSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate candidate expression within patient, tissue and cell family."""

    settings.validate()
    rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    for path in artifact_paths:
        adata = ad.read_h5ad(path)
        required = {
            "patient_id",
            "tissue_code",
            "marker_validation_status",
            "conservative_cell_family",
        }
        missing = required - set(adata.obs.columns)
        if missing:
            raise ValueError(f"{path.name} missing cell metadata: {sorted(missing)}")
        available = [gene for gene in genes if gene in adata.var_names]
        availability_rows.extend(
            {
                "sample_id": str(adata.obs["sample_id"].iloc[0]),
                "gene": gene,
                "available": gene in available,
            }
            for gene in genes
        )
        mask = (
            adata.obs["marker_validation_status"].astype(str).eq(settings.marker_status)
            & adata.obs["tissue_code"].astype(str).isin(settings.included_tissues)
        ).to_numpy()
        if not mask.any() or not available:
            continue
        subset = adata[mask].copy()
        expression, detected, available = _candidate_expression(
            subset, genes, settings
        )
        grouping = subset.obs[
            ["patient_id", "tissue_code", "conservative_cell_family"]
        ].astype(str)
        for key, indices in grouping.groupby(
            ["patient_id", "tissue_code", "conservative_cell_family"],
            sort=True,
        ).indices.items():
            index = np.asarray(indices)
            n_cells = len(index)
            expression_sum = expression[index].sum(axis=0)
            detected_sum = detected[index].sum(axis=0)
            for gene_index, gene in enumerate(available):
                rows.append(
                    {
                        "patient_id": key[0],
                        "tissue_code": key[1],
                        "cell_family": key[2],
                        "gene": gene,
                        "n_cells": n_cells,
                        "expression_sum": expression_sum[gene_index],
                        "detected_cells": int(detected_sum[gene_index]),
                    }
                )
    if not rows:
        return pd.DataFrame(), pd.DataFrame(availability_rows)
    sample_groups = pd.DataFrame(rows)
    patient_groups = (
        sample_groups.groupby(
            ["patient_id", "tissue_code", "cell_family", "gene"], observed=True
        )
        .agg(
            n_cells=("n_cells", "sum"),
            expression_sum=("expression_sum", "sum"),
            detected_cells=("detected_cells", "sum"),
        )
        .reset_index()
    )
    patient_groups["mean_log1p_cp10k"] = (
        patient_groups["expression_sum"] / patient_groups["n_cells"]
    )
    patient_groups["detection_fraction"] = (
        patient_groups["detected_cells"] / patient_groups["n_cells"]
    )
    return patient_groups, pd.DataFrame(availability_rows)


def evaluate_cell_family_localization(
    patient_groups: pd.DataFrame,
    candidates: pd.DataFrame,
    settings: CellularValidationSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each expected family with other supported cells in each patient."""

    rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        gene_data = patient_groups.loc[patient_groups["gene"].eq(candidate.gene)]
        for tissue in settings.included_tissues:
            tissue_data = gene_data.loc[gene_data["tissue_code"].eq(tissue)]
            expected = tissue_data.loc[
                tissue_data["cell_family"].eq(candidate.provisional_cell_family)
            ].copy()
            other = (
                tissue_data.loc[
                    ~tissue_data["cell_family"].eq(candidate.provisional_cell_family)
                ]
                .groupby("patient_id", observed=True)
                .agg(
                    other_n_cells=("n_cells", "sum"),
                    other_expression_sum=("expression_sum", "sum"),
                    other_detected_cells=("detected_cells", "sum"),
                )
                .reset_index()
            )
            other["other_mean_log1p_cp10k"] = (
                other["other_expression_sum"] / other["other_n_cells"]
            )
            other["other_detection_fraction"] = (
                other["other_detected_cells"] / other["other_n_cells"]
            )
            paired = expected.merge(other, on="patient_id", how="inner")
            paired = paired.loc[
                (paired["n_cells"] >= settings.min_expected_family_cells)
                & (paired["other_n_cells"] >= settings.min_comparator_cells)
            ].copy()
            paired["expression_difference"] = (
                paired["mean_log1p_cp10k"] - paired["other_mean_log1p_cp10k"]
            )
            paired["detection_difference"] = (
                paired["detection_fraction"] - paired["other_detection_fraction"]
            )
            for item in paired.itertuples(index=False):
                paired_rows.append(
                    {
                        "gene": candidate.gene,
                        "expected_cell_family": candidate.provisional_cell_family,
                        "priority_tier": candidate.priority_tier,
                        "tissue_code": tissue,
                        "patient_id": item.patient_id,
                        "expected_n_cells": item.n_cells,
                        "other_n_cells": item.other_n_cells,
                        "expected_mean_log1p_cp10k": item.mean_log1p_cp10k,
                        "other_mean_log1p_cp10k": item.other_mean_log1p_cp10k,
                        "expression_difference": item.expression_difference,
                        "expected_detection_fraction": item.detection_fraction,
                        "other_detection_fraction": item.other_detection_fraction,
                        "detection_difference": item.detection_difference,
                    }
                )
            n_pairs = len(paired)
            eligible = n_pairs >= settings.min_paired_patients
            if eligible and np.any(paired["expression_difference"] != 0):
                statistic, p_value = stats.wilcoxon(
                    paired["expression_difference"],
                    alternative="greater",
                    zero_method="wilcox",
                )
            else:
                statistic, p_value = np.nan, np.nan
            rows.append(
                {
                    "gene": candidate.gene,
                    "expected_cell_family": candidate.provisional_cell_family,
                    "priority_tier": candidate.priority_tier,
                    "tissue_code": tissue,
                    "n_paired_patients": n_pairs,
                    "eligible_for_localization_test": eligible,
                    "median_expected_detection_fraction": (
                        paired["detection_fraction"].median() if n_pairs else np.nan
                    ),
                    "median_other_detection_fraction": (
                        paired["other_detection_fraction"].median() if n_pairs else np.nan
                    ),
                    "median_expected_mean_log1p_cp10k": (
                        paired["mean_log1p_cp10k"].median() if n_pairs else np.nan
                    ),
                    "median_other_mean_log1p_cp10k": (
                        paired["other_mean_log1p_cp10k"].median() if n_pairs else np.nan
                    ),
                    "median_paired_expression_difference": (
                        paired["expression_difference"].median() if n_pairs else np.nan
                    ),
                    "positive_expression_difference_fraction": (
                        (paired["expression_difference"] > 0).mean()
                        if n_pairs
                        else np.nan
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    results = pd.DataFrame(rows)
    results["fdr_bh"] = np.nan
    eligible_mask = results["eligible_for_localization_test"] & results["p_value"].notna()
    if eligible_mask.any():
        results.loc[eligible_mask, "fdr_bh"] = multipletests(
        results.loc[eligible_mask, "p_value"], method="fdr_bh"
        )[1]
    results["directionally_consistent_localization"] = (
        results["eligible_for_localization_test"]
        & results["median_paired_expression_difference"].gt(0)
        & results["median_expected_detection_fraction"].ge(
            settings.min_expected_detection_fraction
        )
        & results["positive_expression_difference_fraction"].ge(
            settings.min_positive_patient_fraction
        )
    )
    results["cell_family_localized"] = (
        results["directionally_consistent_localization"]
        & results["fdr_bh"].lt(settings.fdr_threshold)
    )
    results["localization_tier"] = np.select(
        [
            results["cell_family_localized"],
            results["eligible_for_localization_test"]
            & results["median_paired_expression_difference"].gt(0),
            results["eligible_for_localization_test"],
        ],
        [
            "fdr_supported_cell_family_localization",
            "positive_localization_without_full_support",
            "eligible_but_not_localized",
        ],
        default="insufficient_patient_replication",
    )
    return results, pd.DataFrame(paired_rows)


def summarize_gene_evidence(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether each gene has localization support in either tissue."""

    return (
        results.groupby(["gene", "priority_tier"], observed=True)
        .agg(
            n_expected_family_tissue_tests=("gene", "size"),
            n_eligible_tests=("eligible_for_localization_test", "sum"),
            n_localized_tests=("cell_family_localized", "sum"),
            localized_tissues=(
                "tissue_code",
                lambda values: ";".join(
                    sorted(
                        results.loc[
                            values.index[results.loc[values.index, "cell_family_localized"]],
                            "tissue_code",
                        ].unique()
                    )
                ),
            ),
            best_localization_fdr=("fdr_bh", "min"),
            maximum_positive_patient_fraction=(
                "positive_expression_difference_fraction",
                "max",
            ),
        )
        .reset_index()
    )


def plot_cellular_validation(results: pd.DataFrame, output_dir: Path) -> None:
    """Save candidate localization counts and high-priority effect heatmap."""

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = (
        results.groupby(["tissue_code", "localization_tier"], observed=True)
        .size()
        .reset_index(name="n_tests")
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=counts,
        x="n_tests",
        y="tissue_code",
        hue="localization_tier",
        ax=ax,
    )
    ax.set_title("Cell-family localization evidence by tissue")
    ax.set_xlabel("Gene-family tests")
    ax.set_ylabel("Tissue")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "localization_tier_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    high = results.loc[
        results["priority_tier"].eq("high_priority_fdr_supported")
    ].copy()
    if high.empty:
        return
    high["gene_family"] = high["gene"] + " (" + high["expected_cell_family"] + ")"
    pivot = high.pivot_table(
        index="gene_family",
        columns="tissue_code",
        values="median_paired_expression_difference",
        aggfunc="max",
    )
    fig, ax = plt.subplots(figsize=(8, max(5, 0.45 * len(pivot))))
    sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("High-priority genes: expected-family expression enrichment")
    ax.set_xlabel("Tissue")
    ax.set_ylabel("Discovery gene and expected family")
    fig.tight_layout()
    fig.savefig(
        output_dir / "high_priority_localization_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
