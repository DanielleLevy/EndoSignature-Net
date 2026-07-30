"""Patient-level pseudobulk signature discovery for GSE179640.

Raw counts are summed within sample and provisional cell family. Differential
expression is then evaluated across patient-level pseudobulks, never across
cells as if they were independent biological replicates.
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
class PseudobulkSettings:
    """Eligibility, normalization and exploratory-signature settings."""

    min_cells_per_pseudobulk: int = 10
    min_control_patients: int = 3
    min_endometriosis_patients: int = 6
    min_cpm: float = 1.0
    min_samples_expressing: int = 3
    cpm_pseudocount: float = 0.5
    effect_size_threshold: float = 1.0
    fdr_threshold: float = 0.10
    top_genes_per_family: int = 50

    def validate(self) -> None:
        if self.min_cells_per_pseudobulk < 1:
            raise ValueError("min_cells_per_pseudobulk must be positive")
        if self.min_control_patients < 2 or self.min_endometriosis_patients < 2:
            raise ValueError("At least two patients per condition are required")
        if self.min_cpm < 0 or self.min_samples_expressing < 1:
            raise ValueError("Expression filtering settings are invalid")
        if self.cpm_pseudocount <= 0 or self.top_genes_per_family < 1:
            raise ValueError("Pseudocount and top-gene count must be positive")


def aggregate_pseudobulk(
    atlas: ad.AnnData,
    settings: PseudobulkSettings,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Sum raw counts by sample and cell family and report group eligibility."""

    settings.validate()
    if "counts" not in atlas.layers:
        raise ValueError("Atlas must contain raw counts in layers['counts']")
    required = {
        "sample_id",
        "patient_id",
        "condition",
        "provisional_cell_family",
    }
    missing = required - set(atlas.obs.columns)
    if missing:
        raise ValueError(f"Missing pseudobulk metadata: {sorted(missing)}")

    group_columns = [
        "sample_id",
        "patient_id",
        "condition",
        "provisional_cell_family",
    ]
    observed = atlas.obs[group_columns].astype(str)
    grouped = observed.groupby(group_columns, sort=True).indices
    count_matrix = atlas.layers["counts"]
    rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    for key, indices in grouped.items():
        n_cells = len(indices)
        if n_cells < settings.min_cells_per_pseudobulk:
            continue
        summed = count_matrix[indices].sum(axis=0)
        rows.append(np.asarray(summed).ravel())
        metadata_rows.append(
            {
                **dict(zip(group_columns, key)),
                "n_cells": n_cells,
            }
        )
    matrix = np.vstack(rows)
    metadata = pd.DataFrame(metadata_rows)
    metadata["library_size"] = matrix.sum(axis=1)

    counts = (
        metadata.groupby(["provisional_cell_family", "condition"], observed=True)[
            "patient_id"
        ]
        .nunique()
        .unstack(fill_value=0)
    )
    eligibility_rows = []
    all_families = sorted(atlas.obs["provisional_cell_family"].astype(str).unique())
    for family in all_families:
        control = int(counts.loc[family, "Control"]) if family in counts.index else 0
        endometriosis = (
            int(counts.loc[family, "Endometriosis"]) if family in counts.index else 0
        )
        eligible = (
            control >= settings.min_control_patients
            and endometriosis >= settings.min_endometriosis_patients
        )
        eligibility_rows.append(
            {
                "provisional_cell_family": family,
                "n_control_patients": control,
                "n_endometriosis_patients": endometriosis,
                "eligible_for_pseudobulk_de": eligible,
                "reason": (
                    "meets_predeclared_patient_replication_rule"
                    if eligible
                    else "insufficient_patient_replication_after_minimum_cell_rule"
                ),
            }
        )
    return matrix, metadata, pd.DataFrame(eligibility_rows)


def differential_expression_by_family(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    gene_names: pd.Index,
    eligibility: pd.DataFrame,
    settings: PseudobulkSettings,
) -> pd.DataFrame:
    """Run vectorized Welch tests on patient-level log2-CPM pseudobulks."""

    results: list[pd.DataFrame] = []
    eligible_families = eligibility.loc[
        eligibility["eligible_for_pseudobulk_de"], "provisional_cell_family"
    ]
    for family in eligible_families:
        family_mask = metadata["provisional_cell_family"] == family
        family_matrix = matrix[family_mask]
        family_metadata = metadata.loc[family_mask].reset_index(drop=True)
        library_sizes = family_matrix.sum(axis=1)
        cpm = family_matrix / library_sizes[:, None] * 1_000_000.0
        keep = (cpm >= settings.min_cpm).sum(axis=0) >= settings.min_samples_expressing
        log_cpm = np.log2(cpm[:, keep] + settings.cpm_pseudocount)
        control = family_metadata["condition"].to_numpy() == "Control"
        disease = family_metadata["condition"].to_numpy() == "Endometriosis"
        statistic, p_value = stats.ttest_ind(
            log_cpm[disease],
            log_cpm[control],
            axis=0,
            equal_var=False,
            nan_policy="omit",
        )
        log2_fc = log_cpm[disease].mean(axis=0) - log_cpm[control].mean(axis=0)
        full_direction = np.sign(log2_fc)
        control_values = log_cpm[control]
        disease_values = log_cpm[disease]
        control_loo_effects = np.vstack(
            [
                disease_values.mean(axis=0)
                - np.delete(control_values, index, axis=0).mean(axis=0)
                for index in range(control_values.shape[0])
            ]
        )
        disease_loo_effects = np.vstack(
            [
                np.delete(disease_values, index, axis=0).mean(axis=0)
                - control_values.mean(axis=0)
                for index in range(disease_values.shape[0])
            ]
        )
        control_loo_stability = np.mean(
            np.sign(control_loo_effects) == full_direction, axis=0
        )
        disease_loo_stability = np.mean(
            np.sign(disease_loo_effects) == full_direction, axis=0
        )
        adjusted = multipletests(
            np.nan_to_num(p_value, nan=1.0), method="fdr_bh"
        )[1]
        result = pd.DataFrame(
            {
                "study_id": "GSE179640",
                "analysis_scope": "exploratory_patient_level_pseudobulk",
                "provisional_cell_family": family,
                "gene": gene_names[keep],
                "log2_fold_change": log2_fc,
                "welch_t_statistic": statistic,
                "p_value": p_value,
                "fdr_bh": adjusted,
                "mean_log2_cpm_control": log_cpm[control].mean(axis=0),
                "mean_log2_cpm_endometriosis": log_cpm[disease].mean(axis=0),
                "n_control_patients": int(control.sum()),
                "n_endometriosis_patients": int(disease.sum()),
                "control_leave_one_out_direction_stability": control_loo_stability,
                "endometriosis_leave_one_out_direction_stability": disease_loo_stability,
                "minimum_absolute_control_loo_effect": np.min(
                    np.abs(control_loo_effects), axis=0
                ),
            }
        )
        result["direction"] = np.where(
            result["log2_fold_change"] > 0,
            "higher_in_endometriosis",
            "lower_in_endometriosis",
        )
        result["statistical_tier"] = np.select(
            [
                (result["fdr_bh"] < settings.fdr_threshold)
                & (result["log2_fold_change"].abs() >= settings.effect_size_threshold)
                & result["control_leave_one_out_direction_stability"].eq(1.0)
                & result["endometriosis_leave_one_out_direction_stability"].eq(1.0),
                (result["fdr_bh"] < settings.fdr_threshold)
                & (result["log2_fold_change"].abs() >= settings.effect_size_threshold),
                result["log2_fold_change"].abs() >= settings.effect_size_threshold,
            ],
            [
                "fdr_effect_direction_stable",
                "fdr_effect_but_loo_unstable",
                "effect_size_candidate",
            ],
            default="low_effect_exploratory",
        )
        results.append(result)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def select_exploratory_signatures(
    differential_expression: pd.DataFrame,
    settings: PseudobulkSettings,
) -> pd.DataFrame:
    """Select transparent top-ranked candidates without inventing significance."""

    ranked = differential_expression.sort_values(
        ["provisional_cell_family", "fdr_bh", "log2_fold_change"],
        ascending=[True, True, False],
    ).copy()
    ranked["absolute_log2_fold_change"] = ranked["log2_fold_change"].abs()
    ranked = ranked.sort_values(
        ["provisional_cell_family", "fdr_bh", "absolute_log2_fold_change"],
        ascending=[True, True, False],
    )
    selected = (
        ranked.groupby("provisional_cell_family", observed=True)
        .head(settings.top_genes_per_family)
        .copy()
    )
    selected["rank_within_family"] = (
        selected.groupby("provisional_cell_family", observed=True).cumcount() + 1
    )
    return selected


def plot_pseudobulk_results(
    differential_expression: pd.DataFrame,
    eligibility: pd.DataFrame,
    output_dir: Path,
    settings: PseudobulkSettings,
) -> None:
    """Save eligibility and per-family exploratory volcano plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 6))
    long = eligibility.melt(
        id_vars=["provisional_cell_family", "eligible_for_pseudobulk_de"],
        value_vars=["n_control_patients", "n_endometriosis_patients"],
        var_name="condition",
        value_name="n_patients",
    )
    sns.barplot(
        data=long,
        x="provisional_cell_family",
        y="n_patients",
        hue="condition",
        ax=axis,
    )
    axis.tick_params(axis="x", rotation=75)
    axis.axhline(settings.min_control_patients, color="grey", linestyle="--")
    axis.set_title("Patient replication available for cell-family pseudobulk")
    fig.tight_layout()
    fig.savefig(output_dir / "pseudobulk_patient_replication.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for family, frame in differential_expression.groupby(
        "provisional_cell_family", observed=True
    ):
        plot = frame.copy()
        plot["minus_log10_fdr"] = -np.log10(plot["fdr_bh"].clip(lower=1e-300))
        plot["effect_candidate"] = (
            plot["log2_fold_change"].abs() >= settings.effect_size_threshold
        )
        fig, axis = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=plot,
            x="log2_fold_change",
            y="minus_log10_fdr",
            hue="effect_candidate",
            palette={False: "#B8B8B8", True: "#E45756"},
            s=14,
            linewidth=0,
            ax=axis,
        )
        axis.axvline(settings.effect_size_threshold, color="grey", linestyle="--")
        axis.axvline(-settings.effect_size_threshold, color="grey", linestyle="--")
        axis.set_title(f"Exploratory pseudobulk: {family}")
        axis.set_xlabel("Mean log2-CPM difference (Endometriosis - Control)")
        axis.set_ylabel("-log10 BH FDR")
        fig.tight_layout()
        safe_name = family.lower().replace("/", "_")
        fig.savefig(output_dir / f"volcano_{safe_name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
