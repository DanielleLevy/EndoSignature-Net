"""Conservative tissue-level validation of single-cell discovery signatures.

The validation target is fixed before inspecting GSE135485 associations:
patient-level, direction-stable GSE179640 pseudobulk candidates. Because bulk
RNA-seq does not preserve cell identity, repeated gene-family discoveries are
collapsed to a gene-level consensus and later expanded for interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class BulkValidationSettings:
    """Predeclared model, multiplicity and robustness settings."""

    discovery_tier: str = "fdr_effect_direction_stable"
    outlier_sample_id: str = "GSM4012536"
    fdr_threshold: float = 0.10
    condition_reference: str = "Control"
    lane_reference: str = "L001"
    tracked_genes: tuple[str, ...] = ("PTGS1", "MMP10")

    def validate(self) -> None:
        if not 0 < self.fdr_threshold < 1:
            raise ValueError("fdr_threshold must be between zero and one")
        if not self.discovery_tier or not self.outlier_sample_id:
            raise ValueError("Discovery tier and outlier sample must be defined")


def collapse_discovery_candidates(
    differential_expression: pd.DataFrame,
    discovery_tier: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse stable gene-family results into direction-consistent genes.

    Returns the primary consensus genes, any genes with conflicting discovery
    directions, and the original eligible gene-family rows.
    """

    required = {
        "gene",
        "provisional_cell_family",
        "direction",
        "log2_fold_change",
        "fdr_bh",
        "statistical_tier",
    }
    missing = required - set(differential_expression.columns)
    if missing:
        raise ValueError(f"Missing discovery columns: {sorted(missing)}")
    eligible = differential_expression.loc[
        differential_expression["statistical_tier"].eq(discovery_tier)
    ].copy()
    if eligible.empty:
        raise ValueError(f"No discovery rows found for tier: {discovery_tier}")

    rows: list[dict[str, object]] = []
    conflicting: list[dict[str, object]] = []
    for gene, group in eligible.groupby("gene", sort=True):
        directions = sorted(group["direction"].dropna().unique())
        payload = {
            "gene": gene,
            "discovery_direction": directions[0] if len(directions) == 1 else "conflicting",
            "n_discovery_cell_families": group["provisional_cell_family"].nunique(),
            "discovery_cell_families": ";".join(
                sorted(group["provisional_cell_family"].unique())
            ),
            "median_discovery_log2_fold_change": group["log2_fold_change"].median(),
            "maximum_absolute_discovery_log2_fold_change": group[
                "log2_fold_change"
            ].abs().max(),
            "minimum_discovery_fdr": group["fdr_bh"].min(),
        }
        if len(directions) == 1:
            rows.append(payload)
        else:
            payload["observed_directions"] = ";".join(directions)
            conflicting.append(payload)
    return pd.DataFrame(rows), pd.DataFrame(conflicting), eligible


def build_design_matrix(
    metadata: pd.DataFrame,
    lane_reference: str = "L001",
) -> pd.DataFrame:
    """Build a fixed condition-plus-lane design with an explicit intercept."""

    required = {"condition", "lane"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing model metadata: {sorted(missing)}")
    if not {"Control", "Endometriosis"}.issubset(set(metadata["condition"])):
        raise ValueError("Both Control and Endometriosis samples are required")
    lanes = sorted(metadata["lane"].astype(str).unique())
    ordered_lanes = [lane_reference] + [lane for lane in lanes if lane != lane_reference]
    lane = pd.Categorical(metadata["lane"].astype(str), categories=ordered_lanes)
    lane_terms = pd.get_dummies(lane, prefix="lane", drop_first=True, dtype=float)
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "endometriosis": metadata["condition"].eq("Endometriosis").astype(float),
        },
        index=metadata.index,
    )
    return pd.concat([design, lane_terms.set_axis(metadata.index)], axis=1)


def fit_hc3_condition_effect(
    expression: pd.Series,
    metadata: pd.DataFrame,
    lane_reference: str = "L001",
) -> dict[str, float | int]:
    """Fit OLS with HC3 robust standard errors and return the disease effect."""

    expression = expression.loc[metadata["count_matrix_column"]].astype(float)
    design = build_design_matrix(metadata, lane_reference)
    x = design.to_numpy(dtype=float)
    y = expression.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise ValueError("Condition-plus-lane design matrix is rank deficient")
    condition_index = design.columns.get_loc("endometriosis")
    fitted = sm.OLS(y, x).fit(cov_type="HC3")
    coefficient = float(fitted.params[condition_index])
    standard_error = float(fitted.bse[condition_index])
    statistic = float(fitted.tvalues[condition_index])
    p_value = float(fitted.pvalues[condition_index])
    return {
        "bulk_adjusted_log2_cpm_difference": coefficient,
        "hc3_standard_error": standard_error,
        "hc3_z_statistic": statistic,
        "p_value": p_value,
        "n_samples": len(metadata),
        "n_controls": int(metadata["condition"].eq("Control").sum()),
        "n_endometriosis": int(metadata["condition"].eq("Endometriosis").sum()),
        "design_rank": rank,
        "n_design_columns": x.shape[1],
        "design_condition_number": float(np.linalg.cond(x)),
    }


def validate_candidates(
    candidates: pd.DataFrame,
    log_cpm: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: BulkValidationSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test candidates and require direction robustness across sensitivities."""

    settings.validate()
    required_metadata = {"sample_id", "count_matrix_column", "condition", "lane"}
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing validation metadata: {sorted(missing)}")
    if not set(metadata["count_matrix_column"]).issubset(log_cpm.columns):
        raise ValueError("Validation metadata and expression columns do not align")

    available = candidates.loc[candidates["gene"].isin(log_cpm.index)].copy()
    unavailable = candidates.loc[~candidates["gene"].isin(log_cpm.index)].copy()
    unavailable["availability_status"] = "not_in_filtered_gse135485_expression"
    primary_metadata = metadata.loc[
        ~metadata["sample_id"].eq(settings.outlier_sample_id)
    ].copy()
    controls = primary_metadata.loc[
        primary_metadata["condition"].eq("Control"), "sample_id"
    ].tolist()
    if len(controls) < 3:
        raise ValueError("At least three primary control samples are required")

    rows: list[dict[str, object]] = []
    for candidate in available.itertuples(index=False):
        expression = log_cpm.loc[candidate.gene]
        main = fit_hc3_condition_effect(
            expression, primary_metadata, settings.lane_reference
        )
        included_outlier = fit_hc3_condition_effect(
            expression, metadata, settings.lane_reference
        )
        loo_effects: dict[str, float] = {}
        for control_id in controls:
            subset = primary_metadata.loc[
                ~primary_metadata["sample_id"].eq(control_id)
            ]
            fit = fit_hc3_condition_effect(
                expression, subset, settings.lane_reference
            )
            loo_effects[control_id] = float(
                fit["bulk_adjusted_log2_cpm_difference"]
            )
        expected_sign = 1 if candidate.discovery_direction == "higher_in_endometriosis" else -1
        main_effect = float(main["bulk_adjusted_log2_cpm_difference"])
        sensitivity_effect = float(
            included_outlier["bulk_adjusted_log2_cpm_difference"]
        )
        loo_values = np.asarray(list(loo_effects.values()))
        row = candidate._asdict()
        row.update(main)
        row.update(
            {
                "outlier_included_effect": sensitivity_effect,
                "main_direction_matches_discovery": bool(
                    np.sign(main_effect) == expected_sign
                ),
                "outlier_included_direction_matches_discovery": bool(
                    np.sign(sensitivity_effect) == expected_sign
                ),
                "control_loo_direction_stability": float(
                    np.mean(np.sign(loo_values) == expected_sign)
                ),
                "minimum_control_loo_effect": float(loo_values.min()),
                "maximum_control_loo_effect": float(loo_values.max()),
                "control_loo_effects": ";".join(
                    f"{sample_id}:{effect:.6g}"
                    for sample_id, effect in loo_effects.items()
                ),
            }
        )
        rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        return results, unavailable
    results["fdr_bh"] = multipletests(
        results["p_value"].fillna(1.0), method="fdr_bh"
    )[1]
    results["directionally_replicated"] = (
        results["main_direction_matches_discovery"]
        & results["outlier_included_direction_matches_discovery"]
        & results["control_loo_direction_stability"].eq(1.0)
    )
    results["statistically_supported"] = (
        results["directionally_replicated"]
        & results["fdr_bh"].lt(settings.fdr_threshold)
    )
    results["validation_tier"] = np.select(
        [
            results["statistically_supported"],
            results["directionally_replicated"],
            results["main_direction_matches_discovery"],
        ],
        [
            "statistically_supported_and_robust",
            "robust_directional_replication",
            "main_model_direction_only",
        ],
        default="direction_not_replicated",
    )
    return results.sort_values(
        ["statistically_supported", "directionally_replicated", "fdr_bh"],
        ascending=[False, False, True],
    ), unavailable


def expand_gene_family_results(
    validation: pd.DataFrame,
    discovery_gene_family_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one bulk gene-level result to each discovery cell-family row."""

    validation_columns = [
        "gene",
        "bulk_adjusted_log2_cpm_difference",
        "hc3_standard_error",
        "p_value",
        "fdr_bh",
        "directionally_replicated",
        "statistically_supported",
        "validation_tier",
    ]
    return discovery_gene_family_rows.merge(
        validation[validation_columns], on="gene", how="left", suffixes=("_discovery", "_bulk")
    )


def track_predefined_genes(
    genes: tuple[str, ...],
    discovery: pd.DataFrame,
    log_cpm: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: BulkValidationSettings,
) -> pd.DataFrame:
    """Report named biological genes separately from the primary candidate set."""

    primary_metadata = metadata.loc[
        ~metadata["sample_id"].eq(settings.outlier_sample_id)
    ]
    rows = []
    for gene in genes:
        discovery_rows = discovery.loc[discovery["gene"].eq(gene)]
        row: dict[str, object] = {
            "gene": gene,
            "in_bulk_filtered_expression": gene in log_cpm.index,
            "n_discovery_rows": len(discovery_rows),
            "best_discovery_tier": (
                ";".join(sorted(discovery_rows["statistical_tier"].unique()))
                if not discovery_rows.empty
                else "not_tested_after_discovery_expression_filter"
            ),
        }
        if gene in log_cpm.index:
            row.update(
                fit_hc3_condition_effect(
                    log_cpm.loc[gene], primary_metadata, settings.lane_reference
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_validation_results(results: pd.DataFrame, output_dir: Path) -> None:
    """Save discovery-versus-bulk and validation-tier summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data = results.copy()
    plot_data["discovery_effect_for_plot"] = plot_data[
        "median_discovery_log2_fold_change"
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=plot_data,
        x="discovery_effect_for_plot",
        y="bulk_adjusted_log2_cpm_difference",
        hue="validation_tier",
        alpha=0.8,
        ax=ax,
    )
    ax.axhline(0, color="grey", linewidth=1)
    ax.axvline(0, color="grey", linewidth=1)
    ax.set_xlabel("GSE179640 median cell-family log2 fold change")
    ax.set_ylabel("GSE135485 lane-adjusted bulk log2-CPM difference")
    ax.set_title("Single-cell discovery effects versus bulk tissue effects")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "discovery_vs_bulk_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    counts = (
        results["validation_tier"]
        .value_counts()
        .rename_axis("validation_tier")
        .reset_index(name="n_genes")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=counts, x="n_genes", y="validation_tier", ax=ax)
    ax.set_title("Candidate genes by tissue-level validation tier")
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_tier_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
