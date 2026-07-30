"""Cycle-phase-adjusted GSE51981 validation of predeclared signatures."""

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
class MicroarrayValidationSettings:
    """Predeclared contrasts, robustness rules and multiplicity threshold."""

    disease_group: str = "Endometriosis"
    clean_control_group: str = "Healthy_control_no_pathology"
    other_pathology_group: str = "Non_endometriosis_other_pathology"
    phase_reference: str = "Proliferative"
    fdr_threshold: float = 0.10
    tracked_genes: tuple[str, ...] = ("PTGS1", "MMP10")

    def validate(self) -> None:
        if not 0 < self.fdr_threshold < 1:
            raise ValueError("fdr_threshold must be between zero and one")
        if len(
            {
                self.disease_group,
                self.clean_control_group,
                self.other_pathology_group,
            }
        ) != 3:
            raise ValueError("Clinical contrast groups must be distinct")


def select_predeclared_candidates(
    bulk_validation: pd.DataFrame,
) -> pd.DataFrame:
    """Select the robust GSE135485 directional tier before microarray testing."""

    required = {
        "gene",
        "discovery_direction",
        "directionally_replicated",
        "statistically_supported",
        "discovery_cell_families",
        "bulk_adjusted_log2_cpm_difference",
    }
    missing = required - set(bulk_validation.columns)
    if missing:
        raise ValueError(f"Missing bulk-validation columns: {sorted(missing)}")
    candidates = bulk_validation.loc[
        bulk_validation["directionally_replicated"].eq(True)
    ].copy()
    candidates["prior_evidence_tier"] = np.where(
        candidates["statistically_supported"].eq(True),
        "gse135485_fdr_supported",
        "gse135485_directional_only",
    )
    return candidates[
        [
            "gene",
            "discovery_direction",
            "discovery_cell_families",
            "prior_evidence_tier",
            "bulk_adjusted_log2_cpm_difference",
        ]
    ].drop_duplicates("gene").sort_values(["prior_evidence_tier", "gene"])


def build_phase_adjusted_design(
    metadata: pd.DataFrame,
    disease_group: str,
    comparator_group: str,
    phase_reference: str,
) -> pd.DataFrame:
    """Build an intercept, disease indicator and categorical phase design."""

    required = {"clinical_group", "cycle_phase"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing GSE51981 model metadata: {sorted(missing)}")
    observed = set(metadata["clinical_group"])
    if disease_group not in observed or comparator_group not in observed:
        raise ValueError("Both disease and comparator groups are required")
    phases = sorted(metadata["cycle_phase"].astype(str).unique())
    ordered = [phase_reference] + [
        phase for phase in phases if phase != phase_reference
    ]
    phase = pd.Categorical(
        metadata["cycle_phase"].astype(str), categories=ordered
    )
    phase_terms = pd.get_dummies(
        phase, prefix="phase", drop_first=True, dtype=float
    )
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "endometriosis": metadata["clinical_group"]
            .eq(disease_group)
            .astype(float),
        },
        index=metadata.index,
    )
    design = pd.concat(
        [design, phase_terms.set_axis(metadata.index)], axis=1
    )
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Phase-adjusted contrast design is rank deficient")
    return design


def fit_phase_adjusted_effect(
    expression: pd.Series,
    metadata: pd.DataFrame,
    disease_group: str,
    comparator_group: str,
    phase_reference: str = "Proliferative",
) -> dict[str, float | int]:
    """Fit OLS with HC3 robust errors for disease versus one comparator."""

    subset = metadata.loc[
        metadata["clinical_group"].isin([disease_group, comparator_group])
    ].copy()
    expression = expression.loc[subset["sample_id"]].astype(float)
    design = build_phase_adjusted_design(
        subset, disease_group, comparator_group, phase_reference
    )
    x = design.to_numpy(dtype=float)
    fitted = sm.OLS(expression.to_numpy(dtype=float), x).fit(cov_type="HC3")
    condition_index = design.columns.get_loc("endometriosis")
    return {
        "adjusted_expression_difference": float(fitted.params[condition_index]),
        "hc3_standard_error": float(fitted.bse[condition_index]),
        "hc3_z_statistic": float(fitted.tvalues[condition_index]),
        "p_value": float(fitted.pvalues[condition_index]),
        "n_samples": len(subset),
        "n_endometriosis": int(
            subset["clinical_group"].eq(disease_group).sum()
        ),
        "n_comparator": int(
            subset["clinical_group"].eq(comparator_group).sum()
        ),
        "design_rank": int(np.linalg.matrix_rank(x)),
        "n_design_columns": x.shape[1],
        "design_condition_number": float(np.linalg.cond(x)),
    }


def validate_microarray_candidates(
    candidates: pd.DataFrame,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    qc: pd.DataFrame,
    settings: MicroarrayValidationSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run primary, QC, phase-complete, specificity and LOO analyses."""

    settings.validate()
    available = candidates.loc[candidates["gene"].isin(expression.index)].copy()
    unavailable = candidates.loc[
        ~candidates["gene"].isin(expression.index)
    ].copy()
    unavailable["availability_status"] = "not_in_gpl570_mapped_gene_matrix"
    review_samples = set(
        qc.loc[qc["qc_status"].eq("review"), "sample_id"].astype(str)
    )
    primary_metadata = metadata.loc[
        metadata["clinical_group"].isin(
            [settings.disease_group, settings.clean_control_group]
        )
    ].copy()
    qc_sensitivity = primary_metadata.loc[
        ~primary_metadata["sample_id"].isin(review_samples)
    ].copy()
    phase_complete = primary_metadata.loc[
        ~primary_metadata["cycle_phase"].eq("Unknown")
    ].copy()
    primary_sample_ids = primary_metadata["sample_id"].tolist()

    rows: list[dict[str, object]] = []
    for candidate in available.itertuples(index=False):
        gene_expression = expression.loc[candidate.gene]
        primary = fit_phase_adjusted_effect(
            gene_expression,
            primary_metadata,
            settings.disease_group,
            settings.clean_control_group,
            settings.phase_reference,
        )
        qc_fit = fit_phase_adjusted_effect(
            gene_expression,
            qc_sensitivity,
            settings.disease_group,
            settings.clean_control_group,
            settings.phase_reference,
        )
        phase_fit = fit_phase_adjusted_effect(
            gene_expression,
            phase_complete,
            settings.disease_group,
            settings.clean_control_group,
            settings.phase_reference,
        )
        specificity = fit_phase_adjusted_effect(
            gene_expression,
            metadata,
            settings.disease_group,
            settings.other_pathology_group,
            settings.phase_reference,
        )
        loo_effects: dict[str, float] = {}
        for sample_id in primary_sample_ids:
            loo_metadata = primary_metadata.loc[
                ~primary_metadata["sample_id"].eq(sample_id)
            ]
            loo_fit = fit_phase_adjusted_effect(
                gene_expression,
                loo_metadata,
                settings.disease_group,
                settings.clean_control_group,
                settings.phase_reference,
            )
            loo_effects[sample_id] = float(
                loo_fit["adjusted_expression_difference"]
            )
        expected_sign = (
            1
            if candidate.discovery_direction == "higher_in_endometriosis"
            else -1
        )
        loo_values = np.asarray(list(loo_effects.values()))
        row = candidate._asdict()
        row.update({f"primary_{key}": value for key, value in primary.items()})
        row.update(
            {
                "qc_exclusion_effect": qc_fit["adjusted_expression_difference"],
                "phase_complete_effect": phase_fit["adjusted_expression_difference"],
                "other_pathology_adjusted_difference": specificity[
                    "adjusted_expression_difference"
                ],
                "other_pathology_hc3_standard_error": specificity[
                    "hc3_standard_error"
                ],
                "other_pathology_p_value": specificity["p_value"],
                "primary_direction_matches_discovery": bool(
                    np.sign(primary["adjusted_expression_difference"])
                    == expected_sign
                ),
                "qc_exclusion_direction_matches_discovery": bool(
                    np.sign(qc_fit["adjusted_expression_difference"])
                    == expected_sign
                ),
                "phase_complete_direction_matches_discovery": bool(
                    np.sign(phase_fit["adjusted_expression_difference"])
                    == expected_sign
                ),
                "other_pathology_direction_matches_discovery": bool(
                    np.sign(specificity["adjusted_expression_difference"])
                    == expected_sign
                ),
                "leave_one_sample_out_direction_stability": float(
                    np.mean(np.sign(loo_values) == expected_sign)
                ),
                "minimum_leave_one_out_effect": float(loo_values.min()),
                "maximum_leave_one_out_effect": float(loo_values.max()),
            }
        )
        rows.append(row)
    results = pd.DataFrame(rows)
    if results.empty:
        return results, unavailable
    results["primary_fdr_bh"] = multipletests(
        results["primary_p_value"].fillna(1.0), method="fdr_bh"
    )[1]
    results["other_pathology_fdr_bh"] = multipletests(
        results["other_pathology_p_value"].fillna(1.0), method="fdr_bh"
    )[1]
    results["robust_primary_directional_replication"] = (
        results["primary_direction_matches_discovery"]
        & results["qc_exclusion_direction_matches_discovery"]
        & results["phase_complete_direction_matches_discovery"]
        & results["leave_one_sample_out_direction_stability"].eq(1.0)
    )
    results["primary_statistically_supported"] = (
        results["robust_primary_directional_replication"]
        & results["primary_fdr_bh"].lt(settings.fdr_threshold)
    )
    results["endometriosis_specific_support"] = (
        results["primary_statistically_supported"]
        & results["other_pathology_direction_matches_discovery"]
        & results["other_pathology_fdr_bh"].lt(settings.fdr_threshold)
    )
    results["validation_tier"] = np.select(
        [
            results["endometriosis_specific_support"],
            results["primary_statistically_supported"],
            results["robust_primary_directional_replication"],
            results["primary_direction_matches_discovery"],
        ],
        [
            "primary_and_other_pathology_fdr_supported",
            "clean_control_fdr_supported",
            "robust_directional_replication",
            "primary_model_direction_only",
        ],
        default="direction_not_replicated",
    )
    return results.sort_values(
        [
            "endometriosis_specific_support",
            "primary_statistically_supported",
            "robust_primary_directional_replication",
            "primary_fdr_bh",
        ],
        ascending=[False, False, False, True],
    ), unavailable


def track_predefined_microarray_genes(
    genes: tuple[str, ...],
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    settings: MicroarrayValidationSettings,
) -> pd.DataFrame:
    """Fit named exploratory genes outside the predeclared candidate family."""

    rows = []
    for gene in genes:
        row: dict[str, object] = {
            "gene": gene,
            "available": gene in expression.index,
            "analysis_tier": "exploratory_predefined_not_candidate_set",
        }
        if gene in expression.index:
            primary = fit_phase_adjusted_effect(
                expression.loc[gene],
                metadata,
                settings.disease_group,
                settings.clean_control_group,
                settings.phase_reference,
            )
            specificity = fit_phase_adjusted_effect(
                expression.loc[gene],
                metadata,
                settings.disease_group,
                settings.other_pathology_group,
                settings.phase_reference,
            )
            row.update(
                {
                    "clean_control_adjusted_difference": primary[
                        "adjusted_expression_difference"
                    ],
                    "clean_control_p_value": primary["p_value"],
                    "other_pathology_adjusted_difference": specificity[
                        "adjusted_expression_difference"
                    ],
                    "other_pathology_p_value": specificity["p_value"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_microarray_validation(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save cross-platform effect and high-priority contrast plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=results,
        x="bulk_adjusted_log2_cpm_difference",
        y="primary_adjusted_expression_difference",
        hue="validation_tier",
        style="prior_evidence_tier",
        alpha=0.85,
        ax=ax,
    )
    ax.axhline(0, color="grey", linewidth=1)
    ax.axvline(0, color="grey", linewidth=1)
    ax.set_xlabel("GSE135485 adjusted bulk effect")
    ax.set_ylabel("GSE51981 cycle-adjusted microarray effect")
    ax.set_title("Cross-platform candidate effects")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "bulk_vs_microarray_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    high = results.loc[
        results["prior_evidence_tier"].eq("gse135485_fdr_supported")
    ].copy()
    long = high.melt(
        id_vars=["gene"],
        value_vars=[
            "primary_adjusted_expression_difference",
            "other_pathology_adjusted_difference",
        ],
        var_name="contrast",
        value_name="adjusted_difference",
    )
    long["contrast"] = long["contrast"].map(
        {
            "primary_adjusted_expression_difference": (
                "Endometriosis vs clean control"
            ),
            "other_pathology_adjusted_difference": (
                "Endometriosis vs other pathology"
            ),
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=long,
        x="gene",
        y="adjusted_difference",
        hue="contrast",
        ax=ax,
    )
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_ylabel("Cycle-phase-adjusted expression difference")
    ax.set_title("High-priority candidates in GSE51981")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "high_priority_candidate_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
