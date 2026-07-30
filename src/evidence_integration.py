"""Transparent cross-dataset evidence integration and signature freezing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.bulk_validation import collapse_discovery_candidates


@dataclass(frozen=True)
class EvidenceIntegrationSettings:
    """Versioned rules for evidence tiers and frozen research signatures."""

    signature_version: str = "v1.0"
    discovery_tier: str = "fdr_effect_direction_stable"
    core_rule: str = (
        "GSE135485_FDR_and_GSE51981_primary_FDR_and_GSE213216_directional_localization"
    )
    extended_rule: str = (
        "GSE135485_robust_direction_and_GSE51981_primary_FDR"
    )
    specificity_rule: str = "GSE51981_other_pathology_same_direction_and_FDR"


def _cellular_gene_evidence(cellular_tests: pd.DataFrame) -> pd.DataFrame:
    """Collapse tissue-specific localization tests to transparent gene evidence."""

    rows = []
    for gene, group in cellular_tests.groupby("gene", sort=True):
        directional = group.loc[
            group["directionally_consistent_localization"].eq(True)
        ]
        formal = group.loc[group["cell_family_localized"].eq(True)]
        rows.append(
            {
                "gene": gene,
                "gse213216_tested_for_localization": True,
                "gse213216_n_eligible_localization_tests": int(
                    group["eligible_for_localization_test"].sum()
                ),
                "gse213216_directional_localization": not directional.empty,
                "gse213216_n_directional_localization_tests": len(directional),
                "gse213216_directional_tissues": ";".join(
                    sorted(directional["tissue_code"].unique())
                ),
                "gse213216_formal_localization": not formal.empty,
                "gse213216_best_localization_fdr": group["fdr_bh"].min(),
                "gse213216_expected_families": ";".join(
                    sorted(group["expected_cell_family"].unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def integrate_evidence(
    discovery: pd.DataFrame,
    bulk_validation: pd.DataFrame,
    cellular_tests: pd.DataFrame,
    microarray_validation: pd.DataFrame,
    settings: EvidenceIntegrationSettings,
) -> pd.DataFrame:
    """Build one provenance-preserving row per stable discovery gene."""

    candidates, conflicts, _ = collapse_discovery_candidates(
        discovery, settings.discovery_tier
    )
    if not conflicts.empty:
        raise ValueError("Direction-conflicting discovery genes cannot be frozen")
    bulk_columns = {
        "gene": "gene",
        "directionally_replicated": "gse135485_robust_directional_replication",
        "statistically_supported": "gse135485_fdr_supported",
        "bulk_adjusted_log2_cpm_difference": "gse135485_adjusted_effect",
        "fdr_bh": "gse135485_fdr",
        "validation_tier": "gse135485_validation_tier",
    }
    micro_columns = {
        "gene": "gene",
        "robust_primary_directional_replication": (
            "gse51981_robust_directional_replication"
        ),
        "primary_statistically_supported": "gse51981_primary_fdr_supported",
        "endometriosis_specific_support": "gse51981_specificity_fdr_supported",
        "primary_adjusted_expression_difference": "gse51981_primary_adjusted_effect",
        "primary_fdr_bh": "gse51981_primary_fdr",
        "other_pathology_adjusted_difference": (
            "gse51981_other_pathology_adjusted_effect"
        ),
        "other_pathology_fdr_bh": "gse51981_other_pathology_fdr",
        "other_pathology_direction_matches_discovery": (
            "gse51981_other_pathology_direction_match"
        ),
        "validation_tier": "gse51981_validation_tier",
    }
    missing_bulk = set(bulk_columns) - set(bulk_validation.columns)
    missing_micro = set(micro_columns) - set(microarray_validation.columns)
    if missing_bulk or missing_micro:
        raise ValueError(
            f"Missing integration fields: bulk={sorted(missing_bulk)}, "
            f"microarray={sorted(missing_micro)}"
        )
    bulk = bulk_validation[list(bulk_columns)].rename(columns=bulk_columns)
    micro = microarray_validation[list(micro_columns)].rename(columns=micro_columns)
    cellular = _cellular_gene_evidence(cellular_tests)
    evidence = (
        candidates.merge(bulk, on="gene", how="left")
        .merge(cellular, on="gene", how="left")
        .merge(micro, on="gene", how="left")
    )
    evidence["signature_version"] = settings.signature_version
    evidence["gse179640_discovery_stable"] = True
    evidence["gse135485_tested"] = evidence[
        "gse135485_validation_tier"
    ].notna()
    evidence["gse51981_tested"] = evidence[
        "gse51981_validation_tier"
    ].notna()
    boolean_columns = [
        "gse135485_robust_directional_replication",
        "gse135485_fdr_supported",
        "gse213216_tested_for_localization",
        "gse213216_directional_localization",
        "gse213216_formal_localization",
        "gse51981_robust_directional_replication",
        "gse51981_primary_fdr_supported",
        "gse51981_specificity_fdr_supported",
        "gse51981_other_pathology_direction_match",
    ]
    for column in boolean_columns:
        evidence[column] = evidence[column].eq(True)

    evidence["core_multistudy_pathology_candidate"] = (
        evidence["gse135485_fdr_supported"]
        & evidence["gse51981_primary_fdr_supported"]
        & evidence["gse213216_directional_localization"]
    )
    evidence["extended_clean_control_signature"] = (
        evidence["gse135485_robust_directional_replication"]
        & evidence["gse51981_primary_fdr_supported"]
    )
    evidence["specificity_watchlist"] = (
        evidence["extended_clean_control_signature"]
        & evidence["gse51981_other_pathology_direction_match"]
        & ~evidence["gse51981_specificity_fdr_supported"]
    )
    evidence["endometriosis_specific_biomarker_supported"] = (
        evidence["extended_clean_control_signature"]
        & evidence["gse51981_specificity_fdr_supported"]
    )
    evidence["prior_gse135485_high_priority"] = evidence[
        "gse135485_fdr_supported"
    ]
    evidence["prior_high_priority_reassessment"] = np.select(
        [
            evidence["core_multistudy_pathology_candidate"],
            evidence["prior_gse135485_high_priority"]
            & evidence["gse51981_primary_fdr_supported"],
            evidence["prior_gse135485_high_priority"]
            & evidence["gse51981_robust_directional_replication"],
            evidence["prior_gse135485_high_priority"],
        ],
        [
            "retained_as_core_multistudy_candidate",
            "microarray_supported_without_cellular_localization",
            "microarray_directional_only",
            "not_replicated_in_gse51981",
        ],
        default="not_in_prior_high_priority_tier",
    )
    evidence["evidence_tier"] = np.select(
        [
            evidence["endometriosis_specific_biomarker_supported"],
            evidence["core_multistudy_pathology_candidate"],
            evidence["extended_clean_control_signature"],
            evidence["gse135485_robust_directional_replication"]
            & evidence["gse51981_robust_directional_replication"],
            evidence["gse135485_robust_directional_replication"],
        ],
        [
            "endometriosis_specific_supported",
            "core_multistudy_pathology_candidate",
            "extended_clean_control_replication",
            "cross_platform_directional_replication",
            "gse135485_directional_replication_only",
        ],
        default="stable_discovery_without_multistudy_replication",
    )
    evidence["n_disease_direction_replication_datasets"] = (
        evidence["gse135485_robust_directional_replication"].astype(int)
        + evidence["gse51981_robust_directional_replication"].astype(int)
    )
    evidence["modeling_role"] = np.select(
        [
            evidence["core_multistudy_pathology_candidate"],
            evidence["specificity_watchlist"],
            evidence["extended_clean_control_signature"],
        ],
        [
            "frozen_core_pathology_signature",
            "specificity_watchlist_only",
            "frozen_extended_clean_control_signature",
        ],
        default="not_in_frozen_signature",
    )
    return evidence.sort_values(
        [
            "core_multistudy_pathology_candidate",
            "extended_clean_control_signature",
            "specificity_watchlist",
            "n_disease_direction_replication_datasets",
            "minimum_discovery_fdr",
        ],
        ascending=[False, False, False, False, True],
    )


def frozen_signature_tables(
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return core, extended and specificity-watchlist tables."""

    core = evidence.loc[
        evidence["core_multistudy_pathology_candidate"]
    ].copy()
    extended = evidence.loc[
        evidence["extended_clean_control_signature"]
    ].copy()
    watchlist = evidence.loc[evidence["specificity_watchlist"]].copy()
    return core, extended, watchlist


def modeling_readiness_decision(
    evidence: pd.DataFrame,
    settings: EvidenceIntegrationSettings,
) -> dict[str, object]:
    """State what may and may not be modeled without overstating evidence."""

    core, extended, watchlist = frozen_signature_tables(evidence)
    specific = evidence.loc[
        evidence["endometriosis_specific_biomarker_supported"]
    ]
    return {
        "signature_version": settings.signature_version,
        "n_stable_discovery_genes": len(evidence),
        "n_core_multistudy_pathology_genes": len(core),
        "n_extended_clean_control_genes": len(extended),
        "n_specificity_watchlist_genes": len(watchlist),
        "n_formally_endometriosis_specific_genes": len(specific),
        "core_genes": core["gene"].tolist(),
        "extended_genes": extended["gene"].tolist(),
        "specificity_watchlist_genes": watchlist["gene"].tolist(),
        "endometriosis_specific_signature_ready": len(specific) > 0,
        "pathology_vs_clean_control_exploratory_signature_ready": len(extended)
        >= 2,
        "deep_learning_model_ready": False,
        "independent_untouched_test_cohort_available": False,
        "recommended_next_model": (
            "patient-level regularized logistic regression benchmark using the "
            "frozen extended signature, with all preprocessing nested inside "
            "cross-validation"
        ),
        "prohibited_claims": [
            "Do not call the extended signature endometriosis-specific.",
            "Do not evaluate on GSE51981 as an untouched test set because its labels contributed to signature selection.",
            "Do not treat cells as independent patient-level training examples.",
            "Do not compare a deep network only against an unregularized or leakage-prone baseline.",
        ],
    }


def plot_evidence_integration(
    evidence: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save evidence-tier counts and an extended-signature evidence heatmap."""

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = (
        evidence["evidence_tier"]
        .value_counts()
        .rename_axis("evidence_tier")
        .reset_index(name="n_genes")
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=counts, x="n_genes", y="evidence_tier", ax=ax)
    ax.set_title("Stable discovery genes by integrated evidence tier")
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "integrated_evidence_tier_counts.png", dpi=180)
    plt.close(fig)

    extended = evidence.loc[
        evidence["extended_clean_control_signature"]
    ].copy()
    if extended.empty:
        return
    heatmap = extended.set_index("gene")[
        [
            "gse179640_discovery_stable",
            "gse135485_robust_directional_replication",
            "gse135485_fdr_supported",
            "gse213216_directional_localization",
            "gse213216_formal_localization",
            "gse51981_robust_directional_replication",
            "gse51981_primary_fdr_supported",
            "gse51981_other_pathology_direction_match",
            "gse51981_specificity_fdr_supported",
        ]
    ].astype(int)
    heatmap = heatmap.rename(
        columns={
            "gse179640_discovery_stable": "Discovery stable",
            "gse135485_robust_directional_replication": "GSE135485 direction",
            "gse135485_fdr_supported": "GSE135485 FDR",
            "gse213216_directional_localization": "GSE213216 localization",
            "gse213216_formal_localization": "GSE213216 localization FDR",
            "gse51981_robust_directional_replication": "GSE51981 direction",
            "gse51981_primary_fdr_supported": "GSE51981 clean-control FDR",
            "gse51981_other_pathology_direction_match": "Other-pathology direction",
            "gse51981_specificity_fdr_supported": "Specificity FDR",
        }
    )
    fig, ax = plt.subplots(figsize=(12, max(5, 0.5 * len(heatmap))))
    sns.heatmap(
        heatmap,
        cmap=["#F1F1F1", "#2A9D8F"],
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar=False,
        annot=True,
        fmt="d",
        ax=ax,
    )
    ax.set_title("Frozen extended signature: evidence across datasets")
    ax.set_xlabel("Evidence criterion")
    ax.set_ylabel("Gene")
    fig.tight_layout()
    fig.savefig(output_dir / "extended_signature_evidence_heatmap.png", dpi=180)
    plt.close(fig)
