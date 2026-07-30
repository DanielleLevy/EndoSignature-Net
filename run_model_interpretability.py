"""Run the frozen-model interpretability and cross-cohort heterogeneity audit."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.model_interpretability import (
    InterpretabilitySettings,
    coefficient_bootstrap_stability,
    cohort_gene_effects,
    fit_full_model_and_contributions,
    plot_interpretability_summary,
    repeated_cv_permutation_importance,
)


REPORT_DIR = Path("output/reports/model_interpretability/v1.0")
PLOT_DIR = Path("output/eda_plots/model_interpretability/v1.0")
SIGNATURE_PATH = Path(
    "output/reports/evidence_integration/v1.0/"
    "frozen_extended_clean_control_signature.csv"
)


def _standardize(expression: pd.DataFrame) -> pd.DataFrame:
    return expression.sub(expression.mean(axis=1), axis=0).div(
        expression.std(axis=1, ddof=0), axis=0
    )


def _load_training(genes: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    expression = pd.read_csv(
        "output/reports/microarray/GSE51981/gcrma_gene_expression.csv.gz",
        index_col=0,
    )
    metadata = pd.read_csv(
        "output/reports/microarray/GSE51981/verified_sample_metadata.csv"
    )
    metadata = metadata.loc[
        metadata["clinical_group"].isin(
            ["Endometriosis", "Healthy_control_no_pathology"]
        )
    ]
    target = metadata.set_index("sample_id")["clinical_group"].map(
        {"Healthy_control_no_pathology": 0, "Endometriosis": 1}
    )
    return expression.loc[genes, target.index], target


def _load_external(
    genes: list[str], study_id: str, expression_name: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    intake_dir = Path(
        f"output/reports/external_validation/v1.0/{study_id}_intake"
    )
    expression_all = pd.read_csv(intake_dir / expression_name, index_col=0)
    metadata = pd.read_csv(intake_dir / "verified_sample_metadata.csv")
    if "include_in_external_target" in metadata.columns:
        metadata = metadata.loc[metadata["include_in_external_target"].astype(bool)]
    if study_id == "GSE212787":
        mapping = pd.read_csv(
            intake_dir / "frozen_gene_ensembl_mapping.csv"
        ).set_index("gene")
        expression = pd.DataFrame(
            {
                gene: expression_all.loc[mapping.loc[gene, "ensembl_gene_id"]]
                for gene in genes
            }
        ).T
        expression.index.name = "gene"
    else:
        expression = expression_all.loc[genes]
    target = metadata.set_index("sample_id")["condition"].map(
        {"Disease_free_control": 0, "Endometriosis": 1}
    )
    expression = expression.loc[:, target.index]
    return expression, target, metadata


def main() -> None:
    settings = InterpretabilitySettings()
    signature = pd.read_csv(SIGNATURE_PATH)
    genes = signature["gene"].tolist()
    frozen_direction = signature.set_index("gene")["discovery_direction"]

    training_expression, training_target = _load_training(genes)
    gse212787_expression, gse212787_target, gse212787_metadata = _load_external(
        genes, "GSE212787", "target_log_cpm.csv.gz"
    )
    gse153740_expression, gse153740_target, gse153740_metadata = _load_external(
        genes, "GSE153740", "frozen_signature_log2_fpkm.csv.gz"
    )

    stability = coefficient_bootstrap_stability(
        training_expression, training_target, settings
    )
    permutation = repeated_cv_permutation_importance(
        training_expression, training_target, settings
    )
    coefficients, _ = fit_full_model_and_contributions(
        training_expression, training_target, training_expression, settings
    )
    gene_summary = (
        coefficients.merge(stability, on="gene")
        .merge(permutation, on="gene")
        .merge(
            frozen_direction.rename("frozen_discovery_direction"),
            left_on="gene",
            right_index=True,
        )
    )
    gene_summary["coefficient_direction"] = gene_summary[
        "full_model_coefficient"
    ].map(lambda value: "higher_risk" if value > 0 else "lower_risk")

    effect_frames = [
        cohort_gene_effects(
            _standardize(training_expression), training_target, "GSE51981"
        ),
        cohort_gene_effects(
            _standardize(gse212787_expression), gse212787_target, "GSE212787"
        ),
        cohort_gene_effects(
            _standardize(gse153740_expression), gse153740_target, "GSE153740"
        ),
    ]
    effects = pd.concat(effect_frames, ignore_index=True)
    wide_effects = effects.pivot(
        index="gene",
        columns="cohort",
        values="case_control_standardized_mean_difference",
    )
    expected_sign = frozen_direction.map(
        {"higher_in_endometriosis": 1, "lower_in_endometriosis": -1}
    )
    for cohort in ["GSE51981", "GSE212787", "GSE153740"]:
        gene_summary[f"{cohort}_direction_match"] = (
            wide_effects.loc[gene_summary["gene"], cohort].to_numpy() * expected_sign.loc[
                gene_summary["gene"]
            ].to_numpy()
        ) > 0
    gene_summary["external_direction_matches"] = (
        gene_summary[["GSE212787_direction_match", "GSE153740_direction_match"]]
        .sum(axis=1)
        .astype(int)
    )

    contribution_frames = []
    for study_id, expression, target, metadata in [
        ("GSE212787", gse212787_expression, gse212787_target, gse212787_metadata),
        ("GSE153740", gse153740_expression, gse153740_target, gse153740_metadata),
    ]:
        _, contribution = fit_full_model_and_contributions(
            training_expression, training_target, expression, settings
        )
        contribution["study_id"] = study_id
        contribution = contribution.merge(
            metadata[["sample_id", "condition", "cycle_phase"]],
            on="sample_id",
            how="left",
        )
        contribution_frames.append(contribution)
    contributions = pd.concat(contribution_frames, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gene_summary.to_csv(REPORT_DIR / "gene_importance_and_stability.csv", index=False)
    effects.to_csv(REPORT_DIR / "cross_cohort_gene_effects.csv", index=False)
    contributions.to_csv(
        REPORT_DIR / "external_sample_gene_contributions.csv.gz",
        index=False,
        compression="gzip",
    )

    stable_sign_count = int((gene_summary["sign_stability"] >= 0.95).sum())
    both_external_count = int((gene_summary["external_direction_matches"] == 2).sum())
    top_gene = gene_summary.sort_values(
        "permutation_auc_decrease_mean", ascending=False
    ).iloc[0]["gene"]
    summary = {
        "analysis": "frozen_model_interpretability_and_heterogeneity_audit",
        "settings": asdict(settings),
        "training_patients": int(len(training_target)),
        "genes": genes,
        "coefficient_sign_stable_genes_at_0_95": stable_sign_count,
        "genes_matching_frozen_direction_in_both_external_cohorts": both_external_count,
        "top_internal_permutation_importance_gene": top_gene,
        "external_cohorts_are_used_for_retuning": False,
        "claim_boundary": (
            "Interpretability describes the frozen model and transfer heterogeneity; "
            "it does not establish causal biomarkers or rescue failed replication."
        ),
    }
    with (REPORT_DIR / "interpretability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_interpretability_summary(
        gene_summary,
        effects,
        PLOT_DIR / "interpretability_and_heterogeneity_summary.png",
    )
    print(f"Training patients: {len(training_target)}")
    print(f"Stable coefficient signs (>=0.95): {stable_sign_count}/{len(genes)}")
    print(f"Direction matches in both external cohorts: {both_external_count}/{len(genes)}")
    print(f"Top internal permutation-importance gene: {top_gene}")


if __name__ == "__main__":
    main()
