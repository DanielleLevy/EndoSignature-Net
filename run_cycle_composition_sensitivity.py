"""Run the frozen-signature cycle and cell-composition sensitivity audit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.sensitivity_audit import (
    aggregate_repeated_predictions,
    composition_permanova,
    exact_family_composition_tests,
    patient_composition_matrix,
    stratified_prediction_metrics,
)


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "output/reports/sensitivity/v1.0"
PLOT_DIR = ROOT / "output/eda_plots/sensitivity/v1.0"


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(
        ROOT / "output/reports/modeling/GSE51981_internal_baseline/v1.0/repeated_oof_predictions.csv.gz"
    )
    cohort = pd.read_csv(
        ROOT / "output/reports/modeling/GSE51981_internal_baseline/v1.0/modeling_cohort.csv"
    )
    averaged = aggregate_repeated_predictions(predictions)
    cycle_metrics = stratified_prediction_metrics(averaged, cohort)
    cycle_metrics.to_csv(REPORT / "gse51981_cycle_stratified_oof_metrics.csv", index=False)

    external_cycle = pd.read_csv(
        ROOT / "output/reports/external_validation/v1.0/GSE212787_directional_replication/within_cycle_concordance.csv"
    )
    external_cycle.to_csv(REPORT / "gse212787_within_cycle_concordance.csv", index=False)

    composition = pd.read_csv(
        ROOT / "output/reports/cell_types/GSE179640_discovery/provisional_cell_family_composition.csv"
    )
    matrix = patient_composition_matrix(composition)
    family_tests = exact_family_composition_tests(matrix)
    family_tests.to_csv(REPORT / "gse179640_cell_family_composition_tests.csv", index=False)
    permanova = composition_permanova(matrix)

    frozen = cycle_metrics.loc[cycle_metrics.model.eq("frozen_signature_12")].copy()
    frozen_plus = cycle_metrics.loc[cycle_metrics.model.eq("frozen_signature_12_plus_cycle")].copy()
    overall_auc = float(frozen.loc[frozen.cycle_phase.eq("All"), "roc_auc"].iloc[0])
    adjusted_auc = float(frozen_plus.loc[frozen_plus.cycle_phase.eq("All"), "roc_auc"].iloc[0])
    summary = {
        "analysis_version": "v1.0",
        "frozen_signature_overall_oof_auc": overall_auc,
        "frozen_signature_plus_cycle_overall_oof_auc": adjusted_auc,
        "auc_difference_after_adding_cycle": adjusted_auc - overall_auc,
        "gse179640_composition_permanova": permanova,
        "n_cell_families_fdr_below_0_05": int(family_tests.fdr_bh.lt(0.05).sum()),
        "interpretation": (
            "Cycle-stratified performance and external within-cycle concordance are descriptive. "
            "The GSE179640 composition test is exploratory because it contains three controls and "
            "nine endometriosis patients and uses provisional cell-family annotations."
        ),
        "claim_boundary": (
            "The audit can identify sensitivity to cycle or composition but cannot establish either "
            "factor as the causal explanation for cross-cohort replication heterogeneity."
        ),
    }
    (REPORT / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_metrics = cycle_metrics.loc[
        cycle_metrics.model.isin(["frozen_signature_12", "frozen_signature_12_plus_cycle"])
    ].copy()
    plot_metrics["cycle_phase"] = plot_metrics["cycle_phase"].str.replace("_", " ")
    plot_metrics["model"] = plot_metrics["model"].map(
        {"frozen_signature_12": "Frozen 12-gene signature", "frozen_signature_12_plus_cycle": "Signature + cycle"}
    )
    sns.barplot(data=plot_metrics, x="cycle_phase", y="roc_auc", hue="model", ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("GSE51981 repeated OOF performance by cycle phase")
    axes[0].set_xlabel("Cycle phase")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(title="Model", fontsize=8)

    ordered = family_tests.sort_values("fraction_difference")
    colors = ["#c44e52" if value > 0 else "#4c72b0" for value in ordered.fraction_difference]
    axes[1].barh(ordered.cell_family.str.replace("_", " "), ordered.fraction_difference, color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("GSE179640 patient-level cell-family composition")
    axes[1].set_xlabel("Mean fraction difference (endometriosis - control)")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cycle_composition_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Frozen overall OOF AUC: {overall_auc:.3f}")
    print(f"Frozen + cycle overall OOF AUC: {adjusted_auc:.3f}")
    print(f"Composition PERMANOVA R2: {permanova['clr_permanova_r_squared']:.3f}")
    print(f"Composition exact p-value: {permanova['exact_permutation_p']:.3f}")
    print(f"Cell families at FDR < 0.05: {summary['n_cell_families_fdr_below_0_05']}")


if __name__ == "__main__":
    main()
