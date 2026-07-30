"""Run verified metadata, probe mapping and processed EDA for GSE51981."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.microarray_eda import (
    MicroarrayEDASettings,
    collapse_probes_to_genes,
    expression_pca,
    load_gpl570_annotation,
    load_series_matrix,
    pca_factor_associations,
    plot_microarray_eda,
    processed_expression_qc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verified GSE51981 EDA.")
    parser.add_argument(
        "--series-matrix",
        type=Path,
        default=Path(
            "output/source_metadata/GSE51981/GSE51981_series_matrix.txt.gz"
        ),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("output/source_metadata/GSE51981/GPL570.annot.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = MicroarrayEDASettings()
    probe_expression, metadata, series = load_series_matrix(args.series_matrix)
    annotation = load_gpl570_annotation(args.annotation)
    gene_expression, mapping = collapse_probes_to_genes(
        probe_expression, annotation
    )
    qc, correlation = processed_expression_qc(
        gene_expression, metadata, settings
    )
    pca, variance = expression_pca(gene_expression, metadata, settings)
    pca_associations = pca_factor_associations(pca)

    reports = args.output_dir / "reports" / "microarray" / "GSE51981"
    plots = args.output_dir / "eda_plots" / "microarray" / "GSE51981"
    reports.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(reports / "verified_sample_metadata.csv", index=False)
    qc.to_csv(reports / "processed_sample_qc.csv", index=False)
    qc.loc[qc["qc_status"].eq("review")].to_csv(
        reports / "samples_requiring_review.csv", index=False
    )
    correlation.to_csv(reports / "sample_spearman_correlation.csv.gz")
    pca.to_csv(reports / "pca_coordinates.csv", index=False)
    variance.to_csv(reports / "pca_explained_variance.csv", index=False)
    pca_associations.to_csv(reports / "pca_factor_associations.csv", index=False)
    mapping.to_csv(reports / "probe_to_gene_mapping.csv.gz", index=False)
    gene_expression.to_csv(reports / "gcrma_gene_expression.csv.gz")

    mismatch_count = int((~metadata["severity_metadata_concordant"]).sum())
    settings_payload = {
        **asdict(settings),
        "study_id": "GSE51981",
        "platform": "GPL570",
        "input_processing": (
            "official_series_matrix_GCRMA_simultaneous_normalization_fullmodel"
        ),
        "probe_collapse_rule": "median_across_unambiguously_mapped_probes",
        "n_samples": len(metadata),
        "n_input_probes": len(probe_expression),
        "n_annotation_probes_unambiguous": len(annotation),
        "n_mapped_genes": len(gene_expression),
        "severity_metadata_disagreement_samples": mismatch_count,
        "series_title": series.get("title", ""),
    }
    with (reports / "microarray_eda_settings.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(settings_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    cross_tab = (
        metadata.groupby(["clinical_group", "cycle_phase"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    clean_controls = int(
        metadata["clinical_group"].eq("Healthy_control_no_pathology").sum()
    )
    disease = int(metadata["clinical_group"].eq("Endometriosis").sum())
    with (reports / "analysis_suitability.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "primary_candidate_validation": (
                    "supported_with_cycle_phase_adjustment_and_sensitivity_checks"
                ),
                "primary_contrast": (
                    "77 endometriosis versus 34 non-endometriosis samples "
                    "without uterine/pelvic pathology"
                ),
                "secondary_specificity_group": (
                    "37 non-endometriosis samples with other uterine/pelvic pathology"
                ),
                "processed_data_qc_scope": (
                    "distribution_correlation_and_PCA_only; raw CEL diagnostics "
                    "cannot be reproduced from the single local CEL file"
                ),
                "cycle_phase_counts": {
                    str(group): {str(phase): int(value) for phase, value in row.items()}
                    for group, row in cross_tab.to_dict(orient="index").items()
                },
                "n_primary_endometriosis": disease,
                "n_primary_clean_controls": clean_controls,
                "recommended_validation_model": (
                    "probe-collapsed gene expression ~ endometriosis + cycle phase"
                ),
                "required_sensitivity_checks": [
                    "Retain the three distribution-flagged but highly correlated samples in the primary model and exclude them in sensitivity analysis.",
                    "Run leave-one-sample-out direction checks for the small high-priority candidate set.",
                    "Compare endometriosis against the other-pathology group as a secondary specificity analysis.",
                    "Track the two samples with inconsistent official severity fields without changing their disease label.",
                ],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    plot_microarray_eda(gene_expression, qc, pca, variance, plots)

    print(
        f"Official matrix: {probe_expression.shape[0]} probes x "
        f"{probe_expression.shape[1]} samples"
    )
    print(f"Mapped gene matrix: {gene_expression.shape}")
    print(f"Clinical groups: {metadata['clinical_group'].value_counts().to_dict()}")
    print(f"Samples flagged for processed-data review: {qc['qc_status'].eq('review').sum()}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
