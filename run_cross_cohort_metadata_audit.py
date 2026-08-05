"""Build an auditable patient-level metadata table across project cohorts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

from src.cross_cohort_metadata import (
    CLINICAL_FIELDS,
    aggregate_patient_metadata,
    clean_series,
    cohort_summary,
    metadata_completeness,
    normalize_cycle_phase,
    pairwise_comparability,
    plot_metadata_audit,
)


REPORT_DIR = Path("output/reports/cross_cohort_metadata/v1.0")
PLOT_DIR = Path("output/eda_plots/cross_cohort_metadata/v1.0")

SOURCE_PATHS = {
    "GSE179640": Path("output/reports/cohort/verified_sample_metadata.csv"),
    "GSE213216": Path("output/reports/validation/GSE213216/sample_metadata.csv"),
    "GSE135485": Path("output/reports/bulk/GSE135485/verified_sample_metadata.csv"),
    "GSE51981": Path("output/reports/microarray/GSE51981/verified_sample_metadata.csv"),
    "GSE212787": Path(
        "output/reports/external_validation/v1.0/GSE212787_intake/verified_sample_metadata.csv"
    ),
    "GSE153740": Path(
        "output/reports/external_validation/v1.0/GSE153740_intake/verified_sample_metadata.csv"
    ),
    "GSE120103": Path(
        "output/reports/external_validation/v1.0/GSE120103_intake/verified_sample_metadata.csv"
    ),
    "GSE25628": Path(
        "output/reports/external_validation/v1.0/GSE25628_intake/verified_sample_metadata.csv"
    ),
}


def _base_frame(source: pd.DataFrame, study_id: str) -> pd.DataFrame:
    result = pd.DataFrame(index=source.index)
    result["study_id"] = study_id
    result["sample_id"] = source["sample_id"].astype(str)
    result["age"] = pd.NA
    result["country"] = pd.NA
    result["age_provenance"] = "not_available_in_verified_project_metadata"
    result["country_provenance"] = "not_available_in_verified_project_metadata"
    result["source_file"] = str(SOURCE_PATHS[study_id])
    result["metadata_source"] = source.get(
        "metadata_source", pd.Series("verified_project_metadata", index=source.index)
    )
    return result


def _gse179640(source: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(source, "GSE179640")
    result["patient_id"] = source["patient_id"].astype(str)
    result["patient_id_quality"] = "explicit_study_patient_code"
    result["condition"] = source["geo_condition"].replace({"Control": "Disease_free_control"})
    result["cycle_phase"] = pd.NA
    result["disease_stage"] = pd.NA
    result["disease_subtype"] = pd.NA
    result["tissue"] = source["geo_sample_location"]
    result["hormonal_treatment"] = source["hormonal_treatment"]
    result["fertility_status"] = pd.NA
    result["platform"] = "10x_Genomics"
    result["technology"] = "scRNA-seq"
    result["analysis_role"] = "single_cell_discovery"
    result["include_in_analysis_target"] = source["include_in_sc_eda"]
    result["clinical_provenance"] = "verified_NCBI_GEO_sample_records"
    return result


def _gse213216(source: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(source, "GSE213216")
    patient = source["patient_id"].map(
        lambda value: str(int(value)) if pd.notna(value) else pd.NA
    )
    result["patient_id"] = patient.where(patient.notna(), source["sample_id"].astype(str))
    result["patient_id_quality"] = patient.map(
        lambda value: "explicit_study_patient_code" if pd.notna(value) else "sample_id_proxy"
    )
    result["condition"] = "Endometriosis_cohort_no_healthy_control"
    result["cycle_phase"] = pd.NA
    result["disease_stage"] = pd.NA
    result["disease_subtype"] = source["condition"]
    result["tissue"] = source["tissue"]
    result["hormonal_treatment"] = pd.NA
    result["fertility_status"] = pd.NA
    result["platform"] = "10x_Genomics"
    result["technology"] = "scRNA-seq"
    result["analysis_role"] = "single_cell_localization_validation"
    result["include_in_analysis_target"] = source["include_in_sc_eda"]
    result["clinical_provenance"] = "NCBI_GEO_MINiML"
    return result


def _gse135485(source: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(source, "GSE135485")
    result["patient_id"] = source["sample_id"].astype(str)
    result["patient_id_quality"] = "sample_id_proxy_no_explicit_patient_identifier"
    result["condition"] = source["condition"].replace({"Control": "Disease_free_control"})
    result["cycle_phase"] = pd.NA
    result["disease_stage"] = pd.NA
    result["disease_subtype"] = pd.NA
    result["tissue"] = source["tissue"]
    result["hormonal_treatment"] = pd.NA
    result["fertility_status"] = pd.NA
    result["platform"] = source["lane"].map(lambda _: "Illumina_RNA-seq")
    result["technology"] = "bulk_RNA-seq"
    result["analysis_role"] = "secondary_tissue_direction_validation"
    result["include_in_analysis_target"] = True
    result["clinical_provenance"] = "NCBI_GEO_SOFT"
    return result


def _gse51981(source: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(source, "GSE51981")
    result["patient_id"] = source["patient_id"].astype(str)
    result["patient_id_quality"] = "explicit_study_patient_code"
    result["condition"] = source["clinical_group"]
    result["cycle_phase"] = source["cycle_phase"]
    result["disease_stage"] = source["endometriosis_severity"]
    result["disease_subtype"] = pd.NA
    result["tissue"] = source["tissue"]
    result["hormonal_treatment"] = pd.NA
    result["fertility_status"] = pd.NA
    result["platform"] = source["platform_id"]
    result["technology"] = "microarray"
    result["analysis_role"] = "evidence_integration_and_modeling"
    result["include_in_analysis_target"] = True
    result["clinical_provenance"] = "NCBI_GEO_series_matrix"
    return result


def _gse212787(source: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(source, "GSE212787")
    result["patient_id"] = source["patient_id"].astype(str)
    result["patient_id_quality"] = "explicit_title_patient_code"
    result["condition"] = source["condition"]
    result["cycle_phase"] = source["cycle_phase"]
    result["disease_stage"] = pd.NA
    result["disease_subtype"] = source["tissue_class"]
    result["tissue"] = source["tissue"]
    result["hormonal_treatment"] = pd.NA
    result["fertility_status"] = pd.NA
    result["platform"] = source["platform_id"]
    result["technology"] = "bulk_RNA-seq"
    result["analysis_role"] = "locked_external_replication"
    result["include_in_analysis_target"] = source["include_in_external_target"]
    result["clinical_provenance"] = "GEO_titles_and_expression_aliases"
    return result


def _gse153740(source: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(source, "GSE153740")
    result["patient_id"] = source["patient_id"].astype(str)
    result["patient_id_quality"] = "sample_id_proxy_distinct_biological_replicates"
    result["condition"] = source["condition"]
    result["cycle_phase"] = source["cycle_phase"]
    result["disease_stage"] = pd.NA
    result["disease_subtype"] = source["disease_state"]
    result["tissue"] = source["tissue_class"]
    result["hormonal_treatment"] = pd.NA
    result["fertility_status"] = pd.NA
    result["platform"] = source["platform_id"]
    result["technology"] = "bulk_RNA-seq_transcript_FPKM"
    result["analysis_role"] = "locked_external_replication"
    result["include_in_analysis_target"] = True
    result["clinical_provenance"] = "NCBI_GEO_series_matrix"
    return result


def _gse120103(source: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(source, "GSE120103")
    result["patient_id"] = source["patient_id"].astype(str)
    result["patient_id_quality"] = "sample_id_proxy_distinct_arrays"
    result["condition"] = source["condition"]
    result["cycle_phase"] = source["cycle_phase"]
    result["disease_stage"] = source["condition"].where(
        source["condition"].str.contains("Stage", na=False), pd.NA
    )
    result["disease_subtype"] = source["condition"].map(
        lambda value: "Stage_IV_ovarian_endometriosis"
        if value == "Stage_IV_ovarian_endometriosis"
        else pd.NA
    )
    result["tissue"] = source["tissue_target"]
    result["hormonal_treatment"] = pd.NA
    result["fertility_status"] = source["fertility"]
    result["platform"] = source["platform_id"]
    result["technology"] = "microarray"
    result["analysis_role"] = "technical_hold_not_modeled"
    result["include_in_analysis_target"] = False
    result["clinical_provenance"] = "GEO_series_matrix_and_primary_publication"
    return result


def _gse25628(source: pd.DataFrame) -> pd.DataFrame:
    result = _base_frame(source, "GSE25628")
    result["patient_id"] = source["sample_id"].astype(str)
    result["patient_id_quality"] = "sample_id_proxy_patient_identifier_unavailable"
    result["condition"] = source["condition"]
    result["cycle_phase"] = source["cycle_phase"]
    result["disease_stage"] = pd.NA
    result["disease_subtype"] = source["tissue_class"]
    result["tissue"] = source["tissue"]
    result["hormonal_treatment"] = pd.NA
    result["fertility_status"] = pd.NA
    result["platform"] = source["platform_id"]
    result["technology"] = "microarray"
    result["analysis_role"] = "incomplete_panel_hold_not_modeled"
    result["include_in_analysis_target"] = source["include_in_external_target"]
    result["clinical_provenance"] = "NCBI_GEO_series_matrix"
    return result


ADAPTERS = {
    "GSE179640": _gse179640,
    "GSE213216": _gse213216,
    "GSE135485": _gse135485,
    "GSE51981": _gse51981,
    "GSE212787": _gse212787,
    "GSE153740": _gse153740,
    "GSE120103": _gse120103,
    "GSE25628": _gse25628,
}


def main() -> None:
    missing_files = [str(path) for path in SOURCE_PATHS.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Required verified metadata files are missing: {missing_files}")

    frames = []
    for study_id, path in SOURCE_PATHS.items():
        frames.append(ADAPTERS[study_id](pd.read_csv(path)))
    samples = pd.concat(frames, ignore_index=True)
    for column in CLINICAL_FIELDS:
        samples[column] = clean_series(samples[column])
    samples["cycle_phase"] = samples["cycle_phase"].map(normalize_cycle_phase)
    samples["patient_uid"] = samples["study_id"] + ":" + samples["patient_id"].astype(str)
    patients = aggregate_patient_metadata(samples.drop(columns="patient_uid"))
    completeness = metadata_completeness(patients)
    summary = cohort_summary(patients)
    comparability = pairwise_comparability(patients)

    cycle_studies = completeness.loc[
        completeness["field"].eq("cycle_phase")
        & completeness["completeness_fraction"].ge(0.8),
        "study_id",
    ].tolist()
    age_studies = completeness.loc[
        completeness["field"].eq("age")
        & completeness["completeness_fraction"].ge(0.8),
        "study_id",
    ].tolist()
    stage_studies = completeness.loc[
        completeness["field"].eq("disease_stage")
        & completeness["completeness_fraction"].gt(0),
        "study_id",
    ].tolist()
    decision = {
        "analysis": "cross_cohort_patient_metadata_audit",
        "n_studies": int(samples["study_id"].nunique()),
        "n_sample_records": int(len(samples)),
        "n_patient_records_or_proxies": int(len(patients)),
        "cycle_phase": {
            "status": "descriptive_analysis_feasible_causal_attribution_not_feasible",
            "studies_with_at_least_80_percent_completeness": cycle_studies,
            "reason": (
                "Cycle phase is recorded in several cohorts, but phase, study, platform, "
                "and phenotype are confounded and external cohorts are small."
            ),
        },
        "age": {
            "status": "not_feasible",
            "studies_with_at_least_80_percent_completeness": age_studies,
            "reason": "Age is unavailable in the verified project metadata for all cohorts.",
        },
        "disease_stage": {
            "status": "within_cohort_description_only",
            "studies_with_any_recorded_stage": stage_studies,
            "reason": "Stage is recorded only in selected cohorts and is not cross-study complete.",
        },
        "disease_subtype": {
            "status": "not_harmonized_for_formal_cross_cohort_test",
            "reason": (
                "Available subtype fields mix lesion location, tissue class, and disease state; "
                "they are not equivalent phenotypes."
            ),
        },
        "country": {
            "status": "not_feasible",
            "reason": (
                "Patient country is unavailable; study-center country must not be substituted "
                "for participant origin."
            ),
        },
        "platform": {
            "status": "descriptive_sensitivity_feasible_not_identifiable_as_cause",
            "reason": "Platform is observed but largely confounded with cohort.",
        },
        "cell_composition": {
            "status": "requires_separate_deconvolution_analysis",
            "reason": "Cell proportions are not clinical metadata and must be estimated from expression.",
        },
        "next_analysis": (
            "A bounded cycle-phase and cell-composition sensitivity audit is justified; "
            "age, geography, and formal stage-stratified cross-cohort modeling are not."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    samples.to_csv(REPORT_DIR / "cross_cohort_sample_metadata.csv", index=False)
    patients.to_csv(REPORT_DIR / "cross_cohort_patient_metadata.csv", index=False)
    completeness.to_csv(REPORT_DIR / "metadata_completeness.csv", index=False)
    summary.to_csv(REPORT_DIR / "cohort_summary.csv", index=False)
    comparability.to_csv(REPORT_DIR / "cohort_comparability.csv", index=False)
    with (REPORT_DIR / "analysis_feasibility.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_metadata_audit(
        patients,
        completeness,
        PLOT_DIR / "cross_cohort_metadata_audit.png",
    )

    print(f"Studies: {decision['n_studies']}")
    print(f"Sample records: {decision['n_sample_records']}")
    print(f"Patient records/proxies: {decision['n_patient_records_or_proxies']}")
    print(f"Cycle-phase conclusion: {decision['cycle_phase']['status']}")
    print(f"Age conclusion: {decision['age']['status']}")
    print(f"Next analysis: {decision['next_analysis']}")


if __name__ == "__main__":
    main()
