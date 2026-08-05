# Cross-cohort clinical metadata audit

## Purpose

This audit asks whether the heterogeneous external-validation results can be
investigated using verified patient-level clinical covariates. It is a
feasibility and provenance analysis, not a new classifier and not a post-hoc
attempt to rescue the frozen 12-gene signature.

## Methods

`run_cross_cohort_metadata_audit.py` harmonizes the verified metadata already
collected for eight GEO studies: GSE179640, GSE213216, GSE51981, GSE135485,
GSE212787, GSE153740, GSE120103, and GSE25628. The adapters preserve the source
study, sample identifier, patient identifier when deposited, measurement
technology, analysis status, and original source file.

The harmonized clinical fields are condition, age, menstrual-cycle phase,
disease stage, disease subtype, tissue, hormonal treatment, fertility status,
and country. Missing values remain missing: study location is not treated as
participant country, and tissue or lesion location is not promoted to a
clinical disease subtype. When a deposited patient identifier is unavailable,
the sample is explicitly marked as a patient proxy rather than assumed to be
an independent patient.

Completeness is calculated after aggregation to patient or patient-proxy level.
Conflicting values within a patient are retained as joined categories and
flagged through the provenance-preserving table rather than silently selected.

## Dataset inventory

The audit contains 355 sample records and 324 patient records or patient
proxies across eight studies. The difference is mainly due to studies with
multiple specimens per deposited patient.

| Study | Patient records / proxies | Sample records | Technology | Relevant metadata |
|---|---:|---:|---|---|
| GSE51981 | 148 | 148 | Microarray | Cycle phase for 99%; stage for 52% |
| GSE135485 | 58 | 58 | Bulk RNA-seq | Tissue and condition; no verified cycle, age, or stage |
| GSE120103 | 36 | 36 | Microarray | Secretory phase and fertility status; stage for 50% |
| GSE213216 | 23 | 51 | scRNA-seq | Endometriosis atlas without a healthy-control cohort |
| GSE25628 | 22 | 22 | Microarray | Proliferative phase; incomplete frozen-gene panel |
| GSE212787 | 17 | 20 | Bulk RNA-seq | Proliferative or broadly secretory phase |
| GSE179640 | 12 | 12 | scRNA-seq | Hormonal-treatment field for 75%; no cycle phase |
| GSE153740 | 8 | 8 | Transcript-level bulk RNA-seq | All mid-secretory; lesion location recorded |

## Findings and permitted interpretations

- **Cycle phase:** a descriptive sensitivity analysis is feasible because five
  studies have at least 80% completeness. A causal or fully adjusted estimate
  is not feasible: phase is strongly confounded with study, platform, tissue,
  and phenotype, while the most informative external cohort has only eight
  patients.
- **Age:** no verified participant-level age was available in the assembled
  public metadata. Age-adjusted modeling is therefore not supported.
- **Disease stage:** stage is available only for selected patients in GSE51981
  and GSE120103. It can support cautious within-cohort description, not a formal
  cross-cohort stage-stratified model.
- **Disease subtype:** deposited fields mix lesion location, tissue class, and
  disease state. They are preserved but are not sufficiently harmonized for a
  single formal subtype test.
- **Country:** participant country was not verified for any cohort. Study or
  laboratory location is not an acceptable substitute.
- **Platform:** platform sensitivity can be described, but platform and cohort
  are nearly inseparable in the current design.
- **Cell composition:** the metadata audit cannot quantify cellular mixture.
  That requires a separate deconvolution or cell-type composition analysis.

The audit therefore supports one bounded next experiment: a cycle-phase and
cell-composition sensitivity analysis. It does not support new claims about age,
geography, or a universal severity-adjusted biomarker.

## Reproduction

```bash
python run_cross_cohort_metadata_audit.py
```

Versioned tables and the machine-readable feasibility decision are written to
`output/reports/cross_cohort_metadata/v1.0/`. The figure is written to
`output/eda_plots/cross_cohort_metadata/v1.0/` and curated for the repository as
`docs/assets/08_cross_cohort_metadata_audit.png`.

