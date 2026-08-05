# EndoSignature-Net — Portfolio Summary

## One-line description

Built a leakage-aware multi-cohort transcriptomics pipeline that discovers,
validates, models, and interprets an endometriosis-associated gene signature
across single-cell RNA-seq, bulk RNA-seq, and microarray data.

## What I built

- Reproducible GEO metadata and sample manifests.
- Sparse per-sample scRNA-seq QC and doublet auditing.
- A patient-aware eutopic-endometrium discovery cohort.
- Cell-family annotation and patient-level pseudobulk differential expression.
- Multi-study signature evidence integration.
- Repeated nested patient-level architecture benchmarking.
- Locked external-cohort eligibility, application, and bootstrap uncertainty.
- Global and per-patient model interpretability.
- Harmonized clinical-metadata feasibility across eight GEO studies.
- Cycle-phase and patient-level cell-composition sensitivity analyses.
- Automated tests and English methodological documentation.

## Main results

- Frozen 12-gene L2-logistic model: internal mean ROC-AUC **0.860**.
- Clinically harder endometriosis-versus-other-pathology stress test:
  ROC-AUC **0.560**.
- GSE212787 locked external application: ROC-AUC **0.738**
  (95% CI 0.429-0.976), 10/12 direction matches.
- GSE153740 locked external application: ROC-AUC **0.125**
  (95% CI 0.000-0.500), 2/12 direction matches.
- Eleven of 12 model coefficient signs stable in at least 95% of patient
  bootstraps.
- `PDZD2` and `ACSS2` were the only genes with directional agreement in both
  external cohorts.
- The frozen signature retained pooled patient-level ROC-AUC **0.796-0.910**
  within the three recorded GSE51981 cycle phases; adding cycle changed overall
  pooled ROC-AUC from **0.877 to 0.875**.
- The exploratory GSE179640 composition audit found no supported global shift
  (CLR PERMANOVA R-squared **0.094**, exact p=**0.414**) and no cell family at
  FDR below 0.05.

## The strongest engineering/research decisions

- Split and infer at patient level, not cell level.
- Keep count matrices sparse during QC.
- Separate discovery, evidence integration, modeling, and external testing.
- Refuse to apply a locked model when frozen genes or QC gates fail.
- Do not flip a failed external score after seeing outcomes.
- Prefer an interpretable linear model when a more complex model does not
  provide a meaningful, reliable advantage.
- Report claim boundaries alongside every major result.
- Preserve missing clinical metadata rather than substituting study location or
  mixed tissue labels for patient-level covariates.

## Honest conclusion

The project identified a compact, interpretable research signature that
transfers to some but not all cohorts. It did not produce a clinically validated
diagnostic test. The negative replication and interpretability audit show that
endometrial transcriptomic biomarkers are strongly affected by cohort, tissue,
cycle, and platform context. Follow-up sensitivity analysis did not identify
recorded cycle phase or broad cell-family composition as a sufficient single
explanation for the replication gap.

## Project status

Complete as a portfolio-scale computational research project. The appropriate
next scientific step is a new preregistered validation study with a larger
independent cohort, standardized clinical metadata, symptomatic disease
controls, and a deployable single-sample normalization protocol—not additional
retuning on the external cohorts already observed.
