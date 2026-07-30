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

## The strongest engineering/research decisions

- Split and infer at patient level, not cell level.
- Keep count matrices sparse during QC.
- Separate discovery, evidence integration, modeling, and external testing.
- Refuse to apply a locked model when frozen genes or QC gates fail.
- Do not flip a failed external score after seeing outcomes.
- Prefer an interpretable linear model when a more complex model does not
  provide a meaningful, reliable advantage.
- Report claim boundaries alongside every major result.

## Honest conclusion

The project identified a compact, interpretable research signature that
transfers to some but not all cohorts. It did not produce a clinically validated
diagnostic test. The negative replication and interpretability audit show that
endometrial transcriptomic biomarkers are strongly affected by cohort, tissue,
cycle, and platform context.

## CV-ready bullets

- Developed an end-to-end Python pipeline integrating scRNA-seq, bulk RNA-seq,
  and microarray cohorts for endometriosis biomarker research.
- Implemented patient-level pseudobulk analysis, repeated nested
  cross-validation, external-cohort QC gates, bootstrap uncertainty, and model
  interpretability.
- Benchmarked linear, nonlinear, ensemble, and neural architectures; selected an
  interpretable 12-gene L2-logistic model with internal ROC-AUC 0.860.
- Conducted locked cross-platform replication in independent cohorts and
  localized transfer failure to cross-cohort gene-direction heterogeneity.
- Built a tested, documented research codebase with 69 automated tests.

