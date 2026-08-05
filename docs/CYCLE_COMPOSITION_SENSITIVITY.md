# Cycle-phase and cell-composition sensitivity audit

## Question

This audit tests whether menstrual-cycle phase or broad cell-family composition
provides an evident explanation for the frozen signature's cross-cohort
heterogeneity. It does not change the 12 genes, tune a model on external
outcomes, or claim causal adjustment.

## Menstrual-cycle sensitivity

Repeated out-of-fold predictions from the completed GSE51981 internal analysis
were averaged per patient and evaluated overall and separately within
early-secretory, mid-secretory, and proliferative samples. This preserves the
original patient-level folds. The frozen signature was also compared with its
previously specified signature-plus-cycle model; no model was refitted for this
audit.

The frozen signature had an overall ROC-AUC of 0.877. Adding cycle phase yielded
0.875, a difference of -0.002. The frozen model remained directionally useful
within each recorded phase: ROC-AUC was 0.796 in early-secretory (24 patients),
0.839 in mid-secretory (36), and 0.910 in proliferative samples (49). These
subgroup estimates are descriptive, especially where the control count is low.

The existing locked GSE212787 analysis provides an external within-phase check.
Model pairwise concordance was 0.65 across 20 proliferative case-control pairs
and 1.00 across only three secretory pairs. The secretory estimate is too small
to support a strong phase-specific claim. GSE153740 contains only mid-secretory
samples and therefore cannot estimate within-study cycle sensitivity.

## Cell-composition sensitivity

The GSE179640 discovery atlas contains three control and nine endometriosis
patients. Provisional cell-family fractions were aggregated per patient. Each
family was tested using all 220 possible assignments of nine case labels among
12 patients after an arcsine-square-root transformation. False-discovery rates
were controlled with Benjamini-Hochberg correction.

A global exact PERMANOVA on centered-log-ratio compositions produced R-squared
0.094 and p=0.414. No individual family had FDR below 0.05. The largest raw
differences were lower epithelial fraction in endometriosis (-0.276; unadjusted
p=0.077) and higher stromal/fibroblast fraction (+0.156; unadjusted p=0.450).
These are exploratory effect-size observations, not confirmed differences.

## Interpretation

The internal signal does not appear to be explained simply by recorded cycle
phase: adding cycle did not improve overall discrimination, and the frozen model
retained discrimination within each GSE51981 phase. However, this does not rule
out cycle-related cross-cohort effects because phase, cohort, tissue, and
platform remain confounded externally.

Likewise, this audit found no statistically supported global cell-composition
shift in the small discovery atlas. The low control count, provisional cell
annotations, and absence of comparable cell-resolved external controls mean that
composition cannot be excluded as a source of heterogeneity. The analysis
narrows the explanation but does not identify a causal driver of the failed
GSE153740 replication.

## Reproduction

```bash
python run_cycle_composition_sensitivity.py
```

Versioned tables are written to `output/reports/sensitivity/v1.0/`, and the
figure is written to `output/eda_plots/sensitivity/v1.0/`.

