# LinkedIn post draft

I’ve completed **EndoSignature-Net**, a multi-cohort transcriptomics project
exploring molecular signatures of endometriosis.

The project combines single-cell RNA sequencing, bulk RNA sequencing, and
microarray data from public GEO studies. My goal was not simply to train a model
with a high score, but to build a research workflow that respects biological
replication:

• patient-level rather than cell-level inference  
• sparse single-cell QC and doublet auditing  
• patient-level pseudobulk signature discovery  
• repeated nested cross-validation  
• locked external testing without retuning  
• interpretable gene- and patient-level predictions  

The final 12-gene L2-logistic model achieved a mean internal ROC-AUC of **0.860**
for endometriosis versus clean controls.

External validation told a more interesting story:

• GSE212787: ROC-AUC **0.738**, with 10/12 genes matching the expected direction  
• GSE153740: ROC-AUC **0.125**, with only 2/12 genes matching  

I checked the negative result carefully: labels, sample aliases, transcript
mapping, leave-one-patient analysis, and leave-one-gene analysis. The result did
not come from an obvious implementation error or a single outlier.

The interpretability analysis showed that 11/12 coefficient signs were highly
stable inside the training cohort. The problem was not unstable optimization;
it was cross-cohort biological and technical heterogeneity. Only **PDZD2** and
**ACSS2** retained their direction in both external cohorts, and published
evidence for ACSS2 is itself directionally conflicting.

My biggest takeaway: an internally stable model is not automatically a robust
biomarker. Menstrual phase, tissue composition, disease phenotype, control
definition, and measurement platform can all change the apparent signal.

I’m proud that this project reports the negative replication rather than tuning
it away. The final result is not a clinical diagnostic claim—it is a
reproducible, leakage-aware study of where a molecular signature transfers and
where it breaks.

Tech: Python, Scanpy, pandas, scikit-learn, PyTorch, scipy, statsmodels,
matplotlib.

#Bioinformatics #MachineLearning #SingleCellRNAseq #Transcriptomics
#ExplainableAI #Endometriosis #DataScience #ComputationalBiology

