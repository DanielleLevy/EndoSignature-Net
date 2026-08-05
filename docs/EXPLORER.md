# EndoSignature Explorer

The Explorer is a read-only Streamlit application for communicating the
completed EndoSignature-Net study. It presents aggregate, frozen research
results and never retrains or retunes the model.

## Run locally

```bash
python build_explorer_data.py
streamlit run streamlit_app.py
```

The committed `explorer_data/` bundle makes the app deployable without the
large ignored `output/` directory. Rebuild that bundle only after intentionally
rerunning and reviewing the underlying analyses.

## Sections

- **Study story:** internal and locked external performance.
- **Gene explorer:** cohort directions, coefficient stability, held-out
  permutation importance, cell-family context, and literature interpretation.
- **Cycle & cells:** bounded sensitivity analyses.
- **Data reality:** cross-cohort clinical-metadata completeness.
- **Methods & limits:** design choices and claim boundaries.

The application is a research communication artifact, not a diagnostic or
clinical decision-support tool.

