"""Data loading and validation for the EndoSignature Explorer."""

import json
from pathlib import Path

import pandas as pd


REQUIRED_TABLES = {
    "cohort_performance.csv": {"cohort", "analysis", "roc_auc", "n_patients", "status"},
    "gene_evidence.csv": {"gene", "cohort", "standardized_effect"},
    "gene_summary.csv": {"gene", "sign_stability", "permutation_auc_decrease_mean"},
    "cycle_metrics.csv": {"model", "cycle_phase", "roc_auc", "n_patients"},
    "cell_composition.csv": {"cell_family", "fraction_difference", "fdr_bh"},
    "metadata_completeness.csv": {"study_id", "field", "completeness_fraction"},
}


def load_explorer_data(data_dir):
    """Load the curated, deployment-safe research result bundle."""
    data_dir = Path(data_dir)
    tables = {}
    for filename, columns in REQUIRED_TABLES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing explorer data file: {path}")
        frame = pd.read_csv(path)
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        tables[path.stem] = frame
    summary_path = data_dir / "project_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing explorer data file: {summary_path}")
    tables["project_summary"] = json.loads(summary_path.read_text())
    return tables


def gene_view(gene, effects, summary):
    """Return cohort effects and the single summary row for a selected gene."""
    effect_rows = effects.loc[effects["gene"].eq(gene)].copy()
    summary_rows = summary.loc[summary["gene"].eq(gene)].copy()
    if effect_rows.empty or len(summary_rows) != 1:
        raise ValueError(f"Incomplete explorer evidence for gene {gene}")
    return effect_rows, summary_rows.iloc[0]

