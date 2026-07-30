"""Build the uncorrected primary GSE179640 atlas and its EDA artifacts.

Run after metadata, QC and cohort-definition pipelines. This stage intentionally
does not apply Harmony, scVI or another batch-correction method; its purpose is
to reveal whether correction is needed and what structure it must preserve.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

import pandas as pd

from src.atlas import (
    AtlasSettings,
    batch_diagnostics,
    concatenate_samples,
    hvg_atlas,
    load_primary_sample,
    matrix_is_sparse,
    plot_atlas_composition,
    plot_embeddings,
    preprocess_atlas,
    sample_composition,
)
from src.eda import QCThresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the uncorrected primary single-cell atlas.")
    parser.add_argument("--verified-metadata", type=Path, default=Path("output/reports/cohort/verified_sample_metadata.csv"))
    parser.add_argument("--primary-samples", type=Path, default=Path("output/reports/cohort/primary_cohort_samples.csv"))
    parser.add_argument("--doublet-dir", type=Path, default=Path("output/reports/cohort/cell_doublets"))
    parser.add_argument("--qc-thresholds", type=Path, default=Path("output/reports/qc/GSE179640/qc_thresholds.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--n-top-genes", type=int, default=2_000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Process the first N samples for a smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verified = pd.read_csv(args.verified_metadata, keep_default_na=False)
    approved = set(pd.read_csv(args.primary_samples)["sample_id"])
    selected = verified.loc[verified["sample_id"].isin(approved)].sort_values("sample_id")
    if args.limit is not None:
        selected = selected.head(args.limit)
    if selected.empty:
        raise SystemExit("No approved primary-cohort samples were found")

    with args.qc_thresholds.open(encoding="utf-8") as handle:
        qc_thresholds = QCThresholds(**json.load(handle))
    settings = AtlasSettings(
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )
    settings.validate()

    samples = []
    filtering_rows = []
    print(f"Loading {len(selected)} primary samples")
    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        sample_id = row["sample_id"]
        print(f"[{index}/{len(selected)}] {sample_id}", flush=True)
        sample, filtering = load_primary_sample(
            row,
            qc_thresholds,
            args.doublet_dir / f"{sample_id}_doublets.csv.gz",
        )
        samples.append(sample)
        filtering_rows.append(filtering)

    print("Concatenating samples and computing the uncorrected embedding", flush=True)
    combined = concatenate_samples(samples)
    atlas = preprocess_atlas(combined, settings)
    compact = hvg_atlas(atlas)

    reports = args.output_dir / "reports" / "atlas"
    plots = args.output_dir / "eda_plots" / "atlas" / "GSE179640_primary"
    artifacts = args.output_dir / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(filtering_rows).to_csv(reports / "atlas_filtering_summary.csv", index=False)
    composition = sample_composition(atlas)
    composition.to_csv(reports / "atlas_sample_composition.csv", index=False)
    batch_diagnostics(atlas, settings).to_csv(reports / "uncorrected_batch_diagnostics.csv", index=False)
    with (reports / "atlas_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(settings), handle, indent=2, sort_keys=True)
        handle.write("\n")

    plot_embeddings(atlas, plots)
    plot_atlas_composition(composition, plots / "atlas_sample_composition.png")
    atlas.write_h5ad(artifacts / "GSE179640_primary_atlas_full.h5ad", compression="gzip")
    compact.write_h5ad(artifacts / "GSE179640_primary_atlas_hvg.h5ad", compression="gzip")

    print(f"Atlas shape: {atlas.n_obs} cells x {atlas.n_vars} shared genes")
    print(f"Highly variable genes: {int(atlas.var['highly_variable'].sum())}")
    print(f"Sparse normalized matrix: {matrix_is_sparse(atlas)}")
    print(f"Reports: {reports}")
    print(f"Artifacts: {artifacts}")


if __name__ == "__main__":
    main()
