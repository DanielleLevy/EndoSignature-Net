"""Cluster and provisionally annotate the GSE179640 discovery atlas.

This is a discovery-only exploratory step. The broad cell-family labels are not
final biological claims and must be validated in the independent GSE213216
single-cell atlas before downstream signature conclusions are generalized.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

import anndata as ad

from src.cell_types import (
    ClusteringSettings,
    add_cluster_annotations,
    available_marker_panels,
    cluster_composition_tables,
    family_composition,
    plot_cluster_outputs,
    provisional_cluster_annotations,
    rank_cluster_markers,
    run_leiden,
    within_family_mixing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster and provisionally annotate the discovery atlas.")
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path("output/artifacts/GSE179640_primary_atlas_full.h5ad"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--primary-resolution", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ClusteringSettings(primary_resolution=args.primary_resolution, random_state=args.random_state)
    settings.validate()

    print(f"Loading discovery atlas: {args.atlas}", flush=True)
    atlas = ad.read_h5ad(args.atlas)
    atlas.obs["study_id"] = "GSE179640"
    atlas.obs["dataset_role"] = "discovery"
    clustered = run_leiden(atlas, settings)
    panels = available_marker_panels(clustered.var_names)
    annotations = provisional_cluster_annotations(clustered, settings.primary_key, panels)
    annotated = add_cluster_annotations(clustered, settings.primary_key, annotations)

    print("Ranking exploratory cluster markers", flush=True)
    markers = rank_cluster_markers(annotated, settings.primary_key, settings)
    counts, patient_fractions, condition_summary = cluster_composition_tables(annotated, settings.primary_key)
    family_counts = family_composition(annotated)
    mixing = within_family_mixing(annotated, settings)

    reports = args.output_dir / "reports" / "cell_types" / "GSE179640_discovery"
    plots = args.output_dir / "eda_plots" / "cell_types" / "GSE179640_discovery"
    artifacts = args.output_dir / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    annotations.to_csv(reports / "provisional_cluster_annotations.csv", index=False)
    markers.to_csv(reports / "cluster_marker_rankings.csv", index=False)
    counts.to_csv(reports / "cluster_sample_composition.csv", index=False)
    patient_fractions.to_csv(reports / "cluster_patient_fractions.csv", index=False)
    condition_summary.to_csv(reports / "cluster_condition_descriptive_summary.csv", index=False)
    family_counts.to_csv(reports / "provisional_cell_family_composition.csv", index=False)
    mixing.to_csv(reports / "within_family_sample_mixing.csv", index=False)
    with (reports / "clustering_settings.json").open("w", encoding="utf-8") as handle:
        payload = asdict(settings)
        payload["resolutions"] = list(settings.resolutions)
        payload["dataset_role"] = "discovery"
        payload["external_single_cell_validation_dataset"] = "GSE213216"
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    plot_cluster_outputs(annotated, settings.primary_key, panels, family_counts, plots)
    annotated.write_h5ad(artifacts / "GSE179640_discovery_annotated_provisional.h5ad", compression="gzip")

    print(f"Primary clusters: {annotated.obs[settings.primary_key].nunique()}")
    print(f"Provisional families: {annotated.obs['provisional_cell_family'].nunique()}")
    print("Scope: GSE179640 discovery dataset only")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
