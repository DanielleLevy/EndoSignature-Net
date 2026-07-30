"""Run the label-blind GSE120103 raw Agilent rescue audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.agilent_raw_rescue import (
    RawAgilentSettings,
    build_rescued_matrix,
    load_raw_archive,
    rescued_matrix_qc,
)
from src.external_cohort_intake import (
    FROZEN_GENES,
    load_gpl6480_annotation,
    load_gse120103_series_matrix,
    map_frozen_signature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run outcome-blind raw Agilent rescue for GSE120103."
    )
    parser.add_argument(
        "--raw-archive",
        type=Path,
        default=Path("output/source_metadata/GSE120103/GSE120103_RAW.tar"),
    )
    parser.add_argument(
        "--series-matrix",
        type=Path,
        default=Path(
            "output/source_metadata/GSE120103/GSE120103_series_matrix.txt.gz"
        ),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("output/source_metadata/GSE120103/GPL6480.annot.gz"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "output/reports/external_validation/v1.0/GSE120103_raw_rescue"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    settings = RawAgilentSettings()
    signals, manifest = load_raw_archive(args.raw_archive)
    normalized, manifest, raw = build_rescued_matrix(signals, manifest, settings)
    qc, correlation = rescued_matrix_qc(normalized, manifest, settings)
    annotation = load_gpl6480_annotation(args.annotation)
    signature, mapping = map_frozen_signature(normalized, annotation)
    _, metadata, _ = load_gse120103_series_matrix(args.series_matrix)

    manifest = manifest.merge(
        metadata[["sample_id", "condition", "fertility", "cycle_phase"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    qc = qc.merge(
        metadata[["sample_id", "condition", "fertility", "cycle_phase"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.report_dir / "raw_sample_manifest.csv", index=False)
    qc.to_csv(args.report_dir / "post_normalization_sample_qc.csv", index=False)
    correlation.to_csv(args.report_dir / "post_normalization_correlation.csv.gz")
    mapping.to_csv(args.report_dir / "raw_frozen_signature_probe_mapping.csv", index=False)
    signature.to_csv(args.report_dir / "raw_rescued_signature_expression.csv.gz")

    excluded = manifest.loc[~manifest["complete_feature_grid"], "sample_id"].tolist()
    remaining_counts = (
        manifest.loc[manifest["complete_feature_grid"]]
        .groupby(["condition", "fertility"], observed=True)
        .size()
    )
    missing_genes = [gene for gene in FROZEN_GENES if gene not in signature.index]
    review_samples = qc.loc[qc["qc_status"].eq("review"), "sample_id"].tolist()
    decision = (
        "conditional_pass_to_locked_outcome_analysis"
        if not missing_genes and not review_samples and len(excluded) < 4
        else "hold_for_review"
    )
    payload = {
        "study_id": "GSE120103",
        "analysis_stage": "outcome_blind_raw_rescue",
        "outcome_performance_inspected": False,
        "raw_archive_sha256": sha256(args.raw_archive),
        "raw_files": len(manifest),
        "complete_raw_files": int(manifest["complete_feature_grid"].sum()),
        "technically_excluded_samples": excluded,
        "exclusion_rule": (
            "exclude before outcome analysis when the raw table does not contain "
            "45015 total feature rows and 41000 unique non-control probes"
        ),
        "common_probe_count_after_exclusion": len(normalized),
        "all_12_frozen_genes_recovered": not missing_genes,
        "missing_frozen_genes": missing_genes,
        "post_normalization_review_samples": review_samples,
        "remaining_disease_by_fertility_counts": {
            f"{condition}|{fertility}": int(count)
            for (condition, fertility), count in remaining_counts.items()
        },
        "normalization": (
            "median collapse of replicated spots; log2 floor at one; "
            "quantile normalization across technically complete arrays"
        ),
        "normalization_uses_outcomes": False,
        "eligibility_decision": decision,
        "settings": asdict(settings),
        "limitations": [
            "two technically failed samples are removed from one study cell",
            "feature extraction versions differ across deposited files",
            "cohort-level quantile normalization is not a single-sample clinical pipeline",
        ],
    }
    with (args.report_dir / "raw_rescue_decision.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Raw files parsed: {len(manifest)}")
    print(f"Technically complete files: {manifest['complete_feature_grid'].sum()}")
    print(f"Excluded before outcomes: {excluded}")
    print(f"Common normalized probes: {len(normalized)}")
    print(f"Frozen genes recovered: {len(signature)}/{len(FROZEN_GENES)}")
    print(f"Post-normalization review samples: {review_samples}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
