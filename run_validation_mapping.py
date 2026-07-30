"""Map GSE213216 singlets to provisional GSE179640 broad cell families."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

import anndata as ad
import numpy as np
import pandas as pd

from src.atlas import load_primary_sample
from src.eda import QCThresholds
from src.validation_mapping import (
    MappingSettings,
    add_transferred_labels,
    build_classifier,
    family_mapping_summary,
    mapping_sample_summary,
    patient_grouped_reference_cv,
    prepare_query_for_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent cell-family label transfer.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("output/artifacts/GSE179640_discovery_annotated_provisional.h5ad"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("output/reports/validation/GSE213216/sample_metadata.csv"),
    )
    parser.add_argument(
        "--mapping-candidates",
        type=Path,
        default=Path(
            "output/reports/validation/GSE213216/doublets/mapping_candidates.csv"
        ),
    )
    parser.add_argument(
        "--qc-thresholds",
        type=Path,
        default=Path("output/reports/qc/GSE213216/qc_thresholds.json"),
    )
    parser.add_argument(
        "--doublet-dir",
        type=Path,
        default=Path(
            "output/reports/validation/GSE213216/doublets/cell_predictions"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = MappingSettings(confidence_threshold=args.confidence_threshold)
    settings.validate()
    metadata = pd.read_csv(args.metadata, keep_default_na=False)
    candidates = pd.read_csv(args.mapping_candidates, keep_default_na=False)
    selected_ids = set(candidates["sample_id"])
    selected = metadata.loc[metadata["sample_id"].isin(selected_ids)].sort_values(
        "sample_id"
    )
    if args.limit is not None:
        selected = selected.head(args.limit)

    with args.qc_thresholds.open(encoding="utf-8") as handle:
        qc_thresholds = QCThresholds(**json.load(handle))

    print(f"Loading discovery reference: {args.reference}", flush=True)
    reference = ad.read_h5ad(args.reference)
    coordinates = np.asarray(reference.obsm["X_pca"], dtype=np.float64)
    labels = reference.obs["provisional_cell_family"].astype(str).to_numpy()
    patients = reference.obs["patient_id"].astype(str).to_numpy()
    cv_folds, cv_predictions = patient_grouped_reference_cv(
        coordinates, labels, patients, settings
    )
    classifier = build_classifier(settings)
    classifier.fit(coordinates, labels)

    reference_genes = reference.var_names.tolist()
    hvg_mask = reference.var["highly_variable"].to_numpy(dtype=bool)
    loadings = np.asarray(reference.varm["PCs"])

    reports = args.output_dir / "reports" / "validation" / "GSE213216" / "mapping"
    artifacts = args.output_dir / "artifacts" / "GSE213216_mapped_samples"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    cv_folds.to_csv(reports / "reference_patient_grouped_cv.csv", index=False)
    cv_predictions.to_csv(
        reports / "reference_patient_grouped_cv_predictions.csv.gz", index=False
    )

    sample_rows: list[dict[str, object]] = []
    cell_frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    print(f"Mapping {len(selected)} validation samples", flush=True)
    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        sample_id = row["sample_id"]
        print(f"[{index}/{len(selected)}] {sample_id}", flush=True)
        try:
            singlets, _ = load_primary_sample(
                row,
                qc_thresholds,
                args.doublet_dir / f"{sample_id}_doublets.csv.gz",
            )
            mapped = prepare_query_for_reference(
                singlets,
                reference_genes,
                hvg_mask,
                loadings,
                settings,
            )
            mapped = add_transferred_labels(mapped, classifier, settings)
            sample_rows.append(mapping_sample_summary(mapped))
            cell_frames.append(
                pd.DataFrame(
                    {
                        "cell_id": mapped.obs_names,
                        "sample_id": mapped.obs["sample_id"].astype(str).to_numpy(),
                        "patient_id": mapped.obs["patient_id"].astype(str).to_numpy(),
                        "condition": mapped.obs["condition"].astype(str).to_numpy(),
                        "tissue_code": mapped.obs["tissue_code"].astype(str).to_numpy(),
                        "transferred_cell_family": mapped.obs[
                            "transferred_cell_family"
                        ].astype(str).to_numpy(),
                        "mapping_confidence": mapped.obs[
                            "mapping_confidence"
                        ].to_numpy(),
                        "is_high_confidence": (
                            mapped.obs["mapping_confidence_status"].astype(str)
                            == "high_confidence"
                        ).to_numpy(),
                    }
                )
            )
            mapped.write_h5ad(
                artifacts / f"{sample_id}_mapped_singlets.h5ad",
                compression="gzip",
            )
        except Exception as exc:
            errors.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    sample_summary = pd.DataFrame(sample_rows)
    if not cell_frames:
        pd.DataFrame(errors, columns=["sample_id", "error_type", "message"]).to_csv(
            reports / "mapping_processing_errors.csv", index=False
        )
        raise SystemExit("All validation samples failed mapping; inspect the error report")
    cell_predictions = pd.concat(cell_frames, ignore_index=True)
    sample_summary.to_csv(reports / "mapping_sample_summary.csv", index=False)
    cell_predictions.to_csv(reports / "transferred_cell_labels.csv.gz", index=False)
    family_mapping_summary(cell_predictions).to_csv(
        reports / "mapping_family_summary.csv", index=False
    )
    pd.DataFrame(errors, columns=["sample_id", "error_type", "message"]).to_csv(
        reports / "mapping_processing_errors.csv", index=False
    )
    with (reports / "mapping_settings.json").open("w", encoding="utf-8") as handle:
        payload = asdict(settings)
        payload.update(
            {
                "reference_study": "GSE179640",
                "query_study": "GSE213216",
                "reference_label_status": "provisional",
                "classifier_class_weight": "balanced",
                "probability_calibration": "not_calibrated",
                "split_unit": "patient",
                "n_reference_hvgs": int(hvg_mask.sum()),
            }
        )
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Mapped samples: {len(sample_summary)}; failed: {len(errors)}")
    print(f"Mapped singlets: {len(cell_predictions)}")
    print(f"Mean CV macro-F1: {cv_folds['macro_f1'].mean():.3f}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
