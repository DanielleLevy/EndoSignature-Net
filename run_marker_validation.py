"""Validate transferred GSE213216 labels against core marker expression."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib_cache").resolve()))

import anndata as ad
import pandas as pd

from src.marker_validation import (
    MarkerValidationSettings,
    add_marker_validation,
    cell_validation_frame,
    plot_validation_summaries,
    validation_summaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external marker-concordance checks.")
    parser.add_argument(
        "--mapped-dir",
        type=Path,
        default=Path("output/artifacts/GSE213216_mapped_samples"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--mapping-score-threshold", type=float, default=0.60)
    parser.add_argument("--minimum-detected-marker-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-detected-markers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = MarkerValidationSettings(
        mapping_score_threshold=args.mapping_score_threshold,
        minimum_detected_marker_fraction=args.minimum_detected_marker_fraction,
        minimum_detected_markers=args.minimum_detected_markers,
    )
    settings.validate()
    files = sorted(args.mapped_dir.glob("*_mapped_singlets.h5ad"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No mapped H5AD files found in {args.mapped_dir}")

    reports = args.output_dir / "reports" / "validation" / "GSE213216" / "marker_validation"
    artifacts = args.output_dir / "artifacts" / "GSE213216_marker_validated_samples"
    plots = args.output_dir / "eda_plots" / "marker_validation" / "GSE213216"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    print(f"Validating markers in {len(files)} mapped samples")
    for index, path in enumerate(files, start=1):
        sample_id = path.name.split("_mapped_singlets", 1)[0]
        print(f"[{index}/{len(files)}] {sample_id}", flush=True)
        try:
            mapped = ad.read_h5ad(path)
            validated = add_marker_validation(mapped, settings)
            frames.append(cell_validation_frame(validated))
            validated.write_h5ad(
                artifacts / f"{sample_id}_marker_validated.h5ad",
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

    if not frames:
        raise SystemExit("All marker-validation samples failed")
    cells = pd.concat(frames, ignore_index=True)
    family, tissue, sample = validation_summaries(cells)
    cells.to_csv(reports / "cell_marker_validation.csv.gz", index=False)
    family.to_csv(reports / "family_marker_support.csv", index=False)
    tissue.to_csv(reports / "tissue_family_marker_support.csv", index=False)
    sample.to_csv(reports / "sample_marker_support.csv", index=False)
    pd.DataFrame(errors, columns=["sample_id", "error_type", "message"]).to_csv(
        reports / "marker_validation_errors.csv", index=False
    )
    with (reports / "marker_validation_settings.json").open("w", encoding="utf-8") as handle:
        payload = asdict(settings)
        payload["interpretation"] = (
            "external_marker_concordance_not_independent_ground_truth"
        )
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_validation_summaries(family, tissue, plots)

    supported = (cells["marker_validation_status"] == "marker_supported").sum()
    print(f"Validated cells: {len(cells)}")
    print(f"Marker-supported labels: {supported} ({100 * supported / len(cells):.2f}%)")
    print(f"Failed samples: {len(errors)}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
