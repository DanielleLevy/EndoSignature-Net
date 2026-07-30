"""Outcome-blind parsing and normalization of GSE120103 Agilent raw files."""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RawAgilentSettings:
    """Frozen technical rules for the raw rescue audit."""

    expected_total_features: int = 45_015
    expected_unique_non_control_probes: int = 41_000
    signal_floor: float = 1.0
    robust_outlier_mad: float = 3.0


def _section(lines: list[str], marker: str) -> tuple[list[str], list[str]]:
    """Return the header and first data row for an Agilent text section."""

    index = next(
        (position for position, line in enumerate(lines) if line.startswith(marker + "\t")),
        None,
    )
    if index is None:
        raise ValueError(f"Missing Agilent {marker} section")
    header = lines[index].split("\t")[1:]
    for line in lines[index + 1 :]:
        if line.startswith("DATA\t"):
            return header, line.split("\t")[1:]
        if line.startswith(("TYPE\t", "FEPARAMS\t", "STATS\t", "FEATURES\t")):
            break
    raise ValueError(f"Missing data row for Agilent {marker} section")


def _key_value(header: list[str], values: list[str]) -> dict[str, str]:
    """Safely align a section header and data row."""

    padded = values + [""] * max(0, len(header) - len(values))
    return dict(zip(header, padded))


def parse_raw_agilent_bytes(
    compressed: bytes,
    archive_name: str,
) -> tuple[pd.Series, dict[str, object]]:
    """Parse one deposited Agilent feature-extraction file."""

    lines = gzip.decompress(compressed).decode("utf-8", errors="replace").splitlines()
    fe_header, fe_values = _section(lines, "FEPARAMS")
    fe = _key_value(fe_header, fe_values)
    stats_header, stats_values = _section(lines, "STATS")
    stats = _key_value(stats_header, stats_values)

    feature_index = next(
        position for position, line in enumerate(lines) if line.startswith("FEATURES\t")
    )
    header = lines[feature_index].split("\t")[1:]
    required = {"ProbeName", "ControlType", "gProcessedSignal"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"Missing raw feature columns: {sorted(missing)}")
    probe_index = header.index("ProbeName")
    control_index = header.index("ControlType")
    signal_index = header.index("gProcessedSignal")

    total_features = 0
    probes: list[str] = []
    signals: list[float] = []
    for line in lines[feature_index + 1 :]:
        if not line.startswith("DATA\t"):
            continue
        total_features += 1
        values = line.split("\t")[1:]
        if len(values) <= max(probe_index, control_index, signal_index):
            continue
        if values[control_index].strip() != "0":
            continue
        try:
            signal = float(values[signal_index])
        except ValueError:
            continue
        probe = values[probe_index].strip()
        if probe:
            probes.append(probe)
            signals.append(signal)
    if not probes:
        raise ValueError("No non-control probe signals were parsed")

    collapsed = (
        pd.DataFrame({"probe_id": probes, "signal": signals})
        .groupby("probe_id", observed=True)["signal"]
        .median()
    )
    sample_match = re.match(r"(GSM\d+)_", archive_name)
    if sample_match is None:
        raise ValueError(f"Cannot parse GEO sample ID from {archive_name}")
    sample_id = sample_match.group(1)
    feature_extractor_version = fe.get("FeatureExtractor_Version", "")
    if not feature_extractor_version:
        version_match = re.search(r"(GE1(?:-|_v?)?[^_]*_[^_]+)", archive_name)
        feature_extractor_version = version_match.group(1) if version_match else "unknown"

    manifest: dict[str, object] = {
        "sample_id": sample_id,
        "archive_member": archive_name,
        "total_feature_rows": total_features,
        "unique_non_control_probes": int(collapsed.index.nunique()),
        "pdzd2_probe_present": "A_23_P7402" in collapsed.index,
        "feature_extractor_version": feature_extractor_version,
        "scanner_name": fe.get("Scan_ScannerName", ""),
        "scan_date": fe.get("Scan_Date", ""),
        "grid_name": fe.get("Grid_Name", ""),
        "is_good_grid": stats.get("IsGoodGrid", ""),
        "extraction_status": stats.get("ExtractionStatus", ""),
        "qc_metric_results": stats.get("QCMetricResults", ""),
        "number_saturated_features": stats.get("gNumSatFeat", ""),
        "non_control_well_above_background": stats.get(
            "gNonCtrlNumWellAboveBG", ""
        ),
    }
    return collapsed, manifest


def load_raw_archive(
    path: Path,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Read all raw files directly from the tar archive without extraction."""

    signals: dict[str, pd.Series] = {}
    manifests: list[dict[str, object]] = []
    with tarfile.open(path) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read {member.name}")
            series, manifest = parse_raw_agilent_bytes(handle.read(), member.name)
            sample_id = str(manifest["sample_id"])
            if sample_id in signals:
                raise ValueError(f"Duplicate raw sample: {sample_id}")
            signals[sample_id] = series
            manifests.append(manifest)
    return signals, pd.DataFrame(manifests).sort_values("sample_id").reset_index(drop=True)


def quantile_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalize columns using no phenotype information."""

    values = matrix.to_numpy(dtype=np.float64)
    order = np.argsort(values, axis=0, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=0)
    rank_means = sorted_values.mean(axis=1)
    normalized = np.empty_like(values)
    for column in range(values.shape[1]):
        normalized[order[:, column], column] = rank_means
    return pd.DataFrame(normalized, index=matrix.index, columns=matrix.columns)


def build_rescued_matrix(
    signals: dict[str, pd.Series],
    manifest: pd.DataFrame,
    settings: RawAgilentSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exclude structurally incomplete files and normalize valid raw arrays."""

    manifest = manifest.copy()
    manifest["complete_feature_grid"] = (
        manifest["total_feature_rows"].eq(settings.expected_total_features)
        & manifest["unique_non_control_probes"].eq(
            settings.expected_unique_non_control_probes
        )
    )
    manifest["technical_exclusion_reason"] = np.where(
        manifest["complete_feature_grid"], "", "incomplete_raw_feature_table"
    )
    eligible_ids = manifest.loc[manifest["complete_feature_grid"], "sample_id"].tolist()
    common = set(signals[eligible_ids[0]].index)
    for sample_id in eligible_ids[1:]:
        common.intersection_update(signals[sample_id].index)
    common_probes = sorted(common)
    raw = pd.DataFrame(
        {
            sample_id: signals[sample_id].reindex(common_probes)
            for sample_id in eligible_ids
        }
    )
    if raw.isna().any().any():
        raise ValueError("Common-probe raw matrix unexpectedly contains missing values")
    log2_signal = np.log2(raw.clip(lower=settings.signal_floor))
    normalized = quantile_normalize(log2_signal)
    return normalized, manifest, raw


def _robust_z(values: pd.Series) -> pd.Series:
    """Return median/MAD z-scores with a zero-MAD fallback."""

    median = values.median()
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 0.67448975 * (values - median) / mad


def rescued_matrix_qc(
    normalized: pd.DataFrame,
    manifest: pd.DataFrame,
    settings: RawAgilentSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate label-blind post-normalization correlation diagnostics."""

    correlation = normalized.corr(method="spearman")
    qc = manifest.loc[manifest["complete_feature_grid"]].copy()
    qc["median_spearman_to_other_samples"] = [
        correlation.loc[sample].drop(sample).median() for sample in qc["sample_id"]
    ]
    qc["correlation_robust_z"] = _robust_z(
        qc["median_spearman_to_other_samples"]
    ).to_numpy()
    qc["maximum_spearman_to_other_sample"] = [
        correlation.loc[sample].drop(sample).max() for sample in qc["sample_id"]
    ]
    qc["qc_flags"] = np.where(
        qc["correlation_robust_z"].lt(-settings.robust_outlier_mad),
        "low_post_normalization_correlation",
        "",
    )
    qc["qc_status"] = np.where(qc["qc_flags"].eq(""), "pass", "review")
    return qc, correlation
