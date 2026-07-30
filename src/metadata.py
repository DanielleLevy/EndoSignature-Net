"""Build a reproducible sample manifest for the EndoSignature-Net datasets.

The filename parser is deliberately conservative.  It records how every value
was obtained and leaves fields empty when the local files do not contain enough
information.  Filename-derived metadata must be checked against GEO before it
is used as clinical ground truth.
"""

from __future__ import annotations

import csv
import gzip
import re
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


GSM_RE = re.compile(r"^(GSM\d+)")
SAMPLE_RE = re.compile(r"_(C\d+|E\d+|EOR\d+)_")
TISSUE_RE = re.compile(r"_(Ctrl|EuE|EcPA2|EcPA|EcP|EcO)_")

# These relationships are supported by the local archive names and GEO records.
STUDY_RULES = (
    (re.compile(r"^(?:GSM610|GSM659)"), "GSE179640"),
    (re.compile(r"^GSM657"), "GSE213216"),
    (re.compile(r"^GSM125"), "GSE51981"),
    (re.compile(r"^GSE135485"), "GSE135485"),
)

TISSUE_NAMES = {
    "Ctrl": "control",
    "EuE": "eutopic_endometrium",
    "EcP": "ectopic_peritoneum",
    "EcO": "ectopic_ovary",
    # Preserve the source codes: their precise biological meaning should be
    # verified against the study metadata before subgroup analysis.
    "EcPA": "ectopic_peritoneum_adjacent",
    "EcPA2": "ectopic_peritoneum_adjacent_2",
    "EOR": "endometriosis_organoid",
}


@dataclass(frozen=True)
class FileRecord:
    """One physical file present in ``data/``."""

    file_name: str
    file_path: str
    size_bytes: int
    file_format: str
    technology: str
    study_id: str
    sample_id: str


@dataclass(frozen=True)
class SampleRecord:
    """One logical biological sample or dataset-level data object."""

    sample_id: str
    study_id: str
    patient_id: str
    condition: str
    tissue: str
    tissue_code: str
    technology: str
    file_format: str
    file_path: str
    source_files: str
    include_in_sc_eda: bool
    metadata_source: str
    review_status: str
    notes: str


def _study_id(name: str) -> str:
    for pattern, study in STUDY_RULES:
        if pattern.search(name):
            return study
    if name.startswith("GSE"):
        return name.split("_", 1)[0]
    return ""


def _sample_id(name: str) -> str:
    match = GSM_RE.match(name)
    if match:
        return match.group(1)
    if name.startswith("GSE"):
        return name.split("_", 1)[0]
    return Path(name).name


def _format(name: str) -> str:
    if name.endswith("_filtered_feature_bc_matrix.h5"):
        return "10x_h5"
    if name.endswith("matrix.mtx.gz"):
        return "10x_mtx"
    if name.endswith(("barcodes.tsv.gz", "features.tsv.gz")):
        return "10x_mtx_companion"
    if name.endswith("featurecounts.txt.gz"):
        return "featurecounts"
    if name.endswith(".csv.gz"):
        return "csv_gzip"
    if name.endswith(".CEL.gz"):
        return "CEL_gzip"
    if name.endswith(".tar"):
        return "tar_archive"
    return Path(name).suffix.lstrip(".") or "unknown"


def _technology(name: str) -> str:
    fmt = _format(name)
    if fmt.startswith("10x_"):
        return "scRNA-seq"
    if fmt in {"featurecounts", "csv_gzip"}:
        return "bulk_RNA-seq"
    if fmt == "CEL_gzip":
        return "microarray"
    if name == "GSE179640_RAW.tar":
        return "mixed_archive"
    if name == "GSE213216_RAW.tar":
        return "scRNA-seq_archive"
    return "unknown"


def scan_files(data_dir: Path) -> list[FileRecord]:
    """Return a deterministic inventory of regular files in ``data_dir``."""

    records = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        records.append(
            FileRecord(
                file_name=path.name,
                file_path=path.as_posix(),
                size_bytes=path.stat().st_size,
                file_format=_format(path.name),
                technology=_technology(path.name),
                study_id=_study_id(path.name),
                sample_id=_sample_id(path.name),
            )
        )
    return records


def _patient_and_condition(name: str) -> tuple[str, str, str]:
    match = SAMPLE_RE.search(name)
    if not match:
        return "", "", "sample identity is not encoded in the filename"
    code = match.group(1)
    if code.startswith("C"):
        return code, "Control", ""
    if code.startswith("EOR"):
        return code, "Endometriosis", "organoid derived from an endometriosis sample"
    return code, "Endometriosis", ""


def _tissue(name: str, patient_id: str) -> tuple[str, str]:
    match = TISSUE_RE.search(name)
    if match:
        code = match.group(1)
        return TISSUE_NAMES[code], code
    if patient_id.startswith("EOR"):
        return TISSUE_NAMES["EOR"], "EOR"
    return "", ""


def _is_cell_hashing_matrix(members: Sequence[FileRecord]) -> bool:
    """Detect hashtag-only matrices from their small HTO feature vocabulary."""

    feature_files = [Path(item.file_path) for item in members if item.file_name.endswith("features.tsv.gz")]
    if len(feature_files) != 1:
        return False
    with gzip.open(feature_files[0], "rt", encoding="utf-8") as handle:
        features = [line.strip().split("\t")[0] for line in handle if line.strip()]
    return bool(features) and all(feature.startswith("HHTO") or feature == "unmapped" for feature in features)


def build_samples(files: Sequence[FileRecord]) -> list[SampleRecord]:
    """Collapse physical files into logical samples and annotate them.

    The three files of a 10x Matrix Market sample are grouped by GSM accession.
    Dataset-level archives and matrices remain explicit rows because they are
    useful provenance records but do not represent a single patient.
    """

    groups: dict[tuple[str, str], list[FileRecord]] = {}
    for record in files:
        key = (record.sample_id, record.technology)
        groups.setdefault(key, []).append(record)

    samples = []
    for (sample_id, technology), members in sorted(groups.items()):
        names = [member.file_name for member in members]
        representative = names[0]
        patient_id, condition, note = _patient_and_condition(representative)
        tissue, tissue_code = _tissue(representative, patient_id)
        formats = sorted({member.file_format for member in members})
        is_hashing = _is_cell_hashing_matrix(members)
        logical_technology = "cell_hashing" if is_hashing else technology
        is_sc_sample = logical_technology == "scRNA-seq"
        complete_mtx = "10x_mtx" not in formats or len(members) == 3
        required = (patient_id, condition, tissue, members[0].study_id)
        review_status = "parsed_from_filename" if all(required) and complete_mtx else "needs_review"
        if not complete_mtx:
            note = "; ".join(filter(None, (note, "incomplete 10x Matrix Market file set")))
        if is_hashing:
            note = "; ".join(filter(None, (note, "HTO-only cell-hashing matrix; no gene-expression features")))
        if is_sc_sample:
            note = "; ".join(filter(None, (note, "condition/tissue inferred from filename; verify against GEO before analysis")))

        samples.append(
            SampleRecord(
                sample_id=sample_id,
                study_id=members[0].study_id,
                patient_id=patient_id,
                condition=condition,
                tissue=tissue,
                tissue_code=tissue_code,
                technology=logical_technology,
                file_format="+".join(formats),
                file_path=members[0].file_path if len(members) == 1 else "",
                source_files=";".join(member.file_path for member in members),
                include_in_sc_eda=is_sc_sample and review_status == "parsed_from_filename",
                metadata_source="local_filename_and_archive_membership",
                review_status=review_status,
                notes=note,
            )
        )
    return samples


def archive_members(files: Iterable[FileRecord]) -> list[dict[str, str]]:
    """List archive members without extracting large raw-data archives."""

    rows = []
    for record in files:
        if record.file_format != "tar_archive":
            continue
        with tarfile.open(record.file_path, "r") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    rows.append(
                        {
                            "archive": record.file_name,
                            "study_id": record.study_id,
                            "member_name": member.name,
                            "member_size_bytes": str(member.size),
                            "sample_id": _sample_id(Path(member.name).name),
                        }
                    )
    return rows


def write_csv(path: Path, records: Sequence[object]) -> None:
    """Write dataclass or dictionary records as UTF-8 CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in records]
    if not rows:
        raise ValueError(f"Cannot write an empty report: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(samples: Sequence[SampleRecord]) -> list[dict[str, object]]:
    """Count samples by study, technology, condition and tissue."""

    counts = Counter(
        (item.study_id or "unknown", item.technology, item.condition or "unknown", item.tissue or "unknown")
        for item in samples
    )
    return [
        {
            "study_id": key[0],
            "technology": key[1],
            "condition": key[2],
            "tissue": key[3],
            "n_samples": count,
        }
        for key, count in sorted(counts.items())
    ]
