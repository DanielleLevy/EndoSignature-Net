"""Preparation utilities for the independent GSE213216 validation cohort.

The public GEO archive stores each single-cell sample as a nested ``tar.gz``
member inside a large outer TAR file.  This module parses authoritative MINiML
metadata and extracts only Cell Ranger's filtered gene-expression H5 matrix.
Raw matrices, molecule information and spatial images are intentionally skipped
to keep disk usage bounded.
"""

from __future__ import annotations

import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


MINIML_NAMESPACE = {"geo": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}


def _text(element: ET.Element | None) -> str:
    """Return normalized element text or an empty string."""

    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def parse_gse213216_miniml(xml_path: Path) -> pd.DataFrame:
    """Parse sample-level GSE213216 metadata from an official MINiML file."""

    root = ET.parse(xml_path).getroot()
    records: list[dict[str, object]] = []
    for sample in root.findall("geo:Sample", MINIML_NAMESPACE):
        sample_id = sample.attrib["iid"]
        characteristics = {
            item.attrib.get("tag", "").strip().lower(): _text(item)
            for item in sample.findall(".//geo:Characteristics", MINIML_NAMESPACE)
        }
        title = _text(sample.find("geo:Title", MINIML_NAMESPACE))
        source = _text(sample.find(".//geo:Source", MINIML_NAMESPACE))
        tissue = characteristics.get("tissue", source.split(",")[0]).strip()
        protocol = " ".join(
            _text(item) for item in sample.findall(".//geo:Extract-Protocol", MINIML_NAMESPACE)
        ).lower()
        is_single_cell = "scrnaseq" in source.lower()
        is_spatial = "visium" in protocol and not is_single_cell
        modality = "scRNA-seq" if is_single_cell else ("spatial_transcriptomics" if is_spatial else "other")

        supplementary = [
            _text(item)
            for item in sample.findall("geo:Supplementary-Data", MINIML_NAMESPACE)
            if _text(item)
        ]
        records.append(
            {
                "sample_id": sample_id,
                "study_id": "GSE213216",
                "title": title,
                "patient_id": characteristics.get("patient id", ""),
                "condition": _condition_from_tissue(tissue),
                "tissue": tissue,
                "tissue_code": _tissue_code(tissue),
                "anatomic_location": characteristics.get("anatomic location", ""),
                "source_name": source,
                "technology": modality,
                "control_status": "not_established",
                "supplementary_urls": ";".join(supplementary),
                "metadata_source": "NCBI GEO MINiML",
                "metadata_url": (
                    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + sample_id
                ),
            }
        )
    return pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True)


def _condition_from_tissue(tissue: str) -> str:
    """Map GEO tissue labels without inventing a healthy-control label."""

    mapping = {
        "Endometriosis": "endometriosis_lesion",
        "Endometrioma": "endometrioma",
        "Eutopic Endometrium": "eutopic_endometrium",
        "Ovary": "unaffected_ovary",
        "No endometriosis detected": "no_endometriosis_detected_site",
    }
    return mapping.get(tissue, "unresolved")


def _tissue_code(tissue: str) -> str:
    """Return concise, analysis-safe tissue codes."""

    mapping = {
        "Endometriosis": "EndoLesion",
        "Endometrioma": "Endometrioma",
        "Eutopic Endometrium": "EuE",
        "Ovary": "Ovary",
        "No endometriosis detected": "NED_site",
    }
    return mapping.get(tissue, "Unknown")


def list_nested_single_cell_members(archive_path: Path) -> dict[str, str]:
    """Map GSM identifiers to nested sample archives in the outer GEO TAR."""

    members: dict[str, str] = {}
    with tarfile.open(archive_path, mode="r") as outer:
        for member in outer.getmembers():
            name = Path(member.name).name
            if member.isfile() and name.startswith("GSM") and name.endswith(".tar.gz"):
                members[name.split("_", 1)[0]] = member.name
    return members


def extract_filtered_h5(
    archive_path: Path,
    outer_member_name: str,
    destination: Path,
) -> Path:
    """Safely stream one filtered H5 matrix from a nested sample archive."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists():
        return destination

    try:
        with tarfile.open(archive_path, mode="r") as outer:
            outer_member = outer.getmember(outer_member_name)
            nested_stream = outer.extractfile(outer_member)
            if nested_stream is None:
                raise FileNotFoundError(f"Cannot read archive member: {outer_member_name}")
            with tarfile.open(fileobj=nested_stream, mode="r|gz") as nested:
                for member in nested:
                    if member.isfile() and member.name.endswith("/outs/filtered_feature_bc_matrix.h5"):
                        source = nested.extractfile(member)
                        if source is None:
                            break
                        with partial.open("wb") as target:
                            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                        partial.replace(destination)
                        return destination
        raise FileNotFoundError(
            f"No filtered_feature_bc_matrix.h5 in {outer_member_name}"
        )
    finally:
        partial.unlink(missing_ok=True)


def extract_filtered_mtx(
    archive_path: Path,
    outer_member_name: str,
    destination_dir: Path,
) -> list[Path]:
    """Safely extract only the three filtered 10x Matrix Market files."""

    expected = ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destinations = [destination_dir / name for name in expected]
    if all(path.exists() for path in destinations):
        return destinations

    written: list[Path] = []
    try:
        with tarfile.open(archive_path, mode="r") as outer:
            nested_stream = outer.extractfile(outer.getmember(outer_member_name))
            if nested_stream is None:
                raise FileNotFoundError(f"Cannot read archive member: {outer_member_name}")
            with tarfile.open(fileobj=nested_stream, mode="r|gz") as nested:
                for member in nested:
                    marker = "/outs/filtered_feature_bc_matrix/"
                    if not member.isfile() or marker not in member.name:
                        continue
                    basename = Path(member.name).name
                    if basename not in expected:
                        continue
                    source = nested.extractfile(member)
                    if source is None:
                        continue
                    destination = destination_dir / basename
                    partial = destination.with_suffix(destination.suffix + ".partial")
                    with partial.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    partial.replace(destination)
                    written.append(destination)
        if not all(path.exists() for path in destinations):
            raise FileNotFoundError(
                f"Incomplete filtered Matrix Market files in {outer_member_name}"
            )
        return destinations
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        for path in destination_dir.glob("*.partial"):
            path.unlink(missing_ok=True)
        raise


def prepare_gse213216_manifest(
    metadata: pd.DataFrame,
    archive_path: Path,
    matrix_dir: Path,
    extract: bool = True,
) -> pd.DataFrame:
    """Attach archive availability and local matrix paths to GEO metadata."""

    nested_members = list_nested_single_cell_members(archive_path)
    prepared = metadata.copy()
    prepared["archive_member"] = prepared["sample_id"].map(nested_members).fillna("")
    prepared["file_format"] = ""
    prepared["file_path"] = ""
    prepared["source_files"] = ""
    prepared["include_in_sc_eda"] = False
    prepared["availability_status"] = "not_available_as_filtered_h5"

    eligible = prepared.loc[
        (prepared["technology"] == "scRNA-seq") & prepared["archive_member"].ne("")
    ]
    total = len(eligible)
    current = 0
    for index, row in prepared.iterrows():
        if row["technology"] != "scRNA-seq" or not row["archive_member"]:
            continue
        current += 1
        destination = matrix_dir / f"{row['sample_id']}_filtered_feature_bc_matrix.h5"
        print(f"[{current}/{total}] Preparing {row['sample_id']}", flush=True)
        matrix_files: list[Path] = []
        if extract and not destination.exists():
            try:
                extract_filtered_h5(archive_path, row["archive_member"], destination)
            except FileNotFoundError:
                sample_dir = matrix_dir / row["sample_id"]
                matrix_files = extract_filtered_mtx(
                    archive_path, row["archive_member"], sample_dir
                )
        if destination.exists():
            prepared.at[index, "file_format"] = "10x_h5"
            prepared.at[index, "file_path"] = str(destination)
            prepared.at[index, "source_files"] = str(destination)
            prepared.at[index, "include_in_sc_eda"] = True
            prepared.at[index, "availability_status"] = "ready"
        else:
            sample_dir = matrix_dir / row["sample_id"]
            matrix_files = matrix_files or [
                sample_dir / "barcodes.tsv.gz",
                sample_dir / "features.tsv.gz",
                sample_dir / "matrix.mtx.gz",
            ]
            if all(path.exists() for path in matrix_files):
                prepared.at[index, "file_format"] = "10x_mtx_v3"
                prepared.at[index, "file_path"] = str(sample_dir)
                prepared.at[index, "source_files"] = ";".join(
                    str(path) for path in matrix_files
                )
                prepared.at[index, "include_in_sc_eda"] = True
                prepared.at[index, "availability_status"] = "ready"
            else:
                prepared.at[index, "availability_status"] = "not_extracted"
    return prepared
