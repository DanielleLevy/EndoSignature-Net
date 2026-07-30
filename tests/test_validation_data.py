"""Tests for authoritative validation-cohort preparation."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from src.validation_data import (
    extract_filtered_h5,
    extract_filtered_mtx,
    list_nested_single_cell_members,
    parse_gse213216_miniml,
)


MINIML = """<?xml version="1.0"?>
<MINiML xmlns="http://www.ncbi.nlm.nih.gov/geo/info/MINiML">
  <Sample iid="GSM1">
    <Title>Sample 1</Title>
    <Channel>
      <Source>Endometriosis, scRNAseq</Source>
      <Characteristics tag="patient id">7</Characteristics>
      <Characteristics tag="tissue">Endometriosis</Characteristics>
      <Characteristics tag="anatomic location">Pelvic sidewall</Characteristics>
    </Channel>
    <Supplementary-Data type="TAR">ftp://example/GSM1.tar.gz</Supplementary-Data>
  </Sample>
  <Sample iid="GSM2">
    <Title>Spatial sample</Title>
    <Channel>
      <Source>No endometriosis detected</Source>
      <Characteristics tag="tissue">No endometriosis detected</Characteristics>
    </Channel>
    <Extract-Protocol>10X Visium spatial gene expression</Extract-Protocol>
  </Sample>
</MINiML>
"""


def test_parse_miniml_preserves_conservative_control_semantics(tmp_path: Path) -> None:
    xml_path = tmp_path / "family.xml"
    xml_path.write_text(MINIML, encoding="utf-8")

    metadata = parse_gse213216_miniml(xml_path).set_index("sample_id")

    assert metadata.loc["GSM1", "technology"] == "scRNA-seq"
    assert metadata.loc["GSM1", "condition"] == "endometriosis_lesion"
    assert metadata.loc["GSM2", "technology"] == "spatial_transcriptomics"
    assert metadata.loc["GSM2", "condition"] == "no_endometriosis_detected_site"
    assert metadata.loc["GSM2", "control_status"] == "not_established"


def test_selective_nested_h5_extraction(tmp_path: Path) -> None:
    nested_bytes = io.BytesIO()
    with tarfile.open(fileobj=nested_bytes, mode="w:gz") as nested:
        payload = b"filtered-matrix"
        info = tarfile.TarInfo("sample/outs/filtered_feature_bc_matrix.h5")
        info.size = len(payload)
        nested.addfile(info, io.BytesIO(payload))
        raw = b"must-not-be-extracted"
        raw_info = tarfile.TarInfo("sample/outs/raw_feature_bc_matrix.h5")
        raw_info.size = len(raw)
        nested.addfile(raw_info, io.BytesIO(raw))

    archive_path = tmp_path / "outer.tar"
    with tarfile.open(archive_path, mode="w") as outer:
        compressed = nested_bytes.getvalue()
        info = tarfile.TarInfo("GSM1_sample1.tar.gz")
        info.size = len(compressed)
        outer.addfile(info, io.BytesIO(compressed))

    assert list_nested_single_cell_members(archive_path) == {
        "GSM1": "GSM1_sample1.tar.gz"
    }
    destination = tmp_path / "GSM1_filtered.h5"
    extract_filtered_h5(archive_path, "GSM1_sample1.tar.gz", destination)
    assert destination.read_bytes() == b"filtered-matrix"
    assert not list(tmp_path.glob("*.partial"))


def test_selective_nested_matrix_market_extraction(tmp_path: Path) -> None:
    nested_bytes = io.BytesIO()
    with tarfile.open(fileobj=nested_bytes, mode="w:gz") as nested:
        for name in ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"):
            payload = name.encode()
            info = tarfile.TarInfo(f"sample/outs/filtered_feature_bc_matrix/{name}")
            info.size = len(payload)
            nested.addfile(info, io.BytesIO(payload))

    archive_path = tmp_path / "outer.tar"
    with tarfile.open(archive_path, mode="w") as outer:
        compressed = nested_bytes.getvalue()
        info = tarfile.TarInfo("GSM2_sample2.tar.gz")
        info.size = len(compressed)
        outer.addfile(info, io.BytesIO(compressed))

    extracted = extract_filtered_mtx(
        archive_path, "GSM2_sample2.tar.gz", tmp_path / "GSM2"
    )
    assert {path.name for path in extracted} == {
        "barcodes.tsv.gz",
        "features.tsv.gz",
        "matrix.mtx.gz",
    }
