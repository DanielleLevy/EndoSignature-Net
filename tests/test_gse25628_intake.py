"""Tests for GSE25628 target logic."""

from pathlib import Path

from src.gse25628_intake import load_gse25628_series_matrix


def test_gse25628_official_target_counts() -> None:
    path = Path(
        "output/source_metadata/GSE25628/GSE25628_series_matrix.txt.gz"
    )
    if not path.exists():
        return
    _, metadata, _ = load_gse25628_series_matrix(path)
    target = metadata.loc[metadata["include_in_external_target"]]
    assert len(target) == 14
    assert target["condition"].eq("Endometriosis").sum() == 8
    assert target["condition"].eq("Disease_free_control").sum() == 6
    assert metadata["tissue_class"].eq("endometriosis_ectopic").sum() == 8
