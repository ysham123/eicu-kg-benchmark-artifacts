from pathlib import Path

import pytest

from eicu_kg.audit import assert_patient_disjoint, restricted_release_paths


def test_patient_disjointness() -> None:
    assert_patient_disjoint([1, 2], [3], [4, 5])
    with pytest.raises(ValueError, match="train/test"):
        assert_patient_disjoint([1, 2], [3], [2, 4])


def test_restricted_path_scan(tmp_path: Path) -> None:
    (tmp_path / "safe.py").write_text("pass\n", encoding="utf-8")
    assert restricted_release_paths(tmp_path) == []
    bad = tmp_path / "mapping"
    bad.mkdir()
    (bad / "drug.csv").write_text("restricted\n", encoding="utf-8")
    assert restricted_release_paths(tmp_path) == ["mapping/drug.csv"]
