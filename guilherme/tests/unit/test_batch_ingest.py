import pytest

from scripts.ingest_all import discover_year_file

def test_discover_year_file_returns_only_csv(tmp_path):
    year_dir = tmp_path / "2025"
    year_dir.mkdir()
    expected = year_dir / "INFLUD25.csv"
    expected.write_text("a;b\n1;2\n", encoding="utf-8")
    assert discover_year_file(tmp_path, 2025) == expected

def test_discover_year_file_returns_none_when_missing(tmp_path):
    assert discover_year_file(tmp_path, 2025) is None

def test_discover_year_file_refuses_ambiguous_sources(tmp_path):
    year_dir = tmp_path / "2025"
    year_dir.mkdir()
    (year_dir / "a.csv").write_text("a\n1\n", encoding="utf-8")
    (year_dir / "b.csv").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Mais de um CSV"):
        discover_year_file(tmp_path, 2025)
