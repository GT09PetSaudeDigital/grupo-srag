from pathlib import Path
import pandas as pd
import pytest

from srag_api.data.ingest import ingest_year, read_srag_csv, transform_srag_dataframe, write_year_parquet

FIXTURE = Path(__file__).parents[1] / "fixtures" / "sample_srag.csv"

def test_read_srag_csv_reads_semicolon_file():
    df = read_srag_csv(FIXTURE)
    assert len(df) == 3
    assert "TP_IDADE" in df.columns

def test_transform_srag_dataframe_adds_analytic_columns():
    result = transform_srag_dataframe(read_srag_csv(FIXTURE), 2025)
    for col in ["ANO","IDADE_ANOS","FAIXA_ETARIA","ETIOLOGIA_NORMALIZADA","DESFECHO_NORMALIZADO","FOI_UTI","OBITO_SRAG","CODIGO_MUNICIPIO","MUNICIPIO","UF"]:
        assert col in result.columns

def test_write_year_parquet_uses_partition_directory(tmp_path):
    df = transform_srag_dataframe(read_srag_csv(FIXTURE), 2025)
    path = write_year_parquet(df, tmp_path / "parquet", 2025)
    assert path.exists()
    assert len(pd.read_parquet(path)) == 3

def test_ingest_year_writes_parquet_and_quality_report(tmp_path):
    path = ingest_year(FIXTURE, tmp_path / "parquet", tmp_path / "quality", 2025)
    assert path.exists()
    assert (tmp_path / "quality" / "quality_2025.json").exists()

def test_ingest_year_is_incremental_without_force(tmp_path):
    ingest_year(FIXTURE, tmp_path / "parquet", tmp_path / "quality", 2025)
    with pytest.raises(FileExistsError, match="2025"):
        ingest_year(FIXTURE, tmp_path / "parquet", tmp_path / "quality", 2025, force=False)

def test_transform_rejects_unsupported_year():
    with pytest.raises(ValueError, match="Ano nao suportado"):
        transform_srag_dataframe(read_srag_csv(FIXTURE), 2018)
