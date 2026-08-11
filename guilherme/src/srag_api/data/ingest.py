from __future__ import annotations

from pathlib import Path

import pandas as pd

from srag_api.config import SUPPORTED_YEARS
from srag_api.data.clean import (
    add_core_normalized_columns,
    add_geography_columns,
    add_normalized_age_columns,
    add_temporal_columns,
)
from srag_api.data.etiology import add_etiology_column
from srag_api.data.quality import build_quality_report, write_quality_report
from srag_api.data.schema import normalize_column_names, validate_required_columns


def read_srag_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        low_memory=False,
    )


def transform_srag_dataframe(
    df: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    if year not in SUPPORTED_YEARS:
        raise ValueError(f"Ano nao suportado: {year}")

    result = normalize_column_names(df)
    validate_required_columns(result)
    result = result.drop_duplicates().copy()
    result = add_normalized_age_columns(result)
    result = add_core_normalized_columns(result)
    result = add_geography_columns(result)
    result = add_temporal_columns(result)
    result = add_etiology_column(result)
    result["ANO"] = year

    return result


def write_year_parquet(
    df: pd.DataFrame,
    base_dir: Path,
    year: int,
) -> Path:
    output_path = (
        base_dir
        / "srag"
        / f"ano={year}"
        / "srag.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def ingest_year(
    input_path: Path,
    parquet_root: Path,
    quality_root: Path,
    year: int,
    force: bool = False,
) -> Path:
    output_path = (
        parquet_root
        / "srag"
        / f"ano={year}"
        / "srag.parquet"
    )

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Ano {year} ja foi processado. Use force=True para reprocessar."
        )

    raw_df = read_srag_csv(input_path)
    processed_df = transform_srag_dataframe(raw_df, year)

    parquet_path = write_year_parquet(
        processed_df,
        parquet_root,
        year,
    )

    report = build_quality_report(
        raw_df,
        processed_df,
        year,
    )
    write_quality_report(
        report,
        quality_root / f"quality_{year}.json",
    )

    return parquet_path
