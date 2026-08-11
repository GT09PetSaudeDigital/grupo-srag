from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

def _missing_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return len(df)
    return int(df[column].isna().sum())

def build_quality_report(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    year: int,
) -> dict[str, object]:
    etiology = processed_df.get(
        "ETIOLOGIA_NORMALIZADA",
        pd.Series(index=processed_df.index, dtype="object"),
    )

    return {
        "ano": year,
        "registros_recebidos": int(len(raw_df)),
        "registros_processados": int(len(processed_df)),
        "duplicados": int(raw_df.duplicated().sum()),
        "idade_ausente": _missing_count(processed_df, "IDADE_ANOS"),
        "sexo_ausente": _missing_count(processed_df, "CS_SEXO"),
        "municipio_ausente": _missing_count(processed_df, "ID_MUNICIP"),
        "evolucao_ausente": _missing_count(processed_df, "EVOLUCAO"),
        "uti_ausente": _missing_count(processed_df, "UTI"),
        "etiologia_nao_identificada": int((etiology == "Nao identificado").sum()),
    }

def write_quality_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
