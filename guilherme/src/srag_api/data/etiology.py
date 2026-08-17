from __future__ import annotations

import pandas as pd
FINAL_CLASSIFICATION_MAP = {
    1: "INFLUENZA",
    2: "OUTRO_VIRUS_RESPIRATORIO",
    3: "OUTRO_AGENTE_ETIOLOGICO",
    4: "NAO_ESPECIFICADO",
    5: "COVID-19",
}


def normalize_final_classification(value: object) -> str:
    if pd.isna(value):
        return "AUSENTE"

    return FINAL_CLASSIFICATION_MAP.get(value, "OUTRO")

OTHER_RESPIRATORY_FLAGS = (
    "PCR_ADENO",
    "PCR_PARA1",
    "PCR_PARA2",
    "PCR_PARA3",
    "PCR_PARA4",
    "PCR_METAP",
    "PCR_BOCA",
    "PCR_RINO",
)

def _is_positive(row: pd.Series, field: str) -> bool:
    return field in row.index and row.get(field) == 1

def normalize_etiology(row: pd.Series) -> str:
    final_classification = row.get("CLASSI_FIN")

    if final_classification == 5 or _is_positive(row, "PCR_SARS2"):
        return "COVID-19"
    if _is_positive(row, "PCR_FLUAS"):
        return "Influenza A"
    if _is_positive(row, "PCR_FLUBS"):
        return "Influenza B"
    if _is_positive(row, "PCR_VSR"):
        return "VSR"
    if any(_is_positive(row, field) for field in OTHER_RESPIRATORY_FLAGS):
        return "Outros virus respiratorios"
    if final_classification == 9:
        return "Ignorado"
    if pd.notna(final_classification):
        return "Outro agente"
    return "Nao identificado"

def add_etiology_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ETIOLOGIA_NORMALIZADA"] = result.apply(normalize_etiology, axis=1)
    return result
