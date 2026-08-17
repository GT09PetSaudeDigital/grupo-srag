from __future__ import annotations

import pandas as pd


FINAL_CLASSIFICATION_MAP = {
    1: "INFLUENZA",
    2: "OUTRO_VIRUS_RESPIRATORIO",
    3: "OUTRO_AGENTE_ETIOLOGICO",
    4: "NAO_ESPECIFICADO",
    5: "COVID-19",
}


LAB_ETIOLOGY_FIELDS = (
    ("PCR_SARS2", "SARS-CoV-2"),
    ("PCR_FLUAS", "Influenza A"),
    ("PCR_FLUBS", "Influenza B"),
    ("PCR_VSR", "VSR"),
    ("PCR_ADENO", "Adenovirus"),
    ("PCR_PARA1", "Parainfluenza 1"),
    ("PCR_PARA2", "Parainfluenza 2"),
    ("PCR_PARA3", "Parainfluenza 3"),
    ("PCR_PARA4", "Parainfluenza 4"),
    ("PCR_METAP", "Metapneumovirus"),
    ("PCR_BOCA", "Bocavirus"),
    ("PCR_RINO", "Rinovirus"),
)



def _is_positive(row: pd.Series, field: str) -> bool:
    return field in row.index and row.get(field) == 1


def normalize_final_classification(value: object) -> str:
    if pd.isna(value):
        return "AUSENTE"
    return FINAL_CLASSIFICATION_MAP.get(value, "OUTRO")


def normalize_detailed_etiology(row: pd.Series) -> str:
    for field, label in LAB_ETIOLOGY_FIELDS:
        if _is_positive(row, field):
            return label
    return "NAO_IDENTIFICADA"


def add_etiology_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "CLASSI_FIN" in result.columns:
        result["CLASSIFICACAO_FINAL_NORMALIZADA"] = result[
            "CLASSI_FIN"
        ].map(normalize_final_classification)
    else:
        result["CLASSIFICACAO_FINAL_NORMALIZADA"] = "AUSENTE"

    result["ETIOLOGIA_DETALHADA"] = result.apply(
        normalize_detailed_etiology,
        axis=1,
    )
    return result
