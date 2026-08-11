from __future__ import annotations

import math
import pandas as pd

from srag_api.config import AGE_BANDS


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def normalize_age(tp_idade: object, nu_idade_n: object) -> float | None:
    unit = _to_float(tp_idade)
    value = _to_float(nu_idade_n)

    if unit is None or value is None or value < 0:
        return None

    if unit == 1:
        age_years = value / 365.25
    elif unit == 2:
        age_years = value / 12.0
    elif unit == 3:
        age_years = value
    else:
        return None

    if age_years > 120:
        return None

    return age_years


def classify_age_band(age_years: float | None) -> str | None:
    if age_years is None:
        return None

    for lower, upper, label in AGE_BANDS:
        if age_years >= lower and (upper is None or age_years < upper):
            return label

    return None


def add_normalized_age_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["IDADE_ANOS"] = [
        normalize_age(tp_idade, nu_idade_n)
        for tp_idade, nu_idade_n in zip(
            result["TP_IDADE"],
            result["NU_IDADE_N"],
        )
    ]
    result["FAIXA_ETARIA"] = result["IDADE_ANOS"].map(classify_age_band)
    return result


def normalize_yes_no(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "AUSENTE"
    if number == 1:
        return "SIM"
    if number == 2:
        return "NAO"
    if number == 9:
        return "IGNORADO"
    return "IGNORADO"


def normalize_outcome(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "AUSENTE"
    if number == 1:
        return "CURA"
    if number == 2:
        return "OBITO_SRAG"
    if number == 3:
        return "OBITO_OUTRAS_CAUSAS"
    if number == 9:
        return "IGNORADO"
    return "OUTRO"


def add_core_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["DESFECHO_NORMALIZADO"] = result["EVOLUCAO"].map(normalize_outcome)
    result["FOI_UTI"] = result["UTI"].map(normalize_yes_no)
    result["OBITO_SRAG"] = result["DESFECHO_NORMALIZADO"].eq("OBITO_SRAG")
    return result


def add_geography_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["UF"] = result["SG_UF"].astype("string").str.strip().str.upper()
    result["MUNICIPIO"] = (
        result["ID_MUNICIP"].astype("string").str.strip().str.upper()
    )

    if "CO_MUN_RES" in result.columns:
        result["CODIGO_MUNICIPIO"] = pd.to_numeric(
            result["CO_MUN_RES"],
            errors="coerce",
        ).astype("Int64")
    else:
        result["CODIGO_MUNICIPIO"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

    return result


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preserva tempo epidemiológico sem inventar semana quando SEM_PRI não existe."""
    result = df.copy()

    if "DT_SIN_PRI" in result.columns:
        onset = pd.to_datetime(
            result["DT_SIN_PRI"],
            errors="coerce",
            dayfirst=True,
        )
    else:
        onset = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")

    result["DATA_INICIO_SINTOMAS"] = onset
    result["MES"] = onset.dt.month.astype("Int64")

    if "SEM_PRI" in result.columns:
        result["SEMANA_EPIDEMIOLOGICA"] = pd.to_numeric(
            result["SEM_PRI"],
            errors="coerce",
        ).astype("Int64")
    else:
        result["SEMANA_EPIDEMIOLOGICA"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

    return result
