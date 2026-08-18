"""Construção do alvo binário de mortalidade para o ML de admissão."""

from __future__ import annotations

import pandas as pd


TARGET_MAPPING: dict[str, int] = {
    "CURA": 0,
    "OBITO_SRAG": 1,
}


def eligible_outcome_mask(
    df: pd.DataFrame,
    source_column: str = "DESFECHO_NORMALIZADO",
) -> pd.Series:
    """Indica quais registros possuem desfecho elegível para o alvo binário."""
    return df[source_column].isin(TARGET_MAPPING)


def build_mortality_target(
    df: pd.DataFrame,
    source_column: str = "DESFECHO_NORMALIZADO",
) -> pd.Series:
    """Mapeia CURA para 0 e OBITO_SRAG para 1; demais valores viram NA."""
    return df[source_column].map(TARGET_MAPPING).astype("Int64")
