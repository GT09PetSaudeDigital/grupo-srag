"""Divisão temporal para validação do modelo SRAG."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Índices posicionais das partições temporais."""

    train_idx: list[int]
    validation_idx: list[int]
    test_idx: list[int]


def temporal_split(
    years: pd.Series,
    validation_year: int = 2025,
    test_year: int = 2026,
) -> TemporalSplit:
    """Separa treino, validação e teste respeitando a ordem temporal."""
    if years.isna().any():
        raise ValueError("A coluna de ano contém valores ausentes.")

    train_mask = years < validation_year
    validation_mask = years == validation_year
    test_mask = years == test_year

    if not train_mask.any():
        raise ValueError(
            f"Nenhum registro de treino anterior a {validation_year} foi encontrado."
        )

    if not validation_mask.any():
        raise ValueError(
            f"Nenhum registro encontrado para o ano de validação {validation_year}."
        )

    if not test_mask.any():
        raise ValueError(
            f"Nenhum registro encontrado para o ano de teste {test_year}."
        )

    train_idx = [position for position, value in enumerate(train_mask.tolist()) if value]
    validation_idx = [
        position for position, value in enumerate(validation_mask.tolist()) if value
    ]
    test_idx = [position for position, value in enumerate(test_mask.tolist()) if value]

    train_set = set(train_idx)
    validation_set = set(validation_idx)
    test_set = set(test_idx)

    if not train_set.isdisjoint(validation_set):
        raise ValueError("As partições de treino e validação se sobrepõem.")

    if not train_set.isdisjoint(test_set):
        raise ValueError("As partições de treino e teste se sobrepõem.")

    if not validation_set.isdisjoint(test_set):
        raise ValueError("As partições de validação e teste se sobrepõem.")

    return TemporalSplit(
        train_idx=train_idx,
        validation_idx=validation_idx,
        test_idx=test_idx,
    )
