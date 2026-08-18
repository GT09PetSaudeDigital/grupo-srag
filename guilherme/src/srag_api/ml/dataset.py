"""Montagem segura do dataset de admissão para Machine Learning."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import ADMISSION_FEATURES, LEAKAGE_FEATURES
from .target import build_mortality_target, eligible_outcome_mask


METADATA_COLUMNS: tuple[str, ...] = (
    "ANO",
    "SG_UF",
    "DT_NOTIFIC",
)


@dataclass(frozen=True)
class AdmissionDataset:
    """Partições lógicas do dataset de admissão."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


def build_admission_dataset(df: pd.DataFrame) -> AdmissionDataset:
    """Filtra desfechos elegíveis e monta X, y e metadados sem leakage."""
    eligible_mask = eligible_outcome_mask(df)
    filtered = df.loc[eligible_mask].copy()

    selected_features = [
        column
        for column in ADMISSION_FEATURES
        if column in filtered.columns
    ]

    leaked = set(selected_features) & LEAKAGE_FEATURES
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"Features com risco de leakage selecionadas: {names}")

    X = filtered.loc[:, selected_features].copy()
    y = build_mortality_target(filtered)

    metadata_columns = [
        column
        for column in METADATA_COLUMNS
        if column in filtered.columns
    ]
    metadata = filtered.loc[:, metadata_columns].copy()

    return AdmissionDataset(
        X=X,
        y=y,
        metadata=metadata,
    )
