"""Pré-processamento sem leakage para o ML de admissão SRAG."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Monta o pré-processador para features numéricas e categóricas."""
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(numeric_features)))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, list(categorical_features))
        )

    if not transformers:
        raise ValueError(
            "Informe ao menos uma feature numerica ou categorica para o pre-processador."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def fit_preprocessor_on_train(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Ajusta o pré-processador exclusivamente com os dados de treino."""
    return preprocessor.fit(X_train)


def transform_partitions(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """Transforma treino, validação e teste sem realizar novo ajuste."""
    train_transformed = preprocessor.transform(X_train)
    validation_transformed = preprocessor.transform(X_validation)
    test_transformed = preprocessor.transform(X_test)

    return (
        train_transformed,
        validation_transformed,
        test_transformed,
    )
def balance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    strategy: str = "none",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica balanceamento somente aos dados de treino.

    Nesta V1, ``strategy="none"`` é suportada explicitamente.
    Estratégias como SMOTE serão adicionadas apenas na etapa de
    experimentação de modelos, após validação da representação das features.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train e y_train devem possuir o mesmo tamanho."
        )

    if strategy != "none":
        raise ValueError(
            f"Estrategia de balanceamento nao suportada: {strategy}"
        )

    _ = random_state

    return X_train.copy(), y_train.copy()
