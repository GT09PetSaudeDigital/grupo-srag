from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .features import LEAKAGE_FEATURES
from .metrics import BinaryMetrics, evaluate_binary_predictions
from .models import build_gradient_boosting_sample_weight


@dataclass(frozen=True)
class TrainedCandidate:
    name: str
    pipeline: Pipeline
    validation_probabilities: np.ndarray
    validation_metrics: BinaryMetrics


def _validate_no_leakage(columns) -> None:
    leaked = set(columns) & set(LEAKAGE_FEATURES)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"Features de leakage detectadas no treinamento: {names}")


def train_candidate_model(
    *,
    name: str,
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    preprocessor,
) -> TrainedCandidate:
    _validate_no_leakage(X_train.columns)
    _validate_no_leakage(X_validation.columns)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    if isinstance(estimator, GradientBoostingClassifier):
        sample_weight = build_gradient_boosting_sample_weight(y_train)
        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weight,
        )
    else:
        pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_validation)[:, 1]
    validation_metrics = evaluate_binary_predictions(
        y_validation,
        probabilities,
        threshold=0.5,
    )

    return TrainedCandidate(
        name=name,
        pipeline=pipeline,
        validation_probabilities=np.asarray(probabilities, dtype=float),
        validation_metrics=validation_metrics,
    )
