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
from .threshold import select_decision_threshold


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

MODEL_SELECTION_ORDER = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
)


def select_best_candidate(
    candidates: dict[str, TrainedCandidate],
) -> TrainedCandidate:
    if not candidates:
        raise ValueError("E necessario fornecer ao menos um candidato.")

    registry_rank = {
        name: index
        for index, name in enumerate(MODEL_SELECTION_ORDER)
    }

    return max(
        candidates.values(),
        key=lambda candidate: (
            candidate.validation_metrics.auc_pr,
            -registry_rank.get(candidate.name, len(MODEL_SELECTION_ORDER)),
        ),
    )

@dataclass(frozen=True)
class TrainingRunResult:
    candidates: dict[str, TrainedCandidate]
    best_model_name: str
    best_pipeline: Pipeline
    threshold: float
    threshold_policy: str
    validation_metrics: BinaryMetrics
    test_metrics: BinaryMetrics
    train_size: int
    validation_size: int
    test_size: int


def _validate_binary_partition(y: pd.Series, partition_name: str) -> None:
    classes = set(pd.Series(y).dropna().unique().tolist())
    if classes != {0, 1}:
        raise ValueError(
            f"A particao {partition_name} deve conter as duas classes 0 e 1."
        )


def run_admission_training(
    dataset,
    split,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    min_precision: float = 0.50,
    random_state: int = 42,
) -> TrainingRunResult:
    from .models import build_models
    from .preprocessing import build_preprocessor

    X_train = dataset.X.iloc[split.train_idx].copy()
    X_validation = dataset.X.iloc[split.validation_idx].copy()
    X_test = dataset.X.iloc[split.test_idx].copy()

    y_train = dataset.y.iloc[split.train_idx].copy()
    y_validation = dataset.y.iloc[split.validation_idx].copy()
    y_test = dataset.y.iloc[split.test_idx].copy()

    _validate_binary_partition(y_train, "treino")
    _validate_binary_partition(y_validation, "validacao")
    _validate_binary_partition(y_test, "teste")

    _validate_no_leakage(X_train.columns)
    _validate_no_leakage(X_validation.columns)
    _validate_no_leakage(X_test.columns)

    candidates: dict[str, TrainedCandidate] = {}

    for name, estimator in build_models(random_state=random_state).items():
        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        candidates[name] = train_candidate_model(
            name=name,
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            preprocessor=preprocessor,
        )

    best = select_best_candidate(candidates)

    threshold_selection = select_decision_threshold(
        y_validation,
        best.validation_probabilities,
        min_precision=min_precision,
    )

    validation_metrics = evaluate_binary_predictions(
        y_validation,
        best.validation_probabilities,
        threshold=threshold_selection.threshold,
    )

    test_probabilities = best.pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_binary_predictions(
        y_test,
        test_probabilities,
        threshold=threshold_selection.threshold,
    )

    return TrainingRunResult(
        candidates=candidates,
        best_model_name=best.name,
        best_pipeline=best.pipeline,
        threshold=threshold_selection.threshold,
        threshold_policy=threshold_selection.policy,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        train_size=len(X_train),
        validation_size=len(X_validation),
        test_size=len(X_test),
    )

