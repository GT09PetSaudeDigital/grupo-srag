from __future__ import annotations

from collections.abc import Callable
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


def _silent_progress(message: str) -> None:
    """Descarta as mensagens de etapa quando nenhum callback e informado."""


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


def _train_transformed_candidate(
    *,
    name: str,
    estimator: BaseEstimator,
    X_train,
    y_train: pd.Series,
    X_validation,
    y_validation: pd.Series,
    fitted_preprocessor,
) -> TrainedCandidate:
    if isinstance(estimator, GradientBoostingClassifier):
        sample_weight = build_gradient_boosting_sample_weight(y_train)
        estimator.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        estimator.fit(X_train, y_train)

    probabilities = estimator.predict_proba(X_validation)[:, 1]
    validation_metrics = evaluate_binary_predictions(
        y_validation,
        probabilities,
        threshold=0.5,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", fitted_preprocessor),
            ("model", estimator),
        ]
    )

    return TrainedCandidate(
        name=name,
        pipeline=pipeline,
        validation_probabilities=np.asarray(probabilities, dtype=float),
        validation_metrics=validation_metrics,
    )


NUMERIC_MODEL_FEATURES = frozenset({"NU_IDADE_N", "SINT_ATE_NOTIF"})


def split_preprocessing_features(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric = [column for column in X.columns if column in NUMERIC_MODEL_FEATURES]
    categorical = [
        column for column in X.columns if column not in NUMERIC_MODEL_FEATURES
    ]
    return numeric, categorical


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
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    min_precision: float = 0.50,
    random_state: int = 42,
    progress: Callable[[str], None] | None = None,
) -> TrainingRunResult:
    """Treina os candidatos, escolhe o vencedor e avalia o teste temporal.

    ``progress`` recebe as mensagens de etapa. O padrão e ``None``, e nesse
    caso nada e impresso: a camada de biblioteca nao escreve em saida. O
    script de linha de comando injeta ``print`` para acompanhar execucoes
    longas, que podem levar horas sobre a base nacional.
    """
    from .models import build_models
    from .preprocessing import (
        build_hist_gradient_boosting_preprocessor,
        build_preprocessor,
    )

    report = progress if progress is not None else _silent_progress

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

    if numeric_features is None and categorical_features is None:
        numeric_features, categorical_features = split_preprocessing_features(X_train)
    elif numeric_features is None or categorical_features is None:
        raise ValueError(
            "numeric_features e categorical_features devem ser informadas juntas."
        )

    models = build_models(random_state=random_state)
    candidates: dict[str, TrainedCandidate] = {}

    report("[PREPROCESS] ajustando preprocessing compartilhado...")
    sparse_preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    sparse_preprocessor.fit(X_train)
    report("[PREPROCESS] transformando treino...")
    X_train_sparse = sparse_preprocessor.transform(X_train)
    report("[PREPROCESS] transformando validacao...")
    X_validation_sparse = sparse_preprocessor.transform(X_validation)

    for name in (
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ):
        report(f"[TRAIN] {name}...")
        candidates[name] = _train_transformed_candidate(
            name=name,
            estimator=models[name],
            X_train=X_train_sparse,
            y_train=y_train,
            X_validation=X_validation_sparse,
            y_validation=y_validation,
            fitted_preprocessor=sparse_preprocessor,
        )
        report(
            f"[OK] {name} AUC-PR="
            f"{candidates[name].validation_metrics.auc_pr:.4f}"
        )

    del X_train_sparse, X_validation_sparse

    report("[PREPROCESS] preparando hist_gradient_boosting...")
    hist_preprocessor = build_hist_gradient_boosting_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    X_train_hist = hist_preprocessor.fit_transform(X_train)
    X_validation_hist = hist_preprocessor.transform(X_validation)
    hist_name = "hist_gradient_boosting"
    report(f"[TRAIN] {hist_name}...")
    candidates[hist_name] = _train_transformed_candidate(
        name=hist_name,
        estimator=models[hist_name],
        X_train=X_train_hist,
        y_train=y_train,
        X_validation=X_validation_hist,
        y_validation=y_validation,
        fitted_preprocessor=hist_preprocessor,
    )
    report(
        f"[OK] {hist_name} AUC-PR="
        f"{candidates[hist_name].validation_metrics.auc_pr:.4f}"
    )

    best = select_best_candidate(candidates)
    report(f"[SELECT] melhor modelo: {best.name}")

    threshold_selection = select_decision_threshold(
        y_validation,
        best.validation_probabilities,
        min_precision=min_precision,
    )
    report(
        f"[THRESHOLD] {threshold_selection.threshold:.6f} "
        f"({threshold_selection.policy})"
    )

    validation_metrics = evaluate_binary_predictions(
        y_validation,
        best.validation_probabilities,
        threshold=threshold_selection.threshold,
    )

    report("[TEST] avaliando 2026...")
    best_preprocessor = best.pipeline.named_steps["preprocessor"]
    best_estimator = best.pipeline.named_steps["model"]
    X_test_transformed = best_preprocessor.transform(X_test)
    test_probabilities = best_estimator.predict_proba(X_test_transformed)[:, 1]
    test_metrics = evaluate_binary_predictions(
        y_test,
        test_probabilities,
        threshold=threshold_selection.threshold,
    )
    report(f"[OK] teste final AUC-PR={test_metrics.auc_pr:.4f}")

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

