from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    auc_pr: float
    roc_auc: float
    recall: float
    precision: float
    f1: float
    threshold: float
    confusion_matrix: np.ndarray


def evaluate_binary_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    if len(y_true) != len(probabilities):
        raise ValueError("y_true e probabilities devem possuir o mesmo tamanho.")

    classes = set(pd.Series(y_true).dropna().unique().tolist())
    if classes != {0, 1}:
        raise ValueError("A particao deve conter as duas classes 0 e 1.")

    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    return BinaryMetrics(
        auc_pr=float(average_precision_score(y_true, probabilities)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        threshold=float(threshold),
        confusion_matrix=confusion_matrix(y_true, predictions, labels=[0, 1]),
    )
