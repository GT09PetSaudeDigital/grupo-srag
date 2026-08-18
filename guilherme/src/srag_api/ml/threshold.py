from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


PRIMARY_POLICY = "max_recall_precision_ge_0_50"
FALLBACK_POLICY = "fallback_max_f1"


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    policy: str
    precision: float
    recall: float
    f1: float


def select_decision_threshold(
    y_validation: pd.Series,
    probabilities: np.ndarray,
    *,
    min_precision: float = 0.50,
) -> ThresholdSelection:
    if len(y_validation) != len(probabilities):
        raise ValueError(
            "y_validation e probabilities devem possuir o mesmo tamanho."
        )

    classes = set(pd.Series(y_validation).dropna().unique().tolist())
    if classes != {0, 1}:
        raise ValueError(
            "A particao de validacao deve conter as duas classes 0 e 1."
        )

    probabilities = np.asarray(probabilities, dtype=float)
    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        probabilities,
    )

    candidate_precision = precision[:-1]
    candidate_recall = recall[:-1]

    f1 = (
        2.0
        * candidate_precision
        * candidate_recall
        / (candidate_precision + candidate_recall + 1e-12)
    )

    valid = np.flatnonzero(candidate_precision >= min_precision)

    if len(valid):
        best_idx = max(
            valid.tolist(),
            key=lambda idx: (
                candidate_recall[idx],
                candidate_precision[idx],
                thresholds[idx],
            ),
        )
        policy = PRIMARY_POLICY
    else:
        best_idx = max(
            range(len(thresholds)),
            key=lambda idx: (
                f1[idx],
                candidate_recall[idx],
                thresholds[idx],
            ),
        )
        policy = FALLBACK_POLICY

    return ThresholdSelection(
        threshold=float(thresholds[best_idx]),
        policy=policy,
        precision=float(candidate_precision[best_idx]),
        recall=float(candidate_recall[best_idx]),
        f1=float(f1[best_idx]),
    )
