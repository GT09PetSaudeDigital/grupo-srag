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
    """Escolhe o limiar de decisao usando somente a validacao.
    A politica principal maximiza o recall entre os limiares com
    ``precision >= min_precision``. Como ``precision_recall_curve`` devolve
    os thresholds em ordem crescente, o desempate por maior valor escolhe o
    limiar mais alto, isto e, o mais conservador, que emite menos alertas.
    Em caso de empate completo, o fallback maximiza F1 e usa o mesmo criterio
    conservador.

    Esse desempate e uma rede de seguranca, nao um caminho frequente. O grid do
    sklearn tem um valor por probabilidade observada, e dois thresholds
    consecutivos retiram do conjunto previsto ao menos um registro, de modo
    que TP e FP nao podem ficar simultaneamente iguais. Empate exato em recall
    e precision, portanto, nao ocorre nesse grid. A garantia testada e a
    determinismo da selecao, que nao pode depender da ordem de insercao.
    """
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
