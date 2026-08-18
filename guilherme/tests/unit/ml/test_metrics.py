import importlib

import numpy as np
import pandas as pd
import pytest


def _load_metrics_module():
    try:
        return importlib.import_module("srag_api.ml.metrics")
    except ModuleNotFoundError:
        return None


def test_metrics_return_auc_pr_roc_auc_and_threshold_metrics():
    module = _load_metrics_module()
    assert module is not None, "srag_api.ml.metrics ainda nao foi implementado"

    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    result = module.evaluate_binary_predictions(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert 0.0 <= result.auc_pr <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.f1 == 1.0
    assert result.threshold == 0.5


def test_metrics_include_2x2_confusion_matrix():
    module = _load_metrics_module()
    assert module is not None, "srag_api.ml.metrics ainda nao foi implementado"

    result = module.evaluate_binary_predictions(
        pd.Series([0, 0, 1, 1]),
        np.array([0.2, 0.8, 0.7, 0.1]),
        threshold=0.5,
    )

    assert result.confusion_matrix.shape == (2, 2)
    assert result.confusion_matrix.tolist() == [[1, 1], [1, 1]]


def test_metrics_reject_single_class_partition():
    module = _load_metrics_module()
    assert module is not None, "srag_api.ml.metrics ainda nao foi implementado"

    with pytest.raises(ValueError, match="duas classes"):
        module.evaluate_binary_predictions(
            pd.Series([0, 0, 0]),
            np.array([0.1, 0.2, 0.3]),
        )


def test_metrics_reject_probability_length_mismatch():
    module = _load_metrics_module()
    assert module is not None, "srag_api.ml.metrics ainda nao foi implementado"

    with pytest.raises(ValueError, match="mesmo tamanho"):
        module.evaluate_binary_predictions(
            pd.Series([0, 1]),
            np.array([0.2]),
        )
