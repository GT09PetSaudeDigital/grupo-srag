import importlib

import numpy as np
import pandas as pd
import pytest


def _load_threshold_module():
    try:
        return importlib.import_module("srag_api.ml.threshold")
    except ModuleNotFoundError:
        return None


def test_threshold_prefers_max_recall_with_minimum_precision():
    module = _load_threshold_module()
    assert module is not None, "srag_api.ml.threshold ainda nao foi implementado"

    y_validation = pd.Series([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.6, 0.9, 0.8])

    result = module.select_decision_threshold(
        y_validation,
        probabilities,
        min_precision=0.50,
    )

    assert result.threshold == pytest.approx(0.6)
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(2 / 3)
    assert result.policy == "max_recall_precision_ge_0_50"


def test_threshold_never_uses_default_05_when_better_valid_threshold_exists():
    module = _load_threshold_module()
    assert module is not None, "srag_api.ml.threshold ainda nao foi implementado"

    result = module.select_decision_threshold(
        pd.Series([0, 1, 1, 0]),
        np.array([0.1, 0.6, 0.9, 0.8]),
        min_precision=0.50,
    )

    assert result.threshold != 0.5
    assert result.threshold == pytest.approx(0.6)


def test_threshold_falls_back_to_max_f1_when_precision_constraint_is_impossible():
    module = _load_threshold_module()
    assert module is not None, "srag_api.ml.threshold ainda nao foi implementado"

    y_validation = pd.Series([0, 0, 0, 1])
    probabilities = np.array([0.9, 0.8, 0.7, 0.1])

    result = module.select_decision_threshold(
        y_validation,
        probabilities,
        min_precision=0.50,
    )

    assert result.policy == "fallback_max_f1"
    assert result.threshold == pytest.approx(0.1)
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(0.25)
    assert result.f1 == pytest.approx(0.4)


def test_threshold_selection_rejects_single_class_validation():
    module = _load_threshold_module()
    assert module is not None, "srag_api.ml.threshold ainda nao foi implementado"

    with pytest.raises(ValueError, match="duas classes"):
        module.select_decision_threshold(
            pd.Series([0, 0, 0]),
            np.array([0.1, 0.2, 0.3]),
        )


def test_threshold_selection_rejects_length_mismatch():
    module = _load_threshold_module()
    assert module is not None, "srag_api.ml.threshold ainda nao foi implementado"

    with pytest.raises(ValueError, match="mesmo tamanho"):
        module.select_decision_threshold(
            pd.Series([0, 1]),
            np.array([0.2]),
        )
