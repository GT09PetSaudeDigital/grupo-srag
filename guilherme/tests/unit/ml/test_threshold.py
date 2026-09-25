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


def test_threshold_tie_break_is_unreachable_with_the_sklearn_grid():
    """Documenta por que o desempate por limiar mais alto raramente decide.

    ``precision_recall_curve`` devolve os thresholds em ordem crescente e
    cada valor e uma probabilidade observada. Dois thresholds consecutivos
    retiram do conjunto previsto ao menos um registro, de modo que TP e FP
    nao podem permanecer simultaneamente iguais. Logo, empate exato em
    recall e precision nao ocorre nesse grid, e o terceiro criterio da
    ordenacao existe apenas como rede de seguranca.

    Este teste fixa a garantia que realmente importa: a selecao e
    deterministica e nao depende da ordem de insercao.
    """
    module = _load_threshold_module()
    assert module is not None

    y_validation = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.10, 0.10, 0.62, 0.62, 0.62, 0.62, 0.90, 0.90])

    result = module.select_decision_threshold(
        y_validation,
        probabilities,
        min_precision=0.50,
    )

    # O corte 0.90 deixa de fora dois dos quatro obitos, entao o maximo de
    # recall pertence a 0.62, que cobre os quatro com recall 1.0, ainda que
    # carregando dois alarmes falsos.
    assert result.threshold == pytest.approx(0.62)
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(4 / 6)


def test_threshold_selection_is_deterministic():
    module = _load_threshold_module()
    assert module is not None

    y_validation = pd.Series([0, 1, 1, 0, 1, 0, 1, 0])
    probabilities = np.array([0.05, 0.30, 0.31, 0.55, 0.56, 0.70, 0.71, 0.95])

    results = [
        module.select_decision_threshold(
            y_validation,
            probabilities,
            min_precision=0.50,
        )
        for _ in range(5)
    ]

    assert len({(r.threshold, r.policy) for r in results}) == 1


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
