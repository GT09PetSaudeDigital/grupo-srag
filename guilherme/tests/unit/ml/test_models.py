import importlib

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression


def _load_models_module():
    try:
        return importlib.import_module("srag_api.ml.models")
    except ModuleNotFoundError:
        return None


def test_build_models_registers_exactly_four_v1_models():
    module = _load_models_module()
    assert module is not None, "srag_api.ml.models ainda nao foi implementado"

    models = module.build_models()

    assert list(models) == [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
    ]


def test_model_types_are_expected():
    module = _load_models_module()
    assert module is not None, "srag_api.ml.models ainda nao foi implementado"

    models = module.build_models()

    assert isinstance(models["logistic_regression"], LogisticRegression)
    assert isinstance(models["random_forest"], RandomForestClassifier)
    assert isinstance(models["gradient_boosting"], GradientBoostingClassifier)
    assert isinstance(models["hist_gradient_boosting"], HistGradientBoostingClassifier)


def test_supported_models_use_class_balancing_when_available():
    module = _load_models_module()
    assert module is not None, "srag_api.ml.models ainda nao foi implementado"

    models = module.build_models()

    assert models["logistic_regression"].class_weight == "balanced"
    assert models["random_forest"].class_weight in {
        "balanced",
        "balanced_subsample",
    }
    assert models["hist_gradient_boosting"].class_weight == "balanced"


def test_gradient_boosting_sample_weight_is_derived_from_training_labels():
    module = _load_models_module()
    assert module is not None, "srag_api.ml.models ainda nao foi implementado"

    y_train = pd.Series([0, 0, 0, 1])
    weights = module.build_gradient_boosting_sample_weight(y_train)

    assert len(weights) == len(y_train)
    assert weights[y_train == 1][0] > weights[y_train == 0][0]
