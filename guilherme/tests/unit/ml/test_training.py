import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from srag_api.ml.features import LEAKAGE_FEATURES


def _load_training_module():
    try:
        return importlib.import_module("srag_api.ml.training")
    except ModuleNotFoundError:
        return None


class RecordingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.fit_kwargs = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, **kwargs):
        self.fit_kwargs = kwargs
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), 0.6)
        return np.column_stack([1.0 - positive, positive])


def _simple_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["NU_IDADE_N"]),
        ],
        remainder="drop",
    )


def _training_frames():
    X_train = pd.DataFrame(
        {
            "NU_IDADE_N": [20, 40, 60, 80],
        }
    )
    y_train = pd.Series([0, 0, 1, 1])

    X_validation = pd.DataFrame(
        {
            "NU_IDADE_N": [10, 30],
        }
    )
    y_validation = pd.Series([0, 1])

    return X_train, y_train, X_validation, y_validation


def test_train_candidate_returns_probabilities_and_metrics():
    module = _load_training_module()
    assert module is not None, "srag_api.ml.training ainda nao foi implementado"

    X_train, y_train, X_validation, y_validation = _training_frames()

    result = module.train_candidate_model(
        name="dummy",
        estimator=RecordingClassifier(),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        preprocessor=_simple_preprocessor(),
    )

    assert result.name == "dummy"
    assert result.validation_probabilities.tolist() == pytest.approx([0.6, 0.6])
    assert 0.0 <= result.validation_metrics.auc_pr <= 1.0
    assert isinstance(result.pipeline, Pipeline)


def test_preprocessor_is_fitted_only_on_training_partition():
    module = _load_training_module()
    assert module is not None, "srag_api.ml.training ainda nao foi implementado"

    X_train, y_train, X_validation, y_validation = _training_frames()
    preprocessor = _simple_preprocessor()

    result = module.train_candidate_model(
        name="dummy",
        estimator=RecordingClassifier(),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        preprocessor=preprocessor,
    )

    scaler = result.pipeline.named_steps["preprocessor"].named_transformers_["num"]
    assert scaler.mean_[0] == pytest.approx(X_train["NU_IDADE_N"].mean())
    assert scaler.mean_[0] != pytest.approx(X_validation["NU_IDADE_N"].mean())


def test_gradient_boosting_receives_training_sample_weight():
    module = _load_training_module()
    assert module is not None, "srag_api.ml.training ainda nao foi implementado"

    from sklearn.ensemble import GradientBoostingClassifier

    X_train, y_train, X_validation, y_validation = _training_frames()

    result = module.train_candidate_model(
        name="gradient_boosting",
        estimator=GradientBoostingClassifier(random_state=42),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        preprocessor=_simple_preprocessor(),
    )

    assert hasattr(result.pipeline.named_steps["model"], "classes_")


def test_non_gradient_model_does_not_receive_external_sample_weight():
    module = _load_training_module()
    assert module is not None, "srag_api.ml.training ainda nao foi implementado"

    X_train, y_train, X_validation, y_validation = _training_frames()
    estimator = RecordingClassifier()

    result = module.train_candidate_model(
        name="dummy",
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        preprocessor=_simple_preprocessor(),
    )

    fitted = result.pipeline.named_steps["model"]
    assert fitted.fit_kwargs == {}


def test_training_rejects_known_leakage_feature():
    module = _load_training_module()
    assert module is not None, "srag_api.ml.training ainda nao foi implementado"

    X_train, y_train, X_validation, y_validation = _training_frames()
    leaked_column = next(iter(LEAKAGE_FEATURES))

    X_train = X_train.assign(**{leaked_column: [1, 1, 1, 1]})
    X_validation = X_validation.assign(**{leaked_column: [1, 1]})

    with pytest.raises(ValueError, match="leakage"):
        module.train_candidate_model(
            name="dummy",
            estimator=RecordingClassifier(),
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            preprocessor=_simple_preprocessor(),
        )
