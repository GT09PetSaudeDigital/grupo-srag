import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def test_gradient_boosting_sample_weight_is_balanced_and_covers_every_row():
    """O peso balanceado precisa chegar ao estimador, um peso por linha.

    A verificacao anterior apenas checava ``classes_``, que existe mesmo sem
    ``sample_weight``. Aqui o peso e conferido contra
    ``compute_sample_weight(class_weight='balanced')`` e a contagem de linhas
    precisa bater com a particao de treino.
    """
    module = _load_training_module()
    assert module is not None

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    class _RecordingGradientBoosting(GradientBoostingClassifier):
        """Subclasse real, para passar no isinstance, que grava o fit."""

        def fit(self, X, y, **kwargs):
            self.fit_kwargs = kwargs
            return super().fit(X, y, **kwargs)

    X_train, y_train, X_validation, y_validation = _training_frames()

    # Alvo desbalanceado de proposito: com um fixture equilibrado o peso
    # "balanced" seria 1.0 para todas as linhas e o teste nao provaria nada.
    imbalanced = y_train.copy()
    imbalanced.iloc[1:] = 0
    imbalanced.iloc[0] = 1

    result = module.train_candidate_model(
        name="gradient_boosting",
        estimator=_RecordingGradientBoosting(n_estimators=5, random_state=42),
        X_train=X_train,
        y_train=imbalanced,
        X_validation=X_validation,
        y_validation=y_validation,
        preprocessor=_simple_preprocessor(),
    )

    model = result.pipeline.named_steps["model"]
    assert "sample_weight" in model.fit_kwargs

    weights = model.fit_kwargs["sample_weight"]
    expected = compute_sample_weight(class_weight="balanced", y=imbalanced)

    assert len(weights) == len(imbalanced)
    assert np.allclose(weights, expected)
    assert weights.max() > weights.min()



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

def _candidate_with_auc(module, name: str, auc_pr: float):
    from srag_api.ml.metrics import BinaryMetrics

    metrics = BinaryMetrics(
        auc_pr=auc_pr,
        roc_auc=0.5,
        recall=0.5,
        precision=0.5,
        f1=0.5,
        threshold=0.5,
        confusion_matrix=np.array([[1, 0], [0, 1]]),
    )

    return module.TrainedCandidate(
        name=name,
        pipeline=Pipeline([]),
        validation_probabilities=np.array([0.2, 0.8]),
        validation_metrics=metrics,
    )


def test_best_candidate_is_selected_by_auc_pr():
    module = _load_training_module()
    assert module is not None, "srag_api.ml.training ainda nao foi implementado"

    candidates = {
        "logistic_regression": _candidate_with_auc(module, "logistic_regression", 0.60),
        "random_forest": _candidate_with_auc(module, "random_forest", 0.72),
        "gradient_boosting": _candidate_with_auc(module, "gradient_boosting", 0.68),
        "hist_gradient_boosting": _candidate_with_auc(module, "hist_gradient_boosting", 0.70),
    }

    best = module.select_best_candidate(candidates)

    assert best.name == "random_forest"
    assert best.validation_metrics.auc_pr == pytest.approx(0.72)


def test_auc_pr_tie_is_resolved_by_model_registry_order():
    module = _load_training_module()
    assert module is not None, "srag_api.ml.training ainda nao foi implementado"

    candidates = {
        "hist_gradient_boosting": _candidate_with_auc(module, "hist_gradient_boosting", 0.80),
        "random_forest": _candidate_with_auc(module, "random_forest", 0.80),
        "gradient_boosting": _candidate_with_auc(module, "gradient_boosting", 0.75),
        "logistic_regression": _candidate_with_auc(module, "logistic_regression", 0.80),
    }

    best = module.select_best_candidate(candidates)

    assert best.name == "logistic_regression"

def _build_temporal_training_fixture():
    from srag_api.ml.dataset import AdmissionDataset
    from srag_api.ml.split import TemporalSplit

    X = pd.DataFrame(
        {
            "NU_IDADE_N": [20, 40, 60, 80, 30, 70, 35, 75],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 1, 0], dtype="Int64")
    metadata = pd.DataFrame(
        {
            "ANO": [2023, 2024, 2023, 2024, 2025, 2025, 2026, 2026],
        }
    )

    dataset = AdmissionDataset(X=X, y=y, metadata=metadata)
    split = TemporalSplit(
        train_idx=[0, 1, 2, 3],
        validation_idx=[4, 5],
        test_idx=[6, 7],
    )
    return dataset, split


def test_run_admission_training_uses_temporal_partitions():
    module = _load_training_module()
    assert module is not None, "srag_api.ml.training ainda nao foi implementado"
    assert hasattr(module, "run_admission_training"), (
        "run_admission_training ainda nao foi implementado"
    )

    dataset, split = _build_temporal_training_fixture()

    result = module.run_admission_training(
        dataset,
        split,
        numeric_features=["NU_IDADE_N"],
        categorical_features=[],
    )

    assert result.train_size == 4
    assert result.validation_size == 2
    assert result.test_size == 2
    assert set(result.candidates) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
    }
    assert result.best_model_name in result.candidates


def test_run_admission_training_fits_shared_one_hot_preprocessor_once(monkeypatch):
    module = _load_training_module()
    assert module is not None

    dataset, split = _build_temporal_training_fixture()
    dataset.X["CS_SEXO"] = ["F", "M", "F", "M", "F", "M", "F", "M"]
    fit_calls = 0
    original_fit = OneHotEncoder.fit

    def recording_fit(self, X, y=None, **fit_params):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, X, y, **fit_params)

    monkeypatch.setattr(OneHotEncoder, "fit", recording_fit)

    module.run_admission_training(
        dataset,
        split,
        numeric_features=["NU_IDADE_N"],
        categorical_features=["CS_SEXO"],
    )

    assert fit_calls == 1


def test_run_admission_training_selects_threshold_from_validation_only(monkeypatch):
    module = _load_training_module()
    assert module is not None
    assert hasattr(module, "run_admission_training")

    dataset, split = _build_temporal_training_fixture()
    captured = {}

    original_select = module.select_decision_threshold

    def recording_select(y_validation, probabilities, *, min_precision=0.50):
        captured["y"] = list(y_validation)
        captured["n"] = len(probabilities)
        return original_select(
            y_validation,
            probabilities,
            min_precision=min_precision,
        )

    monkeypatch.setattr(module, "select_decision_threshold", recording_select)

    module.run_admission_training(
        dataset,
        split,
        numeric_features=["NU_IDADE_N"],
        categorical_features=[],
    )

    assert captured["y"] == dataset.y.iloc[split.validation_idx].tolist()
    assert captured["n"] == len(split.validation_idx)


def test_run_admission_training_evaluates_test_only_after_selection(monkeypatch):
    module = _load_training_module()
    assert module is not None
    assert hasattr(module, "run_admission_training")

    dataset, split = _build_temporal_training_fixture()
    events = []

    original_select_best = module.select_best_candidate
    original_select_threshold = module.select_decision_threshold
    original_evaluate = module.evaluate_binary_predictions

    def recording_best(candidates):
        events.append("select_model")
        return original_select_best(candidates)

    def recording_threshold(y_validation, probabilities, *, min_precision=0.50):
        events.append("select_threshold")
        return original_select_threshold(
            y_validation,
            probabilities,
            min_precision=min_precision,
        )

    def recording_evaluate(y_true, probabilities, *, threshold=0.5):
        if list(y_true) == dataset.y.iloc[split.test_idx].tolist():
            events.append("evaluate_test")
        return original_evaluate(y_true, probabilities, threshold=threshold)

    monkeypatch.setattr(module, "select_best_candidate", recording_best)
    monkeypatch.setattr(module, "select_decision_threshold", recording_threshold)
    monkeypatch.setattr(module, "evaluate_binary_predictions", recording_evaluate)

    module.run_admission_training(
        dataset,
        split,
        numeric_features=["NU_IDADE_N"],
        categorical_features=[],
    )

    assert events.index("select_model") < events.index("select_threshold")
    assert events.index("select_threshold") < events.index("evaluate_test")


def test_run_admission_training_rejects_single_class_partition():
    module = _load_training_module()
    assert module is not None
    assert hasattr(module, "run_admission_training")

    dataset, split = _build_temporal_training_fixture()
    bad_y = dataset.y.copy()
    bad_y.iloc[split.validation_idx] = 0

    from srag_api.ml.dataset import AdmissionDataset

    bad_dataset = AdmissionDataset(
        X=dataset.X,
        y=bad_y,
        metadata=dataset.metadata,
    )

    with pytest.raises(ValueError, match="duas classes"):
        module.run_admission_training(
            bad_dataset,
            split,
            numeric_features=["NU_IDADE_N"],
            categorical_features=[],
        )


def test_run_admission_training_is_silent_without_progress_callback(capsys):
    module = _load_training_module()
    assert module is not None
    assert hasattr(module, "run_admission_training")

    dataset, split = _build_temporal_training_fixture()

    module.run_admission_training(
        dataset,
        split,
        numeric_features=["NU_IDADE_N"],
        categorical_features=[],
    )

    assert capsys.readouterr().out == ""


def test_run_admission_training_reports_progress_to_callback():
    module = _load_training_module()
    assert module is not None
    assert hasattr(module, "run_admission_training")

    dataset, split = _build_temporal_training_fixture()
    messages: list[str] = []

    module.run_admission_training(
        dataset,
        split,
        numeric_features=["NU_IDADE_N"],
        categorical_features=[],
        progress=messages.append,
    )

    assert any("PREPROCESS" in message for message in messages)
    assert any("[TRAIN] logistic_regression" in message for message in messages)
    assert any("[TRAIN] hist_gradient_boosting" in message for message in messages)
    assert any("[SELECT] melhor modelo" in message for message in messages)
    assert any("[THRESHOLD]" in message for message in messages)
    assert any("[TEST]" in message for message in messages)


