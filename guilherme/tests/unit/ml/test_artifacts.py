import importlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from srag_api.ml.metrics import BinaryMetrics
from srag_api.ml.training import TrainedCandidate, TrainingRunResult


def _load_artifacts_module():
    try:
        return importlib.import_module("srag_api.ml.artifacts")
    except ModuleNotFoundError:
        return None


def _metrics(threshold: float = 0.4) -> BinaryMetrics:
    return BinaryMetrics(
        auc_pr=0.75,
        roc_auc=0.80,
        recall=0.70,
        precision=0.60,
        f1=0.65,
        threshold=threshold,
        confusion_matrix=np.array([[8, 2], [3, 7]]),
    )


def _training_result() -> TrainingRunResult:
    candidates = {}
    for name, auc in [
        ("logistic_regression", 0.60),
        ("random_forest", 0.70),
        ("gradient_boosting", 0.68),
        ("hist_gradient_boosting", 0.75),
    ]:
        metrics = BinaryMetrics(
            auc_pr=auc,
            roc_auc=0.80,
            recall=0.70,
            precision=0.60,
            f1=0.65,
            threshold=0.5,
            confusion_matrix=np.array([[8, 2], [3, 7]]),
        )
        candidates[name] = TrainedCandidate(
            name=name,
            pipeline=Pipeline([]),
            validation_probabilities=np.array([0.2, 0.8]),
            validation_metrics=metrics,
        )

    return TrainingRunResult(
        candidates=candidates,
        best_model_name="hist_gradient_boosting",
        best_pipeline=Pipeline([]),
        threshold=0.4,
        threshold_policy="max_recall_precision_ge_0_50",
        validation_metrics=_metrics(0.4),
        test_metrics=_metrics(0.4),
        train_size=100,
        validation_size=20,
        test_size=20,
    )


def test_save_training_artifacts_creates_expected_files(tmp_path):
    module = _load_artifacts_module()
    assert module is not None, "srag_api.ml.artifacts ainda nao foi implementado"

    result = _training_result()
    paths = module.save_training_artifacts(
        result,
        output_dir=tmp_path,
        metadata={
            "features_used": ["NU_IDADE_N"],
            "features_missing": [],
            "train_years": [2019, 2020, 2021, 2022, 2023, 2024],
            "validation_year": 2025,
            "test_year": 2026,
        },
    )

    expected = [
        paths.best_model,
        paths.metrics_json,
        paths.metrics_csv,
        paths.validation_comparison,
        paths.confusion_matrix_validation,
        paths.confusion_matrix_test,
        paths.run_metadata,
    ]

    for path in expected:
        assert path.exists(), f"Artefato ausente: {path.name}"


def test_best_model_joblib_contains_pipeline_threshold_and_features(tmp_path):
    module = _load_artifacts_module()
    assert module is not None, "srag_api.ml.artifacts ainda nao foi implementado"

    paths = module.save_training_artifacts(
        _training_result(),
        output_dir=tmp_path,
        metadata={
            "features_used": ["NU_IDADE_N"],
            "features_missing": [],
            "train_years": [2019, 2020, 2021, 2022, 2023, 2024],
            "validation_year": 2025,
            "test_year": 2026,
        },
    )

    payload = joblib.load(paths.best_model)

    assert set(payload) >= {"pipeline", "threshold", "features"}
    assert payload["threshold"] == pytest.approx(0.4)
    assert payload["features"] == ["NU_IDADE_N"]


def test_metrics_json_records_selection_and_final_metrics(tmp_path):
    module = _load_artifacts_module()
    assert module is not None, "srag_api.ml.artifacts ainda nao foi implementado"

    paths = module.save_training_artifacts(
        _training_result(),
        output_dir=tmp_path,
        metadata={
            "features_used": ["NU_IDADE_N"],
            "features_missing": [],
            "train_years": [2019, 2020, 2021, 2022, 2023, 2024],
            "validation_year": 2025,
            "test_year": 2026,
        },
    )

    data = json.loads(paths.metrics_json.read_text(encoding="utf-8"))

    assert data["selection_metric"] == "average_precision"
    assert data["best_model"] == "hist_gradient_boosting"
    assert data["threshold"] == pytest.approx(0.4)
    assert data["threshold_policy"] == "max_recall_precision_ge_0_50"
    assert "validation" in data
    assert "test" in data


def test_validation_comparison_has_one_row_per_candidate(tmp_path):
    module = _load_artifacts_module()
    assert module is not None, "srag_api.ml.artifacts ainda nao foi implementado"

    paths = module.save_training_artifacts(
        _training_result(),
        output_dir=tmp_path,
        metadata={
            "features_used": ["NU_IDADE_N"],
            "features_missing": [],
            "train_years": [2019, 2020, 2021, 2022, 2023, 2024],
            "validation_year": 2025,
            "test_year": 2026,
        },
    )

    comparison = pd.read_csv(paths.validation_comparison)

    assert len(comparison) == 4
    assert set(comparison["model"]) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
    }


def test_run_metadata_contains_reproducibility_fields(tmp_path):
    module = _load_artifacts_module()
    assert module is not None, "srag_api.ml.artifacts ainda nao foi implementado"

    paths = module.save_training_artifacts(
        _training_result(),
        output_dir=tmp_path,
        metadata={
            "features_used": ["NU_IDADE_N"],
            "features_missing": ["CS_SEXO"],
            "train_years": [2019, 2020, 2021, 2022, 2023, 2024],
            "validation_year": 2025,
            "test_year": 2026,
            "random_state": 42,
        },
    )

    data = json.loads(paths.run_metadata.read_text(encoding="utf-8"))

    assert data["random_state"] == 42
    assert data["validation_year"] == 2025
    assert data["test_year"] == 2026
    assert data["features_used"] == ["NU_IDADE_N"]
    assert data["features_missing"] == ["CS_SEXO"]
    assert data["best_model"] == "hist_gradient_boosting"
    assert "python_version" in data
    assert "pandas_version" in data
    assert "scikit_learn_version" in data
