import srag_api.ml as ml

from srag_api.ml.artifacts import ArtifactPaths, save_training_artifacts
from srag_api.ml.dataset import AdmissionDataset
from srag_api.ml.features import ADMISSION_FEATURES, LEAKAGE_FEATURES
from srag_api.ml.metrics import BinaryMetrics, evaluate_binary_predictions
from srag_api.ml.models import build_models
from srag_api.ml.split import TemporalSplit
from srag_api.ml.threshold import ThresholdSelection, select_decision_threshold
from srag_api.ml.training import (
    TrainingRunResult,
    run_admission_training,
    select_best_candidate,
    train_candidate_model,
)


def test_ml_package_exposes_primary_training_api():
    assert ml.AdmissionDataset is AdmissionDataset
    assert ml.TemporalSplit is TemporalSplit
    assert ml.build_models is build_models
    assert ml.train_candidate_model is train_candidate_model
    assert ml.select_best_candidate is select_best_candidate
    assert ml.run_admission_training is run_admission_training
    assert ml.TrainingRunResult is TrainingRunResult


def test_ml_package_exposes_evaluation_and_threshold_api():
    assert ml.BinaryMetrics is BinaryMetrics
    assert ml.evaluate_binary_predictions is evaluate_binary_predictions
    assert ml.ThresholdSelection is ThresholdSelection
    assert ml.select_decision_threshold is select_decision_threshold


def test_ml_package_exposes_artifact_api():
    assert ml.ArtifactPaths is ArtifactPaths
    assert ml.save_training_artifacts is save_training_artifacts


def test_ml_package_exposes_feature_contract():
    assert ml.ADMISSION_FEATURES is ADMISSION_FEATURES
    assert ml.LEAKAGE_FEATURES is LEAKAGE_FEATURES


def test_public_api_declares_explicit_all():
    expected = {
        "AdmissionDataset",
        "TemporalSplit",
        "ADMISSION_FEATURES",
        "LEAKAGE_FEATURES",
        "build_models",
        "BinaryMetrics",
        "evaluate_binary_predictions",
        "ThresholdSelection",
        "select_decision_threshold",
        "train_candidate_model",
        "select_best_candidate",
        "TrainingRunResult",
        "run_admission_training",
        "ArtifactPaths",
        "save_training_artifacts",
    }

    assert expected.issubset(set(ml.__all__))
    assert "build_admission_dataset" in ml.__all__
    assert "temporal_split" in ml.__all__
