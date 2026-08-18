"""Ferramentas de Machine Learning para análise de SRAG."""

from .dataset import AdmissionDataset, build_admission_dataset
from .split import TemporalSplit, temporal_split

__all__ = [
    "AdmissionDataset",
    "TemporalSplit",
    "build_admission_dataset",
    "temporal_split",
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
]

from .artifacts import ArtifactPaths, save_training_artifacts
from .features import ADMISSION_FEATURES, LEAKAGE_FEATURES
from .metrics import BinaryMetrics, evaluate_binary_predictions
from .models import build_models
from .threshold import ThresholdSelection, select_decision_threshold
from .training import (
    TrainingRunResult,
    run_admission_training,
    select_best_candidate,
    train_candidate_model,
)
