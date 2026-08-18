from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform

import joblib
import pandas as pd
import sklearn

from .training import TrainingRunResult


@dataclass(frozen=True)
class ArtifactPaths:
    best_model: Path
    metrics_json: Path
    metrics_csv: Path
    validation_comparison: Path
    confusion_matrix_validation: Path
    confusion_matrix_test: Path
    run_metadata: Path


def _metrics_to_dict(metrics) -> dict[str, object]:
    return {
        "auc_pr": metrics.auc_pr,
        "roc_auc": metrics.roc_auc,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "threshold": metrics.threshold,
        "confusion_matrix": metrics.confusion_matrix.tolist(),
    }


def _write_confusion_matrix(path: Path, matrix) -> None:
    frame = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    frame.to_csv(path, index=True)


def save_training_artifacts(
    result: TrainingRunResult,
    *,
    output_dir: str | Path,
    metadata: dict[str, object],
) -> ArtifactPaths:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = ArtifactPaths(
        best_model=output / "best_model.joblib",
        metrics_json=output / "metrics.json",
        metrics_csv=output / "metrics.csv",
        validation_comparison=output / "validation_comparison.csv",
        confusion_matrix_validation=output / "confusion_matrix_validation.csv",
        confusion_matrix_test=output / "confusion_matrix_test.csv",
        run_metadata=output / "run_metadata.json",
    )

    joblib.dump(
        {
            "pipeline": result.best_pipeline,
            "threshold": result.threshold,
            "features": list(metadata.get("features_used", [])),
            "best_model": result.best_model_name,
        },
        paths.best_model,
    )

    metrics_payload = {
        "selection_metric": "average_precision",
        "best_model": result.best_model_name,
        "threshold": result.threshold,
        "threshold_policy": result.threshold_policy,
        "validation": _metrics_to_dict(result.validation_metrics),
        "test": _metrics_to_dict(result.test_metrics),
    }
    paths.metrics_json.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_rows = [
        {
            "partition": "validation",
            **{
                key: value
                for key, value in _metrics_to_dict(result.validation_metrics).items()
                if key != "confusion_matrix"
            },
        },
        {
            "partition": "test",
            **{
                key: value
                for key, value in _metrics_to_dict(result.test_metrics).items()
                if key != "confusion_matrix"
            },
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(paths.metrics_csv, index=False)

    comparison_rows = []
    for name, candidate in result.candidates.items():
        comparison_rows.append(
            {
                "model": name,
                **{
                    key: value
                    for key, value in _metrics_to_dict(
                        candidate.validation_metrics
                    ).items()
                    if key != "confusion_matrix"
                },
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        paths.validation_comparison,
        index=False,
    )

    _write_confusion_matrix(
        paths.confusion_matrix_validation,
        result.validation_metrics.confusion_matrix,
    )
    _write_confusion_matrix(
        paths.confusion_matrix_test,
        result.test_metrics.confusion_matrix,
    )

    run_metadata = dict(metadata)
    run_metadata.update(
        {
            "best_model": result.best_model_name,
            "threshold": result.threshold,
            "threshold_policy": result.threshold_policy,
            "train_size": result.train_size,
            "validation_size": result.validation_size,
            "test_size": result.test_size,
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        }
    )
    paths.run_metadata.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return paths
