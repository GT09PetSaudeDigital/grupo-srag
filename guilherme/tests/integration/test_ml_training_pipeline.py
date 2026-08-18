from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from srag_api.ml import (
    ADMISSION_FEATURES,
    build_admission_dataset,
    run_admission_training,
    save_training_artifacts,
    temporal_split,
)


def _synthetic_normalized_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ANO": [2023, 2023, 2024, 2024, 2025, 2025, 2026, 2026],
            "DESFECHO_NORMALIZADO": [
                "CURA",
                "OBITO_SRAG",
                "CURA",
                "OBITO_SRAG",
                "CURA",
                "OBITO_SRAG",
                "CURA",
                "OBITO_SRAG",
            ],
            "NU_IDADE_N": [22, 78, 35, 69, 40, 73, 44, 76],
            "SINT_ATE_NOTIF": [2, 8, 3, 7, 2, 9, 4, 8],
            "CS_SEXO": ["F", "M", "F", "M", "F", "M", "F", "M"],
            "FEBRE": [1, 1, 1, 2, 1, 1, 2, 1],
        }
    )


def test_ml_training_pipeline_runs_end_to_end_on_synthetic_data(tmp_path):
    df = _synthetic_normalized_data()

    dataset = build_admission_dataset(df)
    split = temporal_split(dataset.metadata["ANO"])

    result = run_admission_training(dataset, split)

    metadata = {
        "features_used": list(dataset.X.columns),
        "features_missing": sorted(set(ADMISSION_FEATURES) - set(dataset.X.columns)),
        "train_years": [2019, 2020, 2021, 2022, 2023, 2024],
        "validation_year": 2025,
        "test_year": 2026,
        "random_state": 42,
    }
    paths = save_training_artifacts(
        result,
        output_dir=tmp_path / "artifacts",
        metadata=metadata,
    )

    assert len(result.candidates) == 4
    assert result.best_model_name in result.candidates
    assert 0.0 <= result.threshold <= 1.0
    assert result.validation_metrics is not None
    assert result.test_metrics is not None

    assert paths.best_model.exists()
    assert paths.metrics_json.exists()
    assert paths.validation_comparison.exists()
    assert paths.run_metadata.exists()


def test_training_cli_exposes_required_arguments():
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "train_ml_admission.py"

    assert script.exists(), "scripts/train_ml_admission.py ainda nao foi implementado"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    help_text = completed.stdout
    assert "--parquet-glob" in help_text
    assert "--output-dir" in help_text
    assert "--validation-year" in help_text
    assert "--test-year" in help_text
