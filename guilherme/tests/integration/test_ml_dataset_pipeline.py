import importlib

import pandas as pd


def _load_ml_package():
    return importlib.import_module("srag_api.ml")


def test_ml_dataset_pipeline_builds_target_without_leakage_and_splits_time():
    ml = _load_ml_package()

    assert hasattr(ml, "build_admission_dataset"), (
        "build_admission_dataset ainda nao foi exportado por srag_api.ml"
    )
    assert hasattr(ml, "temporal_split"), (
        "temporal_split ainda nao foi exportado por srag_api.ml"
    )

    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": [
                "CURA",
                "OBITO_SRAG",
                "CURA",
                "OBITO_SRAG",
                "OBITO_OUTRAS_CAUSAS",
            ],
            "ANO": [2023, 2024, 2025, 2026, 2026],
            "NU_IDADE_N": [20, 70, 35, 80, 50],
            "CS_SEXO": ["F", "M", "F", "M", "M"],
            "DISPNEIA": [2, 1, 2, 1, 1],
            "UTI": [2, 1, 2, 1, 1],
        }
    )

    dataset = ml.build_admission_dataset(df)
    split = ml.temporal_split(dataset.metadata["ANO"])

    assert dataset.y.tolist() == [0, 1, 0, 1]
    assert "UTI" not in dataset.X.columns
    assert dataset.metadata["ANO"].iloc[split.train_idx].tolist() == [2023, 2024]
    assert dataset.metadata["ANO"].iloc[split.validation_idx].tolist() == [2025]
    assert dataset.metadata["ANO"].iloc[split.test_idx].tolist() == [2026]
