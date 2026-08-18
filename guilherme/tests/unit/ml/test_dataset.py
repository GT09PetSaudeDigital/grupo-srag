import importlib

import pandas as pd


def _load_dataset_module():
    try:
        return importlib.import_module("srag_api.ml.dataset")
    except ModuleNotFoundError:
        return None


def test_dataset_keeps_only_eligible_outcomes():
    dataset_module = _load_dataset_module()

    assert dataset_module is not None, "srag_api.ml.dataset ainda nao foi implementado"

    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": [
                "CURA",
                "OBITO_SRAG",
                "OBITO_OUTRAS_CAUSAS",
                "AUSENTE",
            ],
            "NU_IDADE_N": [20, 70, 50, 40],
            "CS_SEXO": ["F", "M", "M", "F"],
            "ANO": [2024, 2024, 2024, 2024],
        }
    )

    result = dataset_module.build_admission_dataset(df)

    assert result.y.tolist() == [0, 1]
    assert len(result.X) == 2
    assert result.X.index.tolist() == [0, 1]


def test_missing_optional_feature_is_not_fabricated():
    dataset_module = _load_dataset_module()

    assert dataset_module is not None, "srag_api.ml.dataset ainda nao foi implementado"

    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": ["CURA"],
            "NU_IDADE_N": [30],
            "ANO": [2024],
        }
    )

    result = dataset_module.build_admission_dataset(df)

    assert "CARDIOPATI" not in result.X.columns
    assert "CS_SEXO" not in result.X.columns


def test_dataset_never_exports_leakage_columns():
    dataset_module = _load_dataset_module()

    assert dataset_module is not None, "srag_api.ml.dataset ainda nao foi implementado"

    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": ["CURA"],
            "NU_IDADE_N": [30],
            "UTI": [1],
            "SUPORT_VEN": [2],
            "DT_EVOLUCA": ["2024-02-01"],
            "ANO": [2024],
        }
    )

    result = dataset_module.build_admission_dataset(df)

    assert "UTI" not in result.X.columns
    assert "SUPORT_VEN" not in result.X.columns
    assert "DT_EVOLUCA" not in result.X.columns


def test_metadata_is_kept_outside_feature_matrix():
    dataset_module = _load_dataset_module()

    assert dataset_module is not None, "srag_api.ml.dataset ainda nao foi implementado"

    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": ["CURA"],
            "NU_IDADE_N": [30],
            "ANO": [2024],
            "DT_NOTIFIC": ["2024-01-10"],
        }
    )

    result = dataset_module.build_admission_dataset(df)

    assert result.metadata["ANO"].tolist() == [2024]
    assert result.metadata["DT_NOTIFIC"].tolist() == ["2024-01-10"]
    assert "ANO" not in result.X.columns
    assert "DT_NOTIFIC" not in result.X.columns
