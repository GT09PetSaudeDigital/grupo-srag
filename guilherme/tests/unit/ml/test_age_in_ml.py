"""A idade que chega ao modelo precisa ter unidade.

``NU_IDADE_N`` e um numero sem significado sozinho: o SIVEP grava a unidade
em ``TP_IDADE`` (1 dia, 2 meses, 3 anos). A ingestao ja normaliza para
``IDADE_ANOS``. O contrato do ML usa essa coluna, e estes testes impedem
que o numero cru volte por engano.
"""

import importlib

import pandas as pd
import pytest

from srag_api.data.clean import normalize_age


def _load_features_module():
    try:
        return importlib.import_module("srag_api.ml.features")
    except ModuleNotFoundError:
        return None


def _load_dataset_module():
    try:
        return importlib.import_module("srag_api.ml.dataset")
    except ModuleNotFoundError:
        return None


def _load_training_module():
    try:
        return importlib.import_module("srag_api.ml.training")
    except ModuleNotFoundError:
        return None


# --------------------------------------------------------------------------
# Conversao, que ja existe na ingestao e precisa continuar correta
# --------------------------------------------------------------------------


def test_age_in_days_is_converted_to_years():
    assert normalize_age(1, 30) == pytest.approx(30 / 365.25)


def test_age_in_months_is_converted_to_years():
    assert normalize_age(2, 18) == pytest.approx(1.5)


def test_age_in_years_is_kept():
    assert normalize_age(3, 67) == pytest.approx(67.0)


def test_negative_age_is_discarded():
    assert normalize_age(3, -1) is None


def test_age_above_the_declared_limit_is_discarded():
    assert normalize_age(3, 121) is None


def test_unknown_age_unit_is_discarded():
    assert normalize_age(9, 40) is None


def test_missing_unit_is_discarded():
    assert normalize_age(None, 40) is None


# --------------------------------------------------------------------------
# O contrato do ML
# --------------------------------------------------------------------------


def test_age_in_years_is_the_only_age_column_in_the_ml_contract():
    features = _load_features_module()
    assert features is not None

    # Filtro por sufixo, e nao por substring: "OBESIDADE" e "IMUNODEPRE"
    # contem IDADE no nome e nao sao variaveis de idade.
    candidates = {
        c
        for c in features.ADMISSION_FEATURES
        if c in {"IDADE_ANOS", "NU_IDADE_N", "TP_IDADE", "IDADE"}
    }

    assert candidates == {"IDADE_ANOS"}


def test_audit_still_reads_the_raw_age_columns_it_verifies():
    """O audit precisa da coluna crua para conferir a conversao.

    ``NU_IDADE_N`` saiu do contrato do ML, mas continua sendo a origem que
    a auditoria precisa ler para conferir a normalizacao. Se ela deixar de
    ser lida, a auditoria passa a validar a conversao contra o proprio
    resultado e para de ter valor.
    """
    audit = importlib.import_module("scripts.audit_ml_admission")

    assert "NU_IDADE_N" in audit.EXTRA
    assert "TP_IDADE" in audit.EXTRA
    assert "IDADE_ANOS" in audit.EXTRA


def test_age_in_years_is_treated_as_numeric_by_the_preprocessor_split():
    features = _load_features_module()
    training = _load_training_module()
    assert features is not None
    assert training is not None

    X = pd.DataFrame(
        {
            "IDADE_ANOS": [20.0, 70.0],
            "CS_SEXO": ["F", "M"],
        }
    )

    numeric, categorical = training.split_preprocessing_features(X)

    assert numeric == ["IDADE_ANOS"]
    assert categorical == ["CS_SEXO"]


def test_dataset_keeps_age_in_years_and_drops_the_raw_column():
    dataset_module = _load_dataset_module()
    assert dataset_module is not None

    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": ["CURA", "OBITO_SRAG"],
            "TP_IDADE": [3, 1],
            "NU_IDADE_N": [67, 30],
            "IDADE_ANOS": [67.0, 30 / 365.25],
            "ANO": [2024, 2024],
        }
    )

    result = dataset_module.build_admission_dataset(df)

    assert "IDADE_ANOS" in result.X.columns
    assert "NU_IDADE_N" not in result.X.columns
    assert "TP_IDADE" not in result.X.columns
    assert result.X["IDADE_ANOS"].tolist() == [67.0, pytest.approx(30 / 365.25)]


def test_dataset_does_not_fabricate_age_in_years_when_absent():
    """Coluna ausente nao e inventada com valor negativo.

    O pipeline nao cria ``IDADE_ANOS`` quando a fonte nao a traz. Nesse
    caso a feature simplesmente nao entra, e a ausencia precisa ficar
    visivel em vez de virar 0.
    """
    dataset_module = _load_dataset_module()
    assert dataset_module is not None

    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": ["CURA"],
            "NU_IDADE_N": [67],
            "ANO": [2024],
        }
    )

    result = dataset_module.build_admission_dataset(df)

    assert "IDADE_ANOS" not in result.X.columns
    assert "NU_IDADE_N" not in result.X.columns
