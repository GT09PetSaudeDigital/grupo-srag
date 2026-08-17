import pandas as pd

from srag_api.data.etiology import (
    add_etiology_column,
    normalize_etiology,
    normalize_final_classification,
)


def make_row(**kwargs):
    return pd.Series(kwargs)


def test_final_classification_influenza():
    assert normalize_final_classification(1) == "INFLUENZA"


def test_final_classification_other_respiratory_virus():
    assert normalize_final_classification(2) == "OUTRO_VIRUS_RESPIRATORIO"


def test_final_classification_other_agent():
    assert normalize_final_classification(3) == "OUTRO_AGENTE_ETIOLOGICO"


def test_final_classification_unspecified():
    assert normalize_final_classification(4) == "NAO_ESPECIFICADO"


def test_final_classification_covid():
    assert normalize_final_classification(5) == "COVID-19"


def test_final_classification_missing():
    assert normalize_final_classification(None) == "AUSENTE"
    assert normalize_final_classification(float("nan")) == "AUSENTE"


def test_final_classification_unexpected():
    assert normalize_final_classification(9) == "OUTRO"
    assert normalize_final_classification(99) == "OUTRO"


def test_influenza_a():
    assert normalize_etiology(
        make_row(CLASSI_FIN=None, PCR_FLUAS=1)
    ) == "Influenza A"


def test_influenza_b():
    assert normalize_etiology(
        make_row(CLASSI_FIN=None, PCR_FLUBS=1)
    ) == "Influenza B"


def test_vsr():
    assert normalize_etiology(
        make_row(CLASSI_FIN=None, PCR_VSR=1)
    ) == "VSR"


def test_unknown_is_not_identified():
    assert normalize_etiology(
        make_row(CLASSI_FIN=None)
    ) == "Nao identificado"


def test_add_etiology_column_keeps_original_columns():
    df = pd.DataFrame(
        {
            "CLASSI_FIN": [5, None],
            "PCR_FLUAS": [None, 1],
        }
    )

    result = add_etiology_column(df)

    assert result["ETIOLOGIA_NORMALIZADA"].tolist() == [
        "COVID-19",
        "Influenza A",
    ]