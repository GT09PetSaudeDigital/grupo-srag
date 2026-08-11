import pandas as pd

from srag_api.data.etiology import add_etiology_column, normalize_etiology

def make_row(**kwargs):
    return pd.Series(kwargs)

def test_covid_has_priority_when_final_classification_is_covid():
    assert normalize_etiology(make_row(CLASSI_FIN=5, PCR_FLUAS=1)) == "COVID-19"

def test_influenza_a():
    assert normalize_etiology(make_row(CLASSI_FIN=None, PCR_FLUAS=1)) == "Influenza A"

def test_influenza_b():
    assert normalize_etiology(make_row(CLASSI_FIN=None, PCR_FLUBS=1)) == "Influenza B"

def test_vsr():
    assert normalize_etiology(make_row(CLASSI_FIN=None, PCR_VSR=1)) == "VSR"

def test_unknown_is_not_identified():
    assert normalize_etiology(make_row(CLASSI_FIN=None)) == "Nao identificado"

def test_add_etiology_column_keeps_original_columns():
    df = pd.DataFrame({"CLASSI_FIN": [5, None], "PCR_FLUAS": [None, 1]})
    result = add_etiology_column(df)
    assert result["ETIOLOGIA_NORMALIZADA"].tolist() == ["COVID-19", "Influenza A"]
