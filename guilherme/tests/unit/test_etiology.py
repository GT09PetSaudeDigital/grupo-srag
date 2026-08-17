import pandas as pd

from srag_api.data.etiology import (
    add_etiology_columns,
    normalize_detailed_etiology,
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


def test_detailed_etiology_sars_cov_2():
    assert normalize_detailed_etiology(
        make_row(PCR_SARS2=1)
    ) == "SARS-CoV-2"


def test_detailed_etiology_influenza_a():
    assert normalize_detailed_etiology(
        make_row(PCR_FLUAS=1)
    ) == "Influenza A"


def test_detailed_etiology_influenza_b():
    assert normalize_detailed_etiology(
        make_row(PCR_FLUBS=1)
    ) == "Influenza B"


def test_detailed_etiology_vsr():
    assert normalize_detailed_etiology(
        make_row(PCR_VSR=1)
    ) == "VSR"


def test_detailed_etiology_specific_other_viruses():
    assert normalize_detailed_etiology(make_row(PCR_ADENO=1)) == "Adenovirus"
    assert normalize_detailed_etiology(make_row(PCR_PARA1=1)) == "Parainfluenza 1"
    assert normalize_detailed_etiology(make_row(PCR_PARA2=1)) == "Parainfluenza 2"
    assert normalize_detailed_etiology(make_row(PCR_PARA3=1)) == "Parainfluenza 3"
    assert normalize_detailed_etiology(make_row(PCR_PARA4=1)) == "Parainfluenza 4"
    assert normalize_detailed_etiology(make_row(PCR_METAP=1)) == "Metapneumovirus"
    assert normalize_detailed_etiology(make_row(PCR_BOCA=1)) == "Bocavirus"
    assert normalize_detailed_etiology(make_row(PCR_RINO=1)) == "Rinovirus"


def test_missing_pcr_column_is_not_negative():
    assert normalize_detailed_etiology(
        make_row(CLASSI_FIN=1)
    ) == "NAO_IDENTIFICADA"


def test_missing_pcr_value_is_not_positive():
    assert normalize_detailed_etiology(
        make_row(PCR_SARS2=None, PCR_FLUAS=None)
    ) == "NAO_IDENTIFICADA"


def test_final_classification_does_not_override_lab_result():
    row = make_row(CLASSI_FIN=4, PCR_SARS2=1)

    assert normalize_final_classification(
        row["CLASSI_FIN"]
    ) == "NAO_ESPECIFICADO"

    assert normalize_detailed_etiology(row) == "SARS-CoV-2"


def test_add_etiology_columns_preserves_source_columns():
    df = pd.DataFrame(
        {
            "CLASSI_FIN": [4, 2],
            "PCR_SARS2": [1, None],
            "PCR_VSR": [None, 1],
        }
    )

    result = add_etiology_columns(df)

    assert result["CLASSI_FIN"].tolist() == [4, 2]
    assert result["PCR_SARS2"].iloc[0] == 1
    assert result["CLASSIFICACAO_FINAL_NORMALIZADA"].tolist() == [
        "NAO_ESPECIFICADO",
        "OUTRO_VIRUS_RESPIRATORIO",
    ]
    assert result["ETIOLOGIA_DETALHADA"].tolist() == [
        "SARS-CoV-2",
        "VSR",
    ]
