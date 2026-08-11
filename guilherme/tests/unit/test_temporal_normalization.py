import pandas as pd

from srag_api.data.clean import add_temporal_columns


def test_temporal_columns_use_onset_date_and_sem_pri():
    df = pd.DataFrame(
        {
            "DT_SIN_PRI": ["15/03/2025", "31/12/2025"],
            "SEM_PRI": [11, 53],
        }
    )

    result = add_temporal_columns(df)

    assert result["MES"].tolist() == [3, 12]
    assert result["SEMANA_EPIDEMIOLOGICA"].tolist() == [11, 53]
    assert str(result.loc[0, "DATA_INICIO_SINTOMAS"].date()) == "2025-03-15"


def test_temporal_columns_do_not_invent_week_when_sem_pri_is_absent():
    df = pd.DataFrame({"DT_SIN_PRI": ["15/03/2025"]})

    result = add_temporal_columns(df)

    assert result["MES"].tolist() == [3]
    assert result["SEMANA_EPIDEMIOLOGICA"].isna().all()


def test_temporal_columns_tolerate_missing_onset_date():
    df = pd.DataFrame({"SEM_PRI": [12]})

    result = add_temporal_columns(df)

    assert result["MES"].isna().all()
    assert result["SEMANA_EPIDEMIOLOGICA"].tolist() == [12]
