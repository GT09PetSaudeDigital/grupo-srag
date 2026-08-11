from pathlib import Path

import pandas as pd
import pytest

from srag_api.data.ingest import transform_srag_dataframe, write_year_parquet
from srag_api.data.repository import SragFilters, SragRepository


def _raw_rows(year: int) -> pd.DataFrame:
    if year == 2024:
        return pd.DataFrame(
            [
                {
                    "TP_IDADE": 3,
                    "NU_IDADE_N": 70,
                    "SG_UF": "PR",
                    "ID_MUNICIP": "CURITIBA",
                    "CO_MUN_RES": 410690,
                    "EVOLUCAO": 2,
                    "UTI": 1,
                    "CS_SEXO": "M",
                    "CLASSI_FIN": 5,
                    "DT_SIN_PRI": "10/01/2024",
                    "SEM_PRI": 2,
                    "CARDIOPATI": 1,
                    "DIABETES": 2,
                    "OBESIDADE": 9,
                },
                {
                    "TP_IDADE": 3,
                    "NU_IDADE_N": 40,
                    "SG_UF": "PR",
                    "ID_MUNICIP": "LONDRINA",
                    "CO_MUN_RES": 411370,
                    "EVOLUCAO": 1,
                    "UTI": 2,
                    "CS_SEXO": "F",
                    "PCR_FLUAS": 1,
                    "DT_SIN_PRI": "20/02/2024",
                    "SEM_PRI": 8,
                    "CARDIOPATI": 2,
                    "DIABETES": 1,
                    "OBESIDADE": 2,
                },
            ]
        )

    return pd.DataFrame(
        [
            {
                "TP_IDADE": 3,
                "NU_IDADE_N": 67,
                "SG_UF": "PR",
                "ID_MUNICIP": "CURITIBA",
                "CO_MUN_RES": 410690,
                "EVOLUCAO": 1,
                "UTI": 2,
                "CS_SEXO": "M",
                "PCR_FLUAS": 1,
                "DT_SIN_PRI": "15/03/2025",
                "SEM_PRI": 11,
                "CARDIOPATI": 1,
                "DIABETES": 1,
                "OBESIDADE": 2,
            },
            {
                "TP_IDADE": 2,
                "NU_IDADE_N": 18,
                "SG_UF": "PR",
                "ID_MUNICIP": "LONDRINA",
                "CO_MUN_RES": 411370,
                "EVOLUCAO": 2,
                "UTI": 1,
                "CS_SEXO": "F",
                "CLASSI_FIN": 5,
                "DT_SIN_PRI": "02/04/2025",
                "SEM_PRI": 14,
                "CARDIOPATI": 2,
                "DIABETES": 2,
                "OBESIDADE": 1,
            },
            {
                "TP_IDADE": 3,
                "NU_IDADE_N": 55,
                "SG_UF": "MT",
                "ID_MUNICIP": "PRIMAVERA DO LESTE",
                "CO_MUN_RES": 510704,
                "EVOLUCAO": 9,
                "UTI": 9,
                "CS_SEXO": "M",
                "PCR_VSR": 1,
                "DT_SIN_PRI": "10/05/2025",
                "SEM_PRI": 19,
                "CARDIOPATI": 1,
                "DIABETES": 2,
                "OBESIDADE": 9,
            },
        ]
    )


@pytest.fixture
def repository(tmp_path: Path) -> SragRepository:
    parquet_root = tmp_path / "parquet"

    for year in (2024, 2025):
        transformed = transform_srag_dataframe(_raw_rows(year), year)
        write_year_parquet(transformed, parquet_root, year)

    return SragRepository(parquet_root)


def test_total_cases_across_years(repository):
    assert repository.get_total_cases() == 5


def test_filters_by_year_and_state(repository):
    filters = SragFilters(ano_inicio=2025, ano_fim=2025, uf="PR")
    assert repository.get_total_cases(filters) == 2


def test_municipality_filter_requires_disambiguation(repository):
    with pytest.raises(ValueError, match="exige UF ou codigo_municipio"):
        repository.get_total_cases(SragFilters(municipio="Curitiba"))


def test_filters_municipality_with_state(repository):
    filters = SragFilters(uf="PR", municipio="Curitiba")
    assert repository.get_total_cases(filters) == 2


def test_death_denominator_counts_only_known_outcomes(repository):
    assert repository.get_deaths() == 2
    assert repository.get_known_outcome_count() == 4
    assert repository.get_unknown_outcome_count() == 1


def test_icu_denominator_counts_only_known_answers(repository):
    assert repository.get_icu_cases() == 2
    assert repository.get_known_icu_count() == 4
    assert repository.get_unknown_icu_count() == 1


def test_age_distribution(repository):
    result = repository.get_age_distribution(SragFilters(uf="PR"))
    assert sum(item["casos"] for item in result) == 4


def test_etiology_distribution(repository):
    result = repository.get_etiology_distribution()
    counts = {item["etiologia"]: item["casos"] for item in result}

    assert counts["COVID-19"] == 2
    assert counts["Influenza A"] == 2
    assert counts["VSR"] == 1


def test_monthly_time_series(repository):
    result = repository.get_time_series(
        SragFilters(ano_inicio=2025, ano_fim=2025),
        frequency="month",
    )

    assert [(row["mes"], row["casos"]) for row in result] == [
        (3, 1),
        (4, 1),
        (5, 1),
    ]


def test_weekly_time_series(repository):
    result = repository.get_time_series(
        SragFilters(uf="PR"),
        frequency="week",
    )

    assert sum(row["casos"] for row in result) == 4


def test_ranking_states_by_cases(repository):
    result = repository.get_ranking(metric="cases", level="uf")
    assert result[0] == {"uf": "PR", "valor": 4}


def test_ranking_municipalities_by_deaths(repository):
    result = repository.get_ranking(
        filters=SragFilters(uf="PR"),
        metric="deaths",
        level="municipio",
    )

    assert result[0]["valor"] == 1
    assert result[0]["municipio"] in {"CURITIBA", "LONDRINA"}


def test_ranking_rejects_invalid_limit(repository):
    with pytest.raises(ValueError, match="1 e 100"):
        repository.get_ranking(limit=101)


def test_available_columns(repository):
    columns = repository.get_available_columns()
    assert {"CARDIOPATI", "DIABETES", "OBESIDADE"}.issubset(columns)


def test_comorbidity_distribution_counts_only_yes(repository):
    result = repository.get_comorbidity_distribution(
        SragFilters(uf="PR"),
        {
            "CARDIOPATI": "CARDIOPATIA",
            "DIABETES": "DIABETES",
            "OBESIDADE": "OBESIDADE",
        },
    )
    counts = {item["comorbidade"]: item["casos"] for item in result}
    assert counts["CARDIOPATIA"] == 2
    assert counts["DIABETES"] == 2
    assert counts["OBESIDADE"] == 1
