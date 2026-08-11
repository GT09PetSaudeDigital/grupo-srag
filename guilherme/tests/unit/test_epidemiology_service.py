from types import SimpleNamespace

from srag_api.services.epidemiology import SragService


class FakeRepository:
    def get_total_cases(self, filters):
        return 100

    def get_deaths(self, filters):
        return 10

    def get_known_outcome_count(self, filters):
        return 80

    def get_unknown_outcome_count(self, filters):
        return 20

    def get_icu_cases(self, filters):
        return 16

    def get_known_icu_count(self, filters):
        return 64

    def get_unknown_icu_count(self, filters):
        return 36

    def get_age_distribution(self, filters):
        return [{"faixa_etaria": "60-74", "casos": 20}]

    def get_etiology_distribution(self, filters):
        return [{"etiologia": "COVID-19", "casos": 30}]

    def get_comorbidity_distribution(self, filters, column_map):
        assert "DIABETES" in column_map
        return [{"comorbidade": "DIABETES", "casos": 12}]

    def get_time_series(self, filters, frequency):
        if frequency == "month":
            return [{"ano": 2025, "mes": 1, "casos": 8}]
        return [{"ano": 2025, "semana_epidemiologica": 1, "casos": 8}]

    def get_ranking(self, filters, level, metric, limit):
        return [{"uf": "PR", "valor": 100}]


class ZeroDenominatorRepository(FakeRepository):
    def get_deaths(self, filters):
        return 0

    def get_known_outcome_count(self, filters):
        return 0

    def get_unknown_outcome_count(self, filters):
        return 100


class ComparisonRepository(FakeRepository):
    def get_total_cases(self, filters):
        return 150 if filters.uf == "PR" else 100

    def get_deaths(self, filters):
        return 15 if filters.uf == "PR" else 5

    def get_known_outcome_count(self, filters):
        return 100

    def get_unknown_outcome_count(self, filters):
        return 0


def filters(**kwargs):
    base = {
        "ano_inicio": None,
        "ano_fim": None,
        "uf": None,
        "municipio": None,
        "codigo_municipio": None,
        "sexo": None,
        "faixa_etaria": None,
        "etiologia": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_cases_returns_simple_count():
    service = SragService(FakeRepository())
    assert service.get_cases(filters()) == {"valor": 100}


def test_mortality_uses_only_known_outcomes():
    service = SragService(FakeRepository())
    assert service.get_mortality(filters()) == {
        "valor": 12.5,
        "numerador": 10,
        "denominador": 80,
        "ignorados": 20,
    }


def test_zero_denominator_returns_null_value():
    service = SragService(ZeroDenominatorRepository())
    result = service.get_mortality(filters())
    assert result == {
        "valor": None,
        "numerador": 0,
        "denominador": 0,
        "ignorados": 100,
    }


def test_deaths_combines_count_and_mortality():
    service = SragService(FakeRepository())
    result = service.get_deaths(filters())
    assert result["obitos"] == 10
    assert result["letalidade"]["valor"] == 12.5


def test_icu_proportion_uses_only_known_answers():
    service = SragService(FakeRepository())
    assert service.get_icu_proportion(filters()) == {
        "valor": 25.0,
        "numerador": 16,
        "denominador": 64,
        "ignorados": 36,
    }


def test_icu_combines_count_and_proportion():
    service = SragService(FakeRepository())
    result = service.get_icu(filters())
    assert result["casos_uti"] == 16
    assert result["proporcao_uti"]["valor"] == 25.0


def test_distributions_delegate_to_repository():
    service = SragService(FakeRepository())
    current = filters()
    assert service.get_age_distribution(current)["dados"][0]["casos"] == 20
    assert service.get_etiology_distribution(current)["dados"][0]["casos"] == 30
    assert service.get_comorbidity_distribution(current)["dados"][0]["casos"] == 12


def test_time_series_translates_public_frequency():
    service = SragService(FakeRepository())
    monthly = service.get_time_series(filters(), "mes")
    weekly = service.get_time_series(filters(), "semana")
    assert monthly["frequencia"] == "mes"
    assert monthly["dados"][0]["mes"] == 1
    assert weekly["frequencia"] == "semana"
    assert weekly["dados"][0]["semana_epidemiologica"] == 1


def test_ranking_keeps_metadata():
    service = SragService(FakeRepository())
    result = service.get_ranking(filters(), level="uf", metric="cases", limit=20)
    assert result == {
        "nivel": "uf",
        "metrica": "cases",
        "dados": [{"uf": "PR", "valor": 100}],
    }


def test_compare_returns_absolute_and_percentage_point_differences():
    service = SragService(ComparisonRepository())
    result = service.compare(filters(uf="PR"), filters(uf="MT"))
    assert result["recorte_a"] == {"casos": 150, "obitos": 15, "letalidade": 15.0}
    assert result["recorte_b"] == {"casos": 100, "obitos": 5, "letalidade": 5.0}
    assert result["diferenca"] == {
        "casos_absoluta": 50,
        "obitos_absoluta": 10,
        "letalidade_pontos_percentuais": 10.0,
    }


def test_time_series_rejects_unknown_frequency():
    service = SragService(FakeRepository())
    try:
        service.get_time_series(filters(), "dia")
    except ValueError as exc:
        assert "mes" in str(exc)
    else:
        raise AssertionError("frequencia invalida deveria gerar ValueError")
