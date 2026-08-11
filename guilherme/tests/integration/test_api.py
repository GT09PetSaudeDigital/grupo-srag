from pathlib import Path

from fastapi.testclient import TestClient

from srag_api.api.app import app
from srag_api.api.dependencies import get_service


class FakeService:
    def get_cases(self, filters):
        return {"valor": 100}

    def get_deaths(self, filters):
        return {
            "obitos": 10,
            "letalidade": {
                "valor": 12.5,
                "numerador": 10,
                "denominador": 80,
                "ignorados": 20,
            },
        }

    def get_icu(self, filters):
        return {
            "casos_uti": 16,
            "proporcao_uti": {
                "valor": 25.0,
                "numerador": 16,
                "denominador": 64,
                "ignorados": 36,
            },
        }

    def get_age_distribution(self, filters):
        return {"dados": [{"faixa_etaria": "60-74", "casos": 20}]}

    def get_etiology_distribution(self, filters):
        return {"dados": [{"etiologia": "COVID-19", "casos": 30}]}

    def get_comorbidity_distribution(self, filters):
        return {"dados": [{"comorbidade": "DIABETES", "casos": 12}]}

    def get_time_series(self, filters, frequency):
        key = "mes" if frequency == "mes" else "semana_epidemiologica"
        return {
            "frequencia": frequency,
            "dados": [{"ano": 2025, key: 1, "casos": 8}],
        }

    def get_ranking(self, filters, level, metric, limit=20):
        if level == "uf":
            dados = [{"uf": "PR", "valor": 100}]
        else:
            dados = [
                {
                    "uf": "PR",
                    "codigo_municipio": 410690,
                    "municipio": "CURITIBA",
                    "valor": 50,
                }
            ]
        return {"nivel": level, "metrica": metric, "dados": dados}

    def compare(self, filters_a, filters_b):
        return {
            "recorte_a": {"casos": 150, "obitos": 15, "letalidade": 15.0},
            "recorte_b": {"casos": 100, "obitos": 5, "letalidade": 5.0},
            "diferenca": {
                "casos_absoluta": 50,
                "obitos_absoluta": 10,
                "letalidade_pontos_percentuais": 10.0,
            },
        }


app.dependency_overrides[get_service] = lambda: FakeService()
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "srag-api"}


def test_cases_returns_filters_and_count():
    response = client.get(
        "/api/v1/casos",
        params={"ano_inicio": 2025, "uf": "pr"},
    )
    assert response.status_code == 200
    assert response.json()["valor"] == 100
    assert response.json()["filtros"] == {"ano_inicio": 2025, "uf": "PR"}


def test_deaths_endpoint():
    response = client.get("/api/v1/obitos", params={"uf": "PR"})
    assert response.status_code == 200
    assert response.json()["letalidade"]["denominador"] == 80


def test_icu_endpoint():
    response = client.get("/api/v1/uti", params={"uf": "PR"})
    assert response.status_code == 200
    assert response.json()["proporcao_uti"]["valor"] == 25.0


def test_age_bands_endpoint():
    response = client.get("/api/v1/faixas-etarias", params={"uf": "PR"})
    assert response.status_code == 200
    assert response.json()["dados"][0]["faixa_etaria"] == "60-74"


def test_etiology_endpoint():
    response = client.get("/api/v1/etiologia", params={"uf": "PR"})
    assert response.status_code == 200
    assert response.json()["dados"][0]["etiologia"] == "COVID-19"


def test_comorbidities_endpoint():
    response = client.get("/api/v1/comorbidades", params={"uf": "PR"})
    assert response.status_code == 200
    assert response.json()["dados"][0]["comorbidade"] == "DIABETES"


def test_monthly_time_series_endpoint():
    response = client.get(
        "/api/v1/serie-temporal",
        params={"frequencia": "mes", "uf": "PR"},
    )
    assert response.status_code == 200
    assert response.json()["frequencia"] == "mes"


def test_invalid_time_series_frequency_is_422():
    response = client.get(
        "/api/v1/serie-temporal",
        params={"frequencia": "dia"},
    )
    assert response.status_code == 422


def test_ranking_endpoint():
    response = client.get(
        "/api/v1/ranking",
        params={"nivel": "uf", "metrica": "cases"},
    )
    assert response.status_code == 200
    assert response.json()["dados"][0] == {"uf": "PR", "valor": 100}


def test_ranking_limit_above_100_is_422():
    response = client.get("/api/v1/ranking", params={"limit": 101})
    assert response.status_code == 422


def test_compare_endpoint():
    response = client.get(
        "/api/v1/comparar",
        params={"a_uf": "PR", "b_uf": "MT"},
    )
    assert response.status_code == 200
    assert response.json()["diferenca"]["casos_absoluta"] == 50


def test_municipality_without_state_or_code_is_400():
    response = client.get(
        "/api/v1/casos",
        params={"municipio": "CURITIBA"},
    )
    assert response.status_code == 400
    assert "exige UF ou codigo_municipio" in response.json()["detail"]


def test_compare_municipality_without_state_or_code_is_400():
    response = client.get(
        "/api/v1/comparar",
        params={"a_municipio": "CURITIBA", "b_uf": "MT"},
    )
    assert response.status_code == 400


def test_year_outside_supported_range_is_422():
    response = client.get("/api/v1/casos", params={"ano_inicio": 2018})
    assert response.status_code == 422


def test_inverted_year_range_is_400():
    response = client.get(
        "/api/v1/casos",
        params={"ano_inicio": 2025, "ano_fim": 2024},
    )
    assert response.status_code == 400
    assert "ano_inicio" in response.json()["detail"]


def test_openapi_contains_all_planned_paths():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    expected = {
        "/health",
        "/api/v1/casos",
        "/api/v1/obitos",
        "/api/v1/uti",
        "/api/v1/comorbidades",
        "/api/v1/faixas-etarias",
        "/api/v1/etiologia",
        "/api/v1/serie-temporal",
        "/api/v1/ranking",
        "/api/v1/comparar",
    }
    assert expected.issubset(paths)


def test_missing_dataset_is_404():
    class MissingDataService(FakeService):
        def get_cases(self, filters):
            raise FileNotFoundError("Nenhum Parquet SRAG encontrado")

    app.dependency_overrides[get_service] = lambda: MissingDataService()
    local_client = TestClient(app, raise_server_exceptions=False)
    try:
        response = local_client.get("/api/v1/casos")
    finally:
        app.dependency_overrides[get_service] = lambda: FakeService()

    assert response.status_code == 404
    assert "Nenhum Parquet" in response.json()["detail"]
