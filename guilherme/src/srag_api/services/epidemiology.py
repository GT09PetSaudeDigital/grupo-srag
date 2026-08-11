from __future__ import annotations

from typing import Any, Literal, Protocol


COMORBIDITY_COLUMNS: dict[str, str] = {
    "CARDIOPATI": "CARDIOPATIA",
    "DIABETES": "DIABETES",
    "OBESIDADE": "OBESIDADE",
    "RENAL": "DOENCA_RENAL",
    "HEPATICA": "DOENCA_HEPATICA",
    "IMUNODEPRE": "IMUNODEPRESSAO",
    "ASMA": "ASMA",
    "PNEUMOPATI": "PNEUMOPATIA",
    "NEUROLOGIC": "DOENCA_NEUROLOGICA",
    "HEMATOLOGI": "DOENCA_HEMATOLOGICA",
}


class RepositoryProtocol(Protocol):
    def get_total_cases(self, filters: object) -> int: ...
    def get_deaths(self, filters: object) -> int: ...
    def get_known_outcome_count(self, filters: object) -> int: ...
    def get_unknown_outcome_count(self, filters: object) -> int: ...
    def get_icu_cases(self, filters: object) -> int: ...
    def get_known_icu_count(self, filters: object) -> int: ...
    def get_unknown_icu_count(self, filters: object) -> int: ...
    def get_age_distribution(self, filters: object) -> list[dict[str, object]]: ...
    def get_etiology_distribution(self, filters: object) -> list[dict[str, object]]: ...
    def get_comorbidity_distribution(
        self,
        filters: object,
        column_map: dict[str, str],
    ) -> list[dict[str, object]]: ...
    def get_time_series(
        self,
        filters: object,
        frequency: Literal["month", "week"],
    ) -> list[dict[str, object]]: ...
    def get_ranking(
        self,
        filters: object,
        level: Literal["uf", "municipio"],
        metric: Literal["cases", "deaths", "icu"],
        limit: int,
    ) -> list[dict[str, object]]: ...


class SragService:
    def __init__(self, repository: RepositoryProtocol):
        self.repository = repository

    @staticmethod
    def _percentage(
        numerator: int,
        denominator: int,
        ignored: int,
    ) -> dict[str, int | float | None]:
        value: float | None = None
        if denominator > 0:
            value = round((numerator / denominator) * 100, 2)

        return {
            "valor": value,
            "numerador": numerator,
            "denominador": denominator,
            "ignorados": ignored,
        }

    def get_cases(self, filters: object) -> dict[str, int]:
        return {"valor": self.repository.get_total_cases(filters)}

    def get_mortality(self, filters: object) -> dict[str, int | float | None]:
        return self._percentage(
            self.repository.get_deaths(filters),
            self.repository.get_known_outcome_count(filters),
            self.repository.get_unknown_outcome_count(filters),
        )

    def get_deaths(self, filters: object) -> dict[str, Any]:
        deaths = self.repository.get_deaths(filters)
        return {
            "obitos": deaths,
            "letalidade": self.get_mortality(filters),
        }

    def get_icu_proportion(self, filters: object) -> dict[str, int | float | None]:
        return self._percentage(
            self.repository.get_icu_cases(filters),
            self.repository.get_known_icu_count(filters),
            self.repository.get_unknown_icu_count(filters),
        )

    def get_icu(self, filters: object) -> dict[str, Any]:
        cases = self.repository.get_icu_cases(filters)
        return {
            "casos_uti": cases,
            "proporcao_uti": self.get_icu_proportion(filters),
        }

    def get_age_distribution(self, filters: object) -> dict[str, list[dict[str, object]]]:
        return {"dados": self.repository.get_age_distribution(filters)}

    def get_etiology_distribution(self, filters: object) -> dict[str, list[dict[str, object]]]:
        return {"dados": self.repository.get_etiology_distribution(filters)}

    def get_comorbidity_distribution(self, filters: object) -> dict[str, list[dict[str, object]]]:
        return {
            "dados": self.repository.get_comorbidity_distribution(
                filters,
                COMORBIDITY_COLUMNS,
            )
        }

    def get_time_series(
        self,
        filters: object,
        frequency: Literal["mes", "semana"],
    ) -> dict[str, object]:
        if frequency not in {"mes", "semana"}:
            raise ValueError("frequency deve ser 'mes' ou 'semana'.")

        repository_frequency: Literal["month", "week"] = (
            "month" if frequency == "mes" else "week"
        )
        return {
            "frequencia": frequency,
            "dados": self.repository.get_time_series(
                filters,
                repository_frequency,
            ),
        }

    def get_ranking(
        self,
        filters: object,
        level: Literal["uf", "municipio"],
        metric: Literal["cases", "deaths", "icu"],
        limit: int = 20,
    ) -> dict[str, object]:
        return {
            "nivel": level,
            "metrica": metric,
            "dados": self.repository.get_ranking(
                filters,
                level=level,
                metric=metric,
                limit=limit,
            ),
        }

    def compare(self, filters_a: object, filters_b: object) -> dict[str, object]:
        cases_a = self.repository.get_total_cases(filters_a)
        cases_b = self.repository.get_total_cases(filters_b)
        deaths_a = self.repository.get_deaths(filters_a)
        deaths_b = self.repository.get_deaths(filters_b)
        mortality_a = self.get_mortality(filters_a)["valor"]
        mortality_b = self.get_mortality(filters_b)["valor"]

        mortality_difference: float | None = None
        if mortality_a is not None and mortality_b is not None:
            mortality_difference = round(float(mortality_a) - float(mortality_b), 2)

        return {
            "recorte_a": {
                "casos": cases_a,
                "obitos": deaths_a,
                "letalidade": mortality_a,
            },
            "recorte_b": {
                "casos": cases_b,
                "obitos": deaths_b,
                "letalidade": mortality_b,
            },
            "diferenca": {
                "casos_absoluta": cases_a - cases_b,
                "obitos_absoluta": deaths_a - deaths_b,
                "letalidade_pontos_percentuais": mortality_difference,
            },
        }
