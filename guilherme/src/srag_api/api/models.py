from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MetricResponse(BaseModel):
    valor: float | None
    numerador: int
    denominador: int
    ignorados: int


class CasesResponse(BaseModel):
    valor: int
    filtros: dict[str, Any]


class DeathsResponse(BaseModel):
    obitos: int
    letalidade: MetricResponse


class IcuResponse(BaseModel):
    casos_uti: int
    proporcao_uti: MetricResponse


class DataListResponse(BaseModel):
    dados: list[dict[str, Any]]


class TimeSeriesResponse(BaseModel):
    frequencia: str
    dados: list[dict[str, Any]]


class RankingResponse(BaseModel):
    nivel: str
    metrica: str
    dados: list[dict[str, Any]]


class ComparisonResponse(BaseModel):
    recorte_a: dict[str, Any]
    recorte_b: dict[str, Any]
    diferenca: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    service: str
