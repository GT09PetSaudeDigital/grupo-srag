from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from srag_api.api.dependencies import get_service
from srag_api.api.models import (
    CasesResponse,
    ComparisonResponse,
    DataListResponse,
    DeathsResponse,
    IcuResponse,
    RankingResponse,
    TimeSeriesResponse,
)
from srag_api.data.repository import SragFilters
from srag_api.services.epidemiology import SragService


router = APIRouter(prefix="/api/v1", tags=["epidemiologia"])

YearQuery = Annotated[int | None, Query(ge=2019, le=2026)]
RankingLimit = Annotated[int, Query(ge=1, le=100)]


def _clean(value: str | None, *, upper: bool = False) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned.upper() if upper else cleaned


def build_filters(
    *,
    ano_inicio: int | None = None,
    ano_fim: int | None = None,
    uf: str | None = None,
    municipio: str | None = None,
    codigo_municipio: int | None = None,
    sexo: str | None = None,
    faixa_etaria: str | None = None,
    etiologia: str | None = None,
) -> SragFilters:
    normalized_uf = _clean(uf, upper=True)
    normalized_municipio = _clean(municipio, upper=True)
    normalized_sexo = _clean(sexo, upper=True)
    normalized_faixa = _clean(faixa_etaria)
    normalized_etiologia = _clean(etiologia)

    if ano_inicio is not None and ano_fim is not None and ano_inicio > ano_fim:
        raise ValueError("ano_inicio nao pode ser maior que ano_fim.")

    if normalized_municipio and not normalized_uf and codigo_municipio is None:
        raise ValueError(
            "Filtro por municipio exige UF ou codigo_municipio."
        )

    return SragFilters(
        ano_inicio=ano_inicio,
        ano_fim=ano_fim,
        uf=normalized_uf,
        municipio=normalized_municipio,
        codigo_municipio=codigo_municipio,
        sexo=normalized_sexo,
        faixa_etaria=normalized_faixa,
        etiologia=normalized_etiologia,
    )


def common_filters(
    ano_inicio: YearQuery = None,
    ano_fim: YearQuery = None,
    uf: str | None = None,
    municipio: str | None = None,
    codigo_municipio: int | None = None,
    sexo: str | None = None,
    faixa_etaria: str | None = None,
    etiologia: str | None = None,
) -> SragFilters:
    return build_filters(
        ano_inicio=ano_inicio,
        ano_fim=ano_fim,
        uf=uf,
        municipio=municipio,
        codigo_municipio=codigo_municipio,
        sexo=sexo,
        faixa_etaria=faixa_etaria,
        etiologia=etiologia,
    )


def _filters_payload(filters: SragFilters) -> dict[str, object]:
    return {
        key: value
        for key, value in {
            "ano_inicio": filters.ano_inicio,
            "ano_fim": filters.ano_fim,
            "uf": filters.uf,
            "municipio": filters.municipio,
            "codigo_municipio": filters.codigo_municipio,
            "sexo": filters.sexo,
            "faixa_etaria": filters.faixa_etaria,
            "etiologia": filters.etiologia,
        }.items()
        if value is not None
    }


@router.get("/casos", response_model=CasesResponse)
def cases(
    filters: Annotated[SragFilters, Depends(common_filters)],
    service: Annotated[SragService, Depends(get_service)],
):
    return {
        **service.get_cases(filters),
        "filtros": _filters_payload(filters),
    }


@router.get("/obitos", response_model=DeathsResponse)
def deaths(
    filters: Annotated[SragFilters, Depends(common_filters)],
    service: Annotated[SragService, Depends(get_service)],
):
    return service.get_deaths(filters)


@router.get("/uti", response_model=IcuResponse)
def icu(
    filters: Annotated[SragFilters, Depends(common_filters)],
    service: Annotated[SragService, Depends(get_service)],
):
    return service.get_icu(filters)


@router.get("/faixas-etarias", response_model=DataListResponse)
def age_bands(
    filters: Annotated[SragFilters, Depends(common_filters)],
    service: Annotated[SragService, Depends(get_service)],
):
    return service.get_age_distribution(filters)


@router.get("/etiologia", response_model=DataListResponse)
def etiology(
    filters: Annotated[SragFilters, Depends(common_filters)],
    service: Annotated[SragService, Depends(get_service)],
):
    return service.get_etiology_distribution(filters)


@router.get("/comorbidades", response_model=DataListResponse)
def comorbidities(
    filters: Annotated[SragFilters, Depends(common_filters)],
    service: Annotated[SragService, Depends(get_service)],
):
    return service.get_comorbidity_distribution(filters)


@router.get("/serie-temporal", response_model=TimeSeriesResponse)
def time_series(
    filters: Annotated[SragFilters, Depends(common_filters)],
    service: Annotated[SragService, Depends(get_service)],
    frequencia: Literal["mes", "semana"] = "mes",
):
    return service.get_time_series(filters, frequencia)


@router.get("/ranking", response_model=RankingResponse)
def ranking(
    filters: Annotated[SragFilters, Depends(common_filters)],
    service: Annotated[SragService, Depends(get_service)],
    nivel: Literal["uf", "municipio"] = "uf",
    metrica: Literal["cases", "deaths", "icu"] = "cases",
    limit: RankingLimit = 20,
):
    return service.get_ranking(
        filters,
        level=nivel,
        metric=metrica,
        limit=limit,
    )


@router.get("/comparar", response_model=ComparisonResponse)
def compare(
    service: Annotated[SragService, Depends(get_service)],
    a_ano_inicio: YearQuery = None,
    a_ano_fim: YearQuery = None,
    a_uf: str | None = None,
    a_municipio: str | None = None,
    a_codigo_municipio: int | None = None,
    a_sexo: str | None = None,
    a_faixa_etaria: str | None = None,
    a_etiologia: str | None = None,
    b_ano_inicio: YearQuery = None,
    b_ano_fim: YearQuery = None,
    b_uf: str | None = None,
    b_municipio: str | None = None,
    b_codigo_municipio: int | None = None,
    b_sexo: str | None = None,
    b_faixa_etaria: str | None = None,
    b_etiologia: str | None = None,
):
    filters_a = build_filters(
        ano_inicio=a_ano_inicio,
        ano_fim=a_ano_fim,
        uf=a_uf,
        municipio=a_municipio,
        codigo_municipio=a_codigo_municipio,
        sexo=a_sexo,
        faixa_etaria=a_faixa_etaria,
        etiologia=a_etiologia,
    )
    filters_b = build_filters(
        ano_inicio=b_ano_inicio,
        ano_fim=b_ano_fim,
        uf=b_uf,
        municipio=b_municipio,
        codigo_municipio=b_codigo_municipio,
        sexo=b_sexo,
        faixa_etaria=b_faixa_etaria,
        etiologia=b_etiologia,
    )
    return service.compare(filters_a, filters_b)
