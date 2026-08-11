# SRAG Services + FastAPI — Design da Fase 3

**Data:** 2026-08-11  
**Status:** Aprovado pelo usuário

## Objetivo

Evoluir a plataforma SRAG da camada analítica em DuckDB para uma API epidemiológica pública, somente-leitura e reutilizável por dashboards, agentes de IA e, futuramente, um MCP Server.

Arquitetura:

```text
SIVEP-Gripe CSV
      ↓
Pipeline de limpeza e validação
      ↓
Parquet particionado por ano
      ↓
DuckDB
      ↓
SragRepository
      ↓
SragService
      ↓
FastAPI
      ↓
Swagger / OpenAPI
```

Responsabilidades:
- `SragRepository`: consulta os dados e conhece SQL/DuckDB.
- `SragService`: calcula métricas epidemiológicas e combina resultados.
- `FastAPI`: recebe parâmetros HTTP, valida entradas e serializa respostas.

## Escopo

Endpoints:

- `GET /health`
- `GET /api/v1/casos`
- `GET /api/v1/obitos`
- `GET /api/v1/uti`
- `GET /api/v1/comorbidades`
- `GET /api/v1/faixas-etarias`
- `GET /api/v1/etiologia`
- `GET /api/v1/serie-temporal`
- `GET /api/v1/ranking`
- `GET /api/v1/comparar`

Filtros comuns:
`ano_inicio`, `ano_fim`, `uf`, `municipio`, `codigo_municipio`, `sexo`, `faixa_etaria`, `etiologia`.

Município por nome exige UF ou código do município.

## Contratos

### Health

```json
{"status":"ok","service":"srag-api"}
```

### Casos

Retorna contagem simples e filtros aplicados.

### Óbitos

Letalidade:

```text
óbitos por SRAG / casos com evolução conhecida
```

Resposta inclui `obitos` e `letalidade` com `valor`, `numerador`, `denominador` e `ignorados`.

Casos com desfecho ignorado, ausente, outro ou óbito por outras causas não entram no denominador da letalidade SRAG.

### UTI

Proporção de UTI:

```text
casos UTI=SIM / casos com informação UTI conhecida
```

`IGNORADO` e `AUSENTE` não entram no denominador.

### Faixas etárias

Faixas:
`<1`, `1-4`, `5-11`, `12-17`, `18-29`, `30-44`, `45-59`, `60-74`, `75+`.

### Etiologia

Categorias:
`COVID-19`, `Influenza A`, `Influenza B`, `VSR`, `Outros virus respiratorios`, `Outro agente`, `Nao identificado`, `Ignorado`.

### Comorbidades

Agregar somente flags realmente existentes no Parquet. Ausência de coluna não significa ausência de doença.

Mapeamento inicial:
`CARDIOPATI`, `DIABETES`, `OBESIDADE`, `RENAL`, `HEPATICA`, `IMUNODEPRE`, `ASMA`, `PNEUMOPATI`, `NEUROLOGIC`, `HEMATOLOGI`.

### Série temporal

Frequências permitidas: `mes` e `semana`.

A semana epidemiológica vem de `SEM_PRI` / `SEMANA_EPIDEMIOLOGICA`; não será reconstruída artificialmente a partir de data civil.

### Ranking

Níveis: `uf`, `municipio`.

Métricas: `cases`, `deaths`, `icu`.

Limite padrão 20 e máximo 100.

### Comparar

Compara dois recortes independentes, por exemplo PR vs MT, Curitiba vs Londrina ou 2024 vs 2025.

O serviço retorna ambos os recortes e diferenças absolutas. Para percentuais, diferenças serão expressas em pontos percentuais.

## SragService

Estrutura:

```text
src/srag_api/services/
    __init__.py
    epidemiology.py
```

Métodos previstos:
- `get_cases`
- `get_deaths`
- `get_mortality`
- `get_icu`
- `get_icu_proportion`
- `get_age_distribution`
- `get_etiology_distribution`
- `get_comorbidity_distribution`
- `get_time_series`
- `get_ranking`
- `compare`

Quando o denominador for zero, a métrica retornará `valor = null`, preservando numerador, denominador e ignorados.

## FastAPI

Estrutura:

```text
src/srag_api/api/
    __init__.py
    app.py
    dependencies.py
    models.py
    routes/
        __init__.py
        epidemiology.py
```

Fluxo:

```text
HTTP → FastAPI/Pydantic → SragFilters → SragService → SragRepository → DuckDB → Parquet
```

## Erros

- `400`: combinação semanticamente inválida.
- `404`: nenhum Parquet processado disponível.
- `422`: erro de validação estrutural do FastAPI/Pydantic.
- `500`: somente falhas inesperadas.

## Segurança

Não haverá:
- endpoint SQL;
- SQL arbitrário;
- nomes arbitrários de coluna;
- `ORDER BY` arbitrário;
- expressões SQL livres.

Seletores terão conjuntos fechados, por exemplo:
- `frequency`: `mes|semana`
- `level`: `uf|municipio`
- `metric`: `cases|deaths|icu`

## Testes

### Unitários do Service

Cobrir:
casos, óbitos, letalidade, UTI, proporção UTI, denominador zero, ignorados, faixas etárias, etiologia, comorbidades, série temporal, ranking e comparação.

### Integração da API

Cobrir todos os endpoints e também:
- município sem UF/código;
- ano inválido;
- frequência inválida;
- métrica inválida;
- ranking acima de 100;
- dataset inexistente;
- denominador zero.

## Swagger / OpenAPI

Execução:

```powershell
uvicorn srag_api.api.app:app --reload
```

Swagger:
`http://127.0.0.1:8000/docs`

OpenAPI:
`http://127.0.0.1:8000/openapi.json`

## Dependências novas

Adicionar:
`fastapi`, `uvicorn`, `httpx`.

Manter:
`pandas`, `pyarrow`, `duckdb`, `pytest`.

## Restrições

- não alterar `guilherme/analise_srag_pr.py`;
- não alterar dados RAW;
- API somente-leitura;
- MCP não acessa DuckDB diretamente;
- lógica epidemiológica fora das rotas;
- preservar testes das Fases 1 e 2.

## Resultado esperado

```text
SIVEP-Gripe
    ↓
CSV
    ↓
limpeza e validação
    ↓
Parquet
    ↓
DuckDB
    ↓
SragRepository
    ↓
SragService
    ↓
FastAPI
    ↓
Swagger / OpenAPI
```
