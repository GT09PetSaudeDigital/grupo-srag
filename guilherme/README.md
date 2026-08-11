# SRAG Epidemiological Data Platform

Pipeline, camada analítica e API epidemiológica reproduzíveis para dados públicos do SIVEP-Gripe.

## Estado atual

### Fase 1 — Pipeline de dados

CSV SIVEP-Gripe → validação → normalização → relatório de qualidade → Parquet.

### Fase 2 — DuckDB + Repository

Os Parquets são consultados por uma camada somente-leitura baseada em DuckDB, com filtros por período, UF, município, sexo, faixa etária e etiologia.

### Fase 3 — Services + FastAPI

`SragService` concentra métricas epidemiológicas e a FastAPI expõe consultas REST com Swagger/OpenAPI. A camada HTTP não executa SQL e não calcula diretamente as métricas epidemiológicas.

## Instalação

No PowerShell, dentro da pasta `guilherme`:

```powershell
python -m pip install -e ".[dev]"
```

## Estrutura dos dados locais

```text
data/
├── raw/
├── parquet/
│   └── srag/
│       └── ano=2025/
│           └── srag.parquet
└── quality/
```

CSVs, Parquets e relatórios de qualidade gerados são ignorados pelo Git.

## Processar um ano

```powershell
python scripts/ingest_year.py --year 2025 --input data/raw/2025/INFLUD25.csv
```

Para reprocessar:

```powershell
python scripts/ingest_year.py --year 2025 --input data/raw/2025/INFLUD25.csv --force
```

## Processar todos os anos disponíveis

```powershell
python scripts/ingest_all.py --raw-root data/raw
```

## Executar a API

Com pelo menos um Parquet já processado em `data/parquet/srag/ano=YYYY/srag.parquet`:

```powershell
python -m uvicorn srag_api.api.app:app --reload
```

URLs locais:

```text
Health:  http://127.0.0.1:8000/health
Swagger: http://127.0.0.1:8000/docs
OpenAPI: http://127.0.0.1:8000/openapi.json
```

## Endpoints

```text
GET /health
GET /api/v1/casos
GET /api/v1/obitos
GET /api/v1/uti
GET /api/v1/comorbidades
GET /api/v1/faixas-etarias
GET /api/v1/etiologia
GET /api/v1/serie-temporal
GET /api/v1/ranking
GET /api/v1/comparar
```

Filtros epidemiológicos comuns:

```text
ano_inicio
ano_fim
uf
municipio
codigo_municipio
sexo
faixa_etaria
etiologia
```

Município por nome exige `uf` ou `codigo_municipio`.

### Exemplos

Casos no Paraná em 2025:

```text
GET /api/v1/casos?ano_inicio=2025&ano_fim=2025&uf=PR
```

Série mensal:

```text
GET /api/v1/serie-temporal?ano_inicio=2025&uf=PR&frequencia=mes
```

Ranking de UFs por casos:

```text
GET /api/v1/ranking?ano_inicio=2025&nivel=uf&metrica=cases&limit=20
```

Comparação PR x MT:

```text
GET /api/v1/comparar?a_ano_inicio=2025&a_uf=PR&b_ano_inicio=2025&b_uf=MT
```

## Métricas epidemiológicas

### Letalidade SRAG

```text
óbitos SRAG / casos com evolução conhecida
```

A resposta expõe `valor`, `numerador`, `denominador` e `ignorados`. Se o denominador for zero, `valor` é `null`.

### Proporção de UTI

```text
casos UTI=SIM / casos com informação UTI conhecida
```

`IGNORADO` e `AUSENTE` não entram no denominador.

## Comorbidades

A API conta somente flags realmente presentes no Parquet e com valor positivo (`1`). Colunas ausentes não são interpretadas como ausência de doença.

O mapeamento permitido fica fechado no código do `SragService`; a API não recebe nomes de colunas SQL arbitrários.

## Repository

Uso direto da camada analítica:

```python
from pathlib import Path

from srag_api.data.repository import SragFilters, SragRepository

repository = SragRepository(Path("data/parquet"))
filtros = SragFilters(ano_inicio=2025, ano_fim=2025, uf="PR")

print(repository.get_total_cases(filtros))
print(repository.get_etiology_distribution(filtros))
```

## Testes

```powershell
python -m pytest -v
```

## Próximas fases

1. revisão metodológica e expansão do Machine Learning;
2. MCP Server reutilizando `SragService`;
3. agente de IA para consultas epidemiológicas.

