# SRAG Epidemiological Data Platform

Pipeline, camada analÃ­tica e API epidemiolÃ³gica reproduzÃ­veis para dados pÃºblicos do SIVEP-Gripe.

## Estado atual

### Fase 1 â€” Pipeline de dados

CSV SIVEP-Gripe â†’ validaÃ§Ã£o â†’ normalizaÃ§Ã£o â†’ relatÃ³rio de qualidade â†’ Parquet.

### Fase 2 â€” DuckDB + Repository

Os Parquets sÃ£o consultados por uma camada somente-leitura baseada em DuckDB, com filtros por perÃ­odo, UF, municÃ­pio, sexo, faixa etÃ¡ria e etiologia.

### Fase 3 â€” Services + FastAPI

`SragService` concentra mÃ©tricas epidemiolÃ³gicas e a FastAPI expÃµe consultas REST com Swagger/OpenAPI. A camada HTTP nÃ£o executa SQL e nÃ£o calcula diretamente as mÃ©tricas epidemiolÃ³gicas.

## InstalaÃ§Ã£o

No PowerShell, dentro da pasta `guilherme`:

```powershell
python -m pip install -e ".[dev]"
```

## Estrutura dos dados locais

```text
data/
â”œâ”€â”€ raw/
â”œâ”€â”€ parquet/
â”‚   â””â”€â”€ srag/
â”‚       â””â”€â”€ ano=2025/
â”‚           â””â”€â”€ srag.parquet
â””â”€â”€ quality/
```

CSVs, Parquets e relatÃ³rios de qualidade gerados sÃ£o ignorados pelo Git.

## Processar um ano

```powershell
python scripts/ingest_year.py --year 2025 --input data/raw/2025/INFLUD25.csv
```

Para reprocessar:

```powershell
python scripts/ingest_year.py --year 2025 --input data/raw/2025/INFLUD25.csv --force
```

## Processar todos os anos disponÃ­veis

```powershell
python scripts/ingest_all.py --raw-root data/raw
```

## Executar a API

Com pelo menos um Parquet jÃ¡ processado em `data/parquet/srag/ano=YYYY/srag.parquet`:

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

Filtros epidemiolÃ³gicos comuns:

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

MunicÃ­pio por nome exige `uf` ou `codigo_municipio`.

### Exemplos

Casos no ParanÃ¡ em 2025:

```text
GET /api/v1/casos?ano_inicio=2025&ano_fim=2025&uf=PR
```

SÃ©rie mensal:

```text
GET /api/v1/serie-temporal?ano_inicio=2025&uf=PR&frequencia=mes
```

Ranking de UFs por casos:

```text
GET /api/v1/ranking?ano_inicio=2025&nivel=uf&metrica=cases&limit=20
```

ComparaÃ§Ã£o PR x MT:

```text
GET /api/v1/comparar?a_ano_inicio=2025&a_uf=PR&b_ano_inicio=2025&b_uf=MT
```

## MÃ©tricas epidemiolÃ³gicas

### Letalidade SRAG

```text
Ã³bitos SRAG / casos com evoluÃ§Ã£o conhecida
```

A resposta expÃµe `valor`, `numerador`, `denominador` e `ignorados`. Se o denominador for zero, `valor` Ã© `null`.

### ProporÃ§Ã£o de UTI

```text
casos UTI=SIM / casos com informaÃ§Ã£o UTI conhecida
```

`IGNORADO` e `AUSENTE` nÃ£o entram no denominador.

## Comorbidades

A API conta somente flags realmente presentes no Parquet e com valor positivo (`1`). Colunas ausentes nÃ£o sÃ£o interpretadas como ausÃªncia de doenÃ§a.

O mapeamento permitido fica fechado no cÃ³digo do `SragService`; a API nÃ£o recebe nomes de colunas SQL arbitrÃ¡rios.

## Repository

Uso direto da camada analÃ­tica:

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

## PrÃ³ximas fases

1. revisÃ£o metodolÃ³gica e expansÃ£o do Machine Learning;
2. MCP Server reutilizando `SragService`;
3. agente de IA para consultas epidemiolÃ³gicas.

