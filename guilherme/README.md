# SRAG Epidemiological Data Platform

Pipeline e camada analítica reprodutíveis para dados públicos do SIVEP-Gripe.

## Estado atual

### Fase 1 — concluída

CSV SIVEP-Gripe → validação → normalização → relatório de qualidade → Parquet.

### Fase 2 — DuckDB + Repository

Os Parquets podem ser consultados por uma camada somente-leitura baseada em DuckDB, com filtros por período, UF, município, sexo, faixa etária e etiologia.

## Instalação

```powershell
cd guilherme
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

Os CSVs, Parquets e relatórios de qualidade gerados são ignorados pelo Git.

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

## Repository

Exemplo:

```python
from pathlib import Path

from srag_api.data.repository import SragFilters, SragRepository

repository = SragRepository(Path("data/parquet"))

filtros = SragFilters(
    ano_inicio=2025,
    ano_fim=2025,
    uf="PR",
)

print(repository.get_total_cases(filtros))
print(repository.get_etiology_distribution(filtros))
```

Principais consultas disponíveis:

- total de casos;
- óbitos;
- internações em UTI;
- contagens com desfecho conhecido/ignorado;
- contagens com UTI conhecida/ignorada;
- distribuição por faixa etária;
- distribuição etiológica;
- séries mensais;
- séries por semana epidemiológica;
- rankings por estado e município.

## Testes

```powershell
python -m pytest -v
```

## Próximas fases

1. Services epidemiológicos e métricas com denominadores explícitos
2. FastAPI + OpenAPI
3. revisão metodológica do Machine Learning
4. MCP Server
