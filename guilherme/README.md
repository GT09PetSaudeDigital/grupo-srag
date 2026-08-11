# SRAG Epidemiological Data Platform

Pipeline reprodutível para preparação e análise de dados públicos do SIVEP-Gripe.

## Estado atual

Fase 1 implementada:

CSV SIVEP-Gripe → validação → normalização → relatório de qualidade → Parquet.

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
└── quality/
```

## Processar um ano

```powershell
python scripts/ingest_year.py --year 2025 --input data/raw/2025/INFLUD25.csv
```

## Reprocessar

```powershell
python scripts/ingest_year.py --year 2025 --input data/raw/2025/INFLUD25.csv --force
```

## Processar todos os anos disponíveis

```powershell
python scripts/ingest_all.py --raw-root data/raw
```

## Testes

```powershell
python -m pytest -v
```

## Próximas fases

1. DuckDB + Repository
2. FastAPI + OpenAPI
3. revisão metodológica do Machine Learning
4. MCP Server
