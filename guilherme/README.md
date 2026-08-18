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

## Classificação final e etiologia laboratorial

O pipeline mantém separadas duas dimensões analíticas que não devem ser confundidas:

- `CLASSIFICACAO_FINAL_NORMALIZADA`: deriva exclusivamente de `CLASSI_FIN`;
- `ETIOLOGIA_DETALHADA`: deriva exclusivamente das flags laboratoriais disponíveis no registro.

Mapeamento de `CLASSI_FIN`:

```text
1 -> INFLUENZA
2 -> OUTRO_VIRUS_RESPIRATORIO
3 -> OUTRO_AGENTE_ETIOLOGICO
4 -> NAO_ESPECIFICADO
5 -> COVID-19
ausente -> AUSENTE
valor inesperado -> OUTRO
```

A etiologia detalhada identifica, quando houver flag laboratorial positiva, agentes como `SARS-CoV-2`, `Influenza A`, `Influenza B`, `VSR`, adenovírus, parainfluenza, metapneumovírus, bocavírus e rinovírus. Quando nenhum campo configurado estiver positivo, o valor é `NAO_IDENTIFICADA`.

As duas colunas são independentes. Por exemplo, um registro com `CLASSI_FIN=4` e `PCR_SARS2=1` permanece com:

```text
CLASSIFICACAO_FINAL_NORMALIZADA = NAO_ESPECIFICADO
ETIOLOGIA_DETALHADA = SARS-CoV-2
```

O pipeline não usa a classificação final para inferir resultado laboratorial e não usa o resultado laboratorial para sobrescrever a classificação final.

### Compatibilidade entre anos

O processamento cobre 2019–2026 com um único pipeline. Diferenças históricas de schema são tratadas como disponibilidade de colunas, e não como regras especiais por ano.

Uma coluna ausente no arquivo de origem não é criada artificialmente com valor negativo. Portanto:

```text
coluna inexistente != resultado negativo
valor ausente       != resultado positivo
```

As colunas originais presentes no CSV são preservadas no conjunto transformado, permitindo auditoria e análises posteriores, inclusive para Machine Learning.

O filtro público `etiologia` e o endpoint `GET /api/v1/etiologia` continuam estáveis; internamente, a camada analítica consulta `ETIOLOGIA_DETALHADA`.

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

## Machine Learning — Predição de Óbito na Admissão

O módulo `src/srag_api/ml/` prepara um dataset nacional de SRAG para experimentos de Machine Learning voltados à predição de óbito por SRAG usando somente informações disponíveis até a notificação/admissão.

### Pergunta de pesquisa

Dado um paciente hospitalizado por SRAG, qual a probabilidade de óbito por SRAG usando apenas dados disponíveis até a notificação/admissão?

### Definição do alvo

- `CURA` → `0`
- `OBITO_SRAG` → `1`
- `OBITO_OUTRAS_CAUSAS` → excluído do treino
- `AUSENTE`, `IGNORADO` e outros desfechos indefinidos → excluídos do treino

### Política de features

A V1 usa grupos de features demográficas, sintomas, comorbidades, UF/região e variáveis temporais conhecidas até a notificação/admissão.

Município fica fora da V1 para reduzir alta cardinalidade e risco de memorização de padrões locais.

Variáveis conhecidas por causar leakage ficam explicitamente bloqueadas, incluindo:

- `EVOLUCAO`
- `DESFECHO_NORMALIZADO`
- `OBITO_SRAG`
- `DT_EVOLUCA`
- `UTI`
- `SUPORT_VEN`
- `QTD_DIAS`
- `DIAS_INTERNA`
- `PCR_EVOLUCAO`

### Validação temporal

A divisão padrão é:

- 2019–2024 → treino
- 2025 → validação
- 2026 → teste fora do tempo

O conjunto de 2026 não participa de imputação, encoding, scaler, balanceamento ou ajuste de hiperparâmetros.

### Pré-processamento

O pré-processador é ajustado exclusivamente no conjunto de treino.

- imputação numérica por mediana
- imputação categórica pelo valor mais frequente
- `OneHotEncoder(handle_unknown="ignore")`
- `StandardScaler` para features numéricas
- validação e teste recebem apenas `transform`

O balanceamento é restrito à partição de treino. Nesta etapa da V1, `strategy="none"` é suportada explicitamente; métodos como SMOTE ficam para a fase de experimentação de modelos.

### Estrutura

```text
src/srag_api/ml/
├── __init__.py
├── features.py
├── target.py
├── dataset.py
├── split.py
└── preprocessing.py
```

O treinamento e comparação de modelos preditivos são uma etapa posterior, depois da validação do dataset V1.
