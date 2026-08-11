# SRAG Services + FastAPI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar a camada `SragService` e uma API FastAPI pública, somente-leitura, cobrindo todos os endpoints epidemiológicos aprovados na Fase 3.

**Architecture:** `SragRepository` continua sendo a única camada que conhece DuckDB/SQL. `SragService` concentra métricas epidemiológicas, denominadores e comparação de recortes. FastAPI valida parâmetros, monta `SragFilters`, chama o service e serializa respostas.

**Tech Stack:** Python 3.10+, pandas, pyarrow, DuckDB, FastAPI, Pydantic, Uvicorn, httpx, pytest.

## Global Constraints

- Não alterar `guilherme/analise_srag_pr.py`.
- Não alterar dados RAW.
- API somente-leitura.
- Não criar endpoint SQL.
- Não aceitar SQL arbitrário, nomes de coluna arbitrários, `ORDER BY` arbitrário ou expressões livres.
- Município por nome exige UF ou código do município.
- Letalidade usa óbitos SRAG / casos com evolução conhecida.
- Proporção de UTI usa UTI=SIM / casos com informação UTI conhecida.
- Denominador zero retorna `valor = null`, preservando numerador, denominador e ignorados.
- Semana epidemiológica vem de `SEM_PRI` / `SEMANA_EPIDEMIOLOGICA`.
- Preservar todos os testes e comportamentos das Fases 1 e 2.

---

## File Structure

### Create

- `guilherme/src/srag_api/services/__init__.py`
- `guilherme/src/srag_api/services/epidemiology.py`
- `guilherme/src/srag_api/api/__init__.py`
- `guilherme/src/srag_api/api/models.py`
- `guilherme/src/srag_api/api/dependencies.py`
- `guilherme/src/srag_api/api/routes/__init__.py`
- `guilherme/src/srag_api/api/routes/epidemiology.py`
- `guilherme/src/srag_api/api/app.py`
- `guilherme/tests/unit/test_epidemiology_service.py`
- `guilherme/tests/unit/test_api_models.py`
- `guilherme/tests/unit/test_api_dependencies.py`
- `guilherme/tests/integration/test_api.py`

### Modify

- `guilherme/pyproject.toml`
- `guilherme/src/srag_api/data/repository.py`
- `guilherme/README.md`

---

### Task 1: Dependências FastAPI + scaffolding

**Files:**
- Modify: `guilherme/pyproject.toml`
- Create: `guilherme/src/srag_api/services/__init__.py`
- Create: `guilherme/src/srag_api/api/__init__.py`
- Create: `guilherme/src/srag_api/api/routes/__init__.py`

**Interfaces:**
- Produces: dependências FastAPI instaláveis e pacotes importáveis.

- [ ] **Step 1: Atualizar `pyproject.toml`**

Adicionar em `[project].dependencies`:

```toml
"fastapi>=0.116,<1",
"uvicorn>=0.35,<1",
"httpx>=0.28,<1",
```

- [ ] **Step 2: Instalar projeto em modo editável**

```powershell
python -m pip install -e ".[dev]"
```

- [ ] **Step 3: Criar pacotes vazios**

Criar:

```text
src/srag_api/services/__init__.py
src/srag_api/api/__init__.py
src/srag_api/api/routes/__init__.py
```

- [ ] **Step 4: Verificar imports**

```powershell
python -c "import fastapi, httpx, uvicorn; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml src/srag_api/services/__init__.py src/srag_api/api/__init__.py src/srag_api/api/routes/__init__.py
git commit -m "chore: adiciona dependencias e estrutura FastAPI"
```

---

### Task 2: Repository de comorbidades

**Files:**
- Modify: `guilherme/src/srag_api/data/repository.py`
- Test: `guilherme/tests/integration/test_repository.py`

**Interfaces:**
- Consumes: `SragFilters`.
- Produces:
  - `get_available_columns() -> set[str]`
  - `get_comorbidity_distribution(filters, column_map) -> list[dict[str, object]]`

- [ ] **Step 1: Adicionar testes**

Adicionar flags na fixture:

```python
"CARDIOPATI": 1,
"DIABETES": 2,
"OBESIDADE": 9,
```

Adicionar testes:

```python
def test_available_columns(repository):
    columns = repository.get_available_columns()
    assert "CARDIOPATI" in columns
    assert "DIABETES" in columns


def test_comorbidity_distribution_counts_only_yes(repository):
    result = repository.get_comorbidity_distribution(
        SragFilters(uf="PR"),
        {
            "CARDIOPATI": "CARDIOPATIA",
            "DIABETES": "DIABETES",
            "OBESIDADE": "OBESIDADE",
        },
    )

    counts = {item["comorbidade"]: item["casos"] for item in result}
    assert counts["CARDIOPATIA"] >= 1
    assert counts.get("OBESIDADE", 0) == 0
```

- [ ] **Step 2: Rodar para falhar**

```powershell
python -m pytest tests/integration/test_repository.py -v
```

- [ ] **Step 3: Implementar `get_available_columns`**

```python
def get_available_columns(self) -> set[str]:
    with self._connect() as connection:
        rows = connection.execute(
            f"DESCRIBE {self.VIEW_NAME}"
        ).fetchall()

    return {str(row[0]).upper() for row in rows}
```

- [ ] **Step 4: Implementar agregação segura de comorbidades**

Somente colunas definidas em mapa interno podem virar SQL; nenhuma coluna vem diretamente da URL.

- [ ] **Step 5: Rodar testes**

```powershell
python -m pytest tests/integration/test_repository.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/data/repository.py tests/integration/test_repository.py
git commit -m "feat: adiciona agregacao segura de comorbidades"
```

---

### Task 3: SragService — métricas principais

**Files:**
- Create: `guilherme/src/srag_api/services/epidemiology.py`
- Modify: `guilherme/src/srag_api/services/__init__.py`
- Create: `guilherme/tests/unit/test_epidemiology_service.py`

**Interfaces:**
- Produces:
  - `get_cases`
  - `get_deaths`
  - `get_mortality`
  - `get_icu`
  - `get_icu_proportion`

- [ ] **Step 1: Criar repository fake**

```python
class FakeRepository:
    def get_total_cases(self, filters):
        return 100

    def get_deaths(self, filters):
        return 10

    def get_known_outcome_count(self, filters):
        return 80

    def get_unknown_outcome_count(self, filters):
        return 20

    def get_icu_cases(self, filters):
        return 16

    def get_known_icu_count(self, filters):
        return 64

    def get_unknown_icu_count(self, filters):
        return 36
```

- [ ] **Step 2: Escrever testes**

```python
def test_mortality_uses_known_outcomes():
    service = SragService(FakeRepository())

    assert service.get_mortality(SragFilters()) == {
        "valor": 12.5,
        "numerador": 10,
        "denominador": 80,
        "ignorados": 20,
    }
```

Adicionar teste equivalente para UTI e para contagem de casos.

- [ ] **Step 3: Testar denominador zero**

```python
class ZeroDenominatorRepository(FakeRepository):
    def get_deaths(self, filters):
        return 0

    def get_known_outcome_count(self, filters):
        return 0

    def get_unknown_outcome_count(self, filters):
        return 100
```

Esperado: `valor is None`.

- [ ] **Step 4: Rodar para falhar**

```powershell
python -m pytest tests/unit/test_epidemiology_service.py -v
```

- [ ] **Step 5: Implementar `SragService`**

Criar helper `_percentage()` e métodos públicos descritos acima.

- [ ] **Step 6: Exportar service**

```python
from srag_api.services.epidemiology import SragService

__all__ = ["SragService"]
```

- [ ] **Step 7: Rodar testes**

```powershell
python -m pytest tests/unit/test_epidemiology_service.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src/srag_api/services/epidemiology.py src/srag_api/services/__init__.py tests/unit/test_epidemiology_service.py
git commit -m "feat: adiciona service epidemiologico SRAG"
```

---

### Task 4: SragService — distribuições, séries, ranking e comparação

**Files:**
- Modify: `guilherme/src/srag_api/services/epidemiology.py`
- Modify: `guilherme/tests/unit/test_epidemiology_service.py`

**Interfaces:**
- Produces:
  - `get_age_distribution`
  - `get_etiology_distribution`
  - `get_comorbidity_distribution`
  - `get_time_series`
  - `get_ranking`
  - `compare`

- [ ] **Step 1: Expandir fake repository com retornos determinísticos**

Adicionar métodos fake para distribuições, séries e ranking.

- [ ] **Step 2: Adicionar testes das distribuições e séries**

Verificar que o service retorna envelope `{"dados": ...}` e preserva a frequência pública `mes|semana`.

- [ ] **Step 3: Adicionar teste de comparação**

Usar PR com 150 casos/15 óbitos e MT com 100 casos/5 óbitos; esperar:
- diferença de casos = 50;
- diferença de óbitos = 10;
- diferença de letalidade = 10 pontos percentuais.

- [ ] **Step 4: Implementar métodos**

Delegar distribuições/ranking/relação temporal ao repository e calcular comparação no service.

- [ ] **Step 5: Rodar testes**

```powershell
python -m pytest tests/unit/test_epidemiology_service.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/services/epidemiology.py tests/unit/test_epidemiology_service.py
git commit -m "feat: adiciona analises e comparacao ao service SRAG"
```

---

### Task 5: Modelos Pydantic

**Files:**
- Create: `guilherme/src/srag_api/api/models.py`
- Create: `guilherme/tests/unit/test_api_models.py`

**Interfaces:**
- Produces modelos OpenAPI para health, métricas, listas, ranking e comparação.

- [ ] **Step 1: Testar `MetricResponse` com valor nulo**

```python
model = MetricResponse(
    valor=None,
    numerador=0,
    denominador=0,
    ignorados=10,
)
assert model.valor is None
```

- [ ] **Step 2: Rodar para falhar**

```powershell
python -m pytest tests/unit/test_api_models.py -v
```

- [ ] **Step 3: Criar modelos**

Criar:
`MetricResponse`, `CasesResponse`, `DeathsResponse`, `IcuResponse`, `DataListResponse`, `TimeSeriesResponse`, `RankingResponse`, `ComparisonResponse`, `HealthResponse`.

- [ ] **Step 4: Rodar testes**

```powershell
python -m pytest tests/unit/test_api_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/api/models.py tests/unit/test_api_models.py
git commit -m "feat: adiciona modelos de resposta da API SRAG"
```

---

### Task 6: Dependencies da API

**Files:**
- Create: `guilherme/src/srag_api/api/dependencies.py`
- Create: `guilherme/tests/unit/test_api_dependencies.py`

**Interfaces:**
- Produces:
  - `get_parquet_root() -> Path`
  - `get_repository() -> SragRepository`
  - `get_service() -> SragService`

- [ ] **Step 1: Testar caminho padrão**

```python
assert get_parquet_root() == Path("data/parquet")
```

- [ ] **Step 2: Rodar para falhar**

```powershell
python -m pytest tests/unit/test_api_dependencies.py -v
```

- [ ] **Step 3: Implementar dependencies**

Instanciar `SragRepository(get_parquet_root())` e então `SragService(repository)`.

- [ ] **Step 4: Rodar teste**

```powershell
python -m pytest tests/unit/test_api_dependencies.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/api/dependencies.py tests/unit/test_api_dependencies.py
git commit -m "feat: adiciona dependencias da API SRAG"
```

---

### Task 7: Endpoints básicos

**Files:**
- Create: `guilherme/src/srag_api/api/routes/epidemiology.py`
- Create: `guilherme/tests/integration/test_api.py`

**Interfaces:**
- Produces:
  - `/api/v1/casos`
  - `/api/v1/obitos`
  - `/api/v1/uti`
  - `/api/v1/faixas-etarias`
  - `/api/v1/etiologia`
  - `/api/v1/comorbidades`

- [ ] **Step 1: Criar `build_filters()`**

Mapear os oito filtros comuns para `SragFilters`.

- [ ] **Step 2: Criar fake service para TestClient**

Usar `dependency_overrides[get_service]` para impedir acesso a Parquets reais.

- [ ] **Step 3: Testar seis endpoints**

Esperar status 200 e payloads coerentes.

- [ ] **Step 4: Implementar seis rotas**

As rotas apenas:
1. recebem/validam parâmetros;
2. montam filtros;
3. chamam service;
4. retornam resposta.

- [ ] **Step 5: Converter `ValueError` de domínio em HTTP 400**

Exemplo: município sem UF/código.

- [ ] **Step 6: Rodar integração**

```powershell
python -m pytest tests/integration/test_api.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/srag_api/api/routes/epidemiology.py tests/integration/test_api.py
git commit -m "feat: adiciona endpoints epidemiologicos basicos"
```

---

### Task 8: Série temporal e ranking

**Files:**
- Modify: `guilherme/src/srag_api/api/routes/epidemiology.py`
- Modify: `guilherme/tests/integration/test_api.py`

**Interfaces:**
- Produces:
  - `/api/v1/serie-temporal`
  - `/api/v1/ranking`

- [ ] **Step 1: Testar série mensal e semanal**

- [ ] **Step 2: Restringir frequência**

```python
Literal["mes", "semana"]
```

Frequência inválida deve gerar HTTP 422.

- [ ] **Step 3: Implementar ranking**

Parâmetros:

```python
nivel: Literal["uf", "municipio"] = "uf"
metrica: Literal["cases", "deaths", "icu"] = "cases"
limit: Annotated[int, Query(ge=1, le=100)] = 20
```

- [ ] **Step 4: Testar `limit=101`**

Expected: HTTP 422.

- [ ] **Step 5: Rodar integração**

```powershell
python -m pytest tests/integration/test_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/api/routes/epidemiology.py tests/integration/test_api.py
git commit -m "feat: adiciona serie temporal e ranking na API"
```

---

### Task 9: Endpoint comparar

**Files:**
- Modify: `guilherme/src/srag_api/api/routes/epidemiology.py`
- Modify: `guilherme/tests/integration/test_api.py`

**Interfaces:**
- Produces `/api/v1/comparar`.

- [ ] **Step 1: Definir parâmetros `a_*` e `b_*`**

Cada recorte terá os oito filtros comuns com prefixos `a_` e `b_`.

- [ ] **Step 2: Testar PR vs MT**

Esperar `recorte_a`, `recorte_b` e `diferenca`.

- [ ] **Step 3: Implementar rota**

Montar dois `SragFilters` e chamar `service.compare(filters_a, filters_b)`.

- [ ] **Step 4: Testar município inválido em um recorte**

Expected: HTTP 400.

- [ ] **Step 5: Rodar integração**

```powershell
python -m pytest tests/integration/test_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/api/routes/epidemiology.py tests/integration/test_api.py
git commit -m "feat: adiciona comparacao de recortes SRAG"
```

---

### Task 10: App FastAPI + health + 404

**Files:**
- Create: `guilherme/src/srag_api/api/app.py`
- Modify: `guilherme/tests/integration/test_api.py`

**Interfaces:**
- Produces `app`, `/health` e exception handler de `FileNotFoundError`.

- [ ] **Step 1: Criar app**

```python
app = FastAPI(
    title="SRAG Epidemiological API",
    version="0.3.0",
)
```

Registrar router.

- [ ] **Step 2: Criar `/health`**

Esperado:

```json
{"status":"ok","service":"srag-api"}
```

- [ ] **Step 3: Criar handler 404**

`FileNotFoundError` deve retornar JSON com status 404.

- [ ] **Step 4: Testar app real**

Importar `app` no `TestClient`.

- [ ] **Step 5: Rodar integração**

```powershell
python -m pytest tests/integration/test_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/api/app.py tests/integration/test_api.py
git commit -m "feat: cria aplicacao FastAPI SRAG"
```

---

### Task 11: Validação temporal HTTP

**Files:**
- Modify: `guilherme/src/srag_api/api/routes/epidemiology.py`
- Modify: `guilherme/tests/integration/test_api.py`

**Interfaces:**
- Garante anos 2019–2026 e intervalo coerente.

- [ ] **Step 1: Restringir anos**

Usar `Query(ge=2019, le=2026)` em filtros normais e de comparação.

- [ ] **Step 2: Testar ano 2018**

Expected: HTTP 422.

- [ ] **Step 3: Validar intervalo invertido**

`ano_inicio > ano_fim` deve gerar `ValueError("ano_inicio nao pode ser maior que ano_fim.")`, convertido em HTTP 400.

- [ ] **Step 4: Rodar integração**

```powershell
python -m pytest tests/integration/test_api.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/api/routes/epidemiology.py tests/integration/test_api.py
git commit -m "feat: valida filtros temporais da API SRAG"
```

---

### Task 12: README + verificação completa

**Files:**
- Modify: `guilherme/README.md`

**Interfaces:**
- Documenta instalação, execução, Swagger e endpoints.

- [ ] **Step 1: Documentar execução**

```powershell
cd guilherme
python -m pip install -e ".[dev]"
uvicorn srag_api.api.app:app --reload
```

- [ ] **Step 2: Documentar URLs**

```text
Swagger: http://127.0.0.1:8000/docs
OpenAPI: http://127.0.0.1:8000/openapi.json
Health: http://127.0.0.1:8000/health
```

- [ ] **Step 3: Listar os dez endpoints**

- [ ] **Step 4: Rodar suíte completa**

```powershell
python -m pytest -v
```

Expected: todos os testes das Fases 1, 2 e 3 PASS.

- [ ] **Step 5: Smoke test**

Com API rodando:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/openapi.json
```

- [ ] **Step 6: Commit**

```powershell
git add README.md
git commit -m "docs: documenta API epidemiologica SRAG"
```

- [ ] **Step 7: Verificação pós-commit**

```powershell
git status
python -m pytest -v
```

Expected:
- working tree clean;
- suíte completa PASS.
