# SRAG Etiology Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separar a classificação final oficial do SIVEP-Gripe da etiologia laboratorial detalhada, mantendo compatibilidade da API e preservando diferenças históricas de schema entre 2019 e 2026.

**Architecture:** `CLASSI_FIN` será normalizado em `CLASSIFICACAO_FINAL_NORMALIZADA`, enquanto campos PCR gerarão `ETIOLOGIA_DETALHADA`. O pipeline manterá as colunas fonte intactas; o repository migrará o filtro e a distribuição de etiologia para `ETIOLOGIA_DETALHADA`, preservando a interface pública `/api/v1/etiologia`.

**Tech Stack:** Python 3.13, pandas >=2.0, pyarrow >=14.0, DuckDB >=1.4,<2, FastAPI >=0.116,<1, pytest >=8.

## Global Constraints

- Preservar todas as colunas originais do SIVEP-Gripe.
- Coluna inexistente não equivale a resultado negativo.
- Não criar lógica específica por ano sem evidência documental ou empírica.
- `CLASSIFICACAO_FINAL_NORMALIZADA` deve depender somente de `CLASSI_FIN`.
- `ETIOLOGIA_DETALHADA` deve depender somente de campos laboratoriais disponíveis.
- Não corrigir silenciosamente conflitos entre `CLASSI_FIN` e resultados laboratoriais.
- Manter `/api/v1/etiologia` funcional durante a transição.
- Não reprocessar todas as bases reais antes da validação do schema real de 2019 e 2026.
- Não incluir treinamento de Machine Learning nesta alteração.
- Trabalhar em branch separada: `feature/etiology-normalization`.

---

## File Map

**Modify**
- `guilherme/src/srag_api/data/etiology.py` — novas funções de normalização.
- `guilherme/src/srag_api/data/ingest.py` — usar `add_etiology_columns`.
- `guilherme/src/srag_api/data/repository.py` — filtrar/agrupar por `ETIOLOGIA_DETALHADA`.
- `guilherme/tests/unit/test_etiology.py` — novo contrato unitário.
- `guilherme/tests/unit/test_ingest.py` — garantir integração no dataframe transformado.
- `guilherme/tests/integration/test_repository.py` — garantir filtro/distribuição sobre nova coluna.
- `guilherme/tests/integration/test_api.py` — garantir compatibilidade do endpoint `/etiologia`.
- `guilherme/README.md` — documentar semântica das duas colunas.

**Keep unchanged unless a test proves necessary**
- `guilherme/src/srag_api/services/epidemiology.py`
- `guilherme/src/srag_api/api/routes/epidemiology.py`
- `guilherme/src/srag_api/api/models.py`

---

### Task 1: Normalize official final classification

**Files:**
- Modify: `guilherme/src/srag_api/data/etiology.py`
- Test: `guilherme/tests/unit/test_etiology.py`

**Interfaces:**
- Produces: `normalize_final_classification(value: object) -> str`
- Produces values: `INFLUENZA`, `OUTRO_VIRUS_RESPIRATORIO`, `OUTRO_AGENTE_ETIOLOGICO`, `NAO_ESPECIFICADO`, `COVID-19`, `AUSENTE`, `OUTRO`

- [ ] **Step 1: Replace old classification-priority tests with explicit mapping tests**

Add:

```python
import math

import pandas as pd

from srag_api.data.etiology import normalize_final_classification


def test_final_classification_influenza():
    assert normalize_final_classification(1) == "INFLUENZA"


def test_final_classification_other_respiratory_virus():
    assert normalize_final_classification(2) == "OUTRO_VIRUS_RESPIRATORIO"


def test_final_classification_other_agent():
    assert normalize_final_classification(3) == "OUTRO_AGENTE_ETIOLOGICO"


def test_final_classification_unspecified():
    assert normalize_final_classification(4) == "NAO_ESPECIFICADO"


def test_final_classification_covid():
    assert normalize_final_classification(5) == "COVID-19"


def test_final_classification_missing():
    assert normalize_final_classification(None) == "AUSENTE"
    assert normalize_final_classification(float("nan")) == "AUSENTE"


def test_final_classification_unexpected():
    assert normalize_final_classification(9) == "OUTRO"
    assert normalize_final_classification(99) == "OUTRO"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run from `guilherme`:

```powershell
python -m pytest tests/unit/test_etiology.py -v
```

Expected: FAIL because `normalize_final_classification` does not exist yet.

- [ ] **Step 3: Implement the minimal mapping**

In `src/srag_api/data/etiology.py`:

```python
FINAL_CLASSIFICATION_MAP = {
    1: "INFLUENZA",
    2: "OUTRO_VIRUS_RESPIRATORIO",
    3: "OUTRO_AGENTE_ETIOLOGICO",
    4: "NAO_ESPECIFICADO",
    5: "COVID-19",
}


def normalize_final_classification(value: object) -> str:
    if pd.isna(value):
        return "AUSENTE"
    return FINAL_CLASSIFICATION_MAP.get(value, "OUTRO")
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

```powershell
python -m pytest tests/unit/test_etiology.py -v
```

Expected: classification tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/data/etiology.py tests/unit/test_etiology.py
git commit -m "feat: normaliza classificacao final SRAG"
```

---

### Task 2: Add detailed laboratory etiology

**Files:**
- Modify: `guilherme/src/srag_api/data/etiology.py`
- Test: `guilherme/tests/unit/test_etiology.py`

**Interfaces:**
- Consumes: `pd.Series`
- Produces: `normalize_detailed_etiology(row: pd.Series) -> str`
- Produces: `add_etiology_columns(df: pd.DataFrame) -> pd.DataFrame`
- New columns:
  - `CLASSIFICACAO_FINAL_NORMALIZADA`
  - `ETIOLOGIA_DETALHADA`

- [ ] **Step 1: Add detailed etiology tests**

Add:

```python
from srag_api.data.etiology import (
    add_etiology_columns,
    normalize_detailed_etiology,
)


def make_row(**kwargs):
    return pd.Series(kwargs)


def test_detailed_etiology_sars_cov_2():
    assert normalize_detailed_etiology(
        make_row(PCR_SARS2=1)
    ) == "SARS-CoV-2"


def test_detailed_etiology_influenza_a():
    assert normalize_detailed_etiology(
        make_row(PCR_FLUAS=1)
    ) == "Influenza A"


def test_detailed_etiology_influenza_b():
    assert normalize_detailed_etiology(
        make_row(PCR_FLUBS=1)
    ) == "Influenza B"


def test_detailed_etiology_vsr():
    assert normalize_detailed_etiology(
        make_row(PCR_VSR=1)
    ) == "VSR"


def test_detailed_etiology_specific_other_viruses():
    assert normalize_detailed_etiology(
        make_row(PCR_ADENO=1)
    ) == "Adenovirus"
    assert normalize_detailed_etiology(
        make_row(PCR_PARA1=1)
    ) == "Parainfluenza 1"
    assert normalize_detailed_etiology(
        make_row(PCR_PARA2=1)
    ) == "Parainfluenza 2"
    assert normalize_detailed_etiology(
        make_row(PCR_PARA3=1)
    ) == "Parainfluenza 3"
    assert normalize_detailed_etiology(
        make_row(PCR_PARA4=1)
    ) == "Parainfluenza 4"
    assert normalize_detailed_etiology(
        make_row(PCR_METAP=1)
    ) == "Metapneumovirus"
    assert normalize_detailed_etiology(
        make_row(PCR_BOCA=1)
    ) == "Bocavirus"
    assert normalize_detailed_etiology(
        make_row(PCR_RINO=1)
    ) == "Rinovirus"


def test_missing_pcr_column_is_not_negative():
    assert normalize_detailed_etiology(
        make_row(CLASSI_FIN=1)
    ) == "NAO_IDENTIFICADA"


def test_missing_pcr_value_is_not_positive():
    assert normalize_detailed_etiology(
        make_row(PCR_SARS2=None, PCR_FLUAS=None)
    ) == "NAO_IDENTIFICADA"


def test_final_classification_does_not_override_lab_result():
    row = make_row(CLASSI_FIN=4, PCR_SARS2=1)
    assert normalize_final_classification(row["CLASSI_FIN"]) == "NAO_ESPECIFICADO"
    assert normalize_detailed_etiology(row) == "SARS-CoV-2"


def test_add_etiology_columns_preserves_source_columns():
    df = pd.DataFrame(
        {
            "CLASSI_FIN": [4, 2],
            "PCR_SARS2": [1, None],
            "PCR_VSR": [None, 1],
        }
    )

    result = add_etiology_columns(df)

    assert result["CLASSI_FIN"].tolist() == [4, 2]
    assert result["PCR_SARS2"].iloc[0] == 1
    assert result["CLASSIFICACAO_FINAL_NORMALIZADA"].tolist() == [
        "NAO_ESPECIFICADO",
        "OUTRO_VIRUS_RESPIRATORIO",
    ]
    assert result["ETIOLOGIA_DETALHADA"].tolist() == [
        "SARS-CoV-2",
        "VSR",
    ]
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m pytest tests/unit/test_etiology.py -v
```

Expected: FAIL because new detailed functions do not exist.

- [ ] **Step 3: Implement explicit laboratory mapping**

Replace the old single-category implementation with:

```python
LAB_ETIOLOGY_FIELDS = (
    ("PCR_SARS2", "SARS-CoV-2"),
    ("PCR_FLUAS", "Influenza A"),
    ("PCR_FLUBS", "Influenza B"),
    ("PCR_VSR", "VSR"),
    ("PCR_ADENO", "Adenovirus"),
    ("PCR_PARA1", "Parainfluenza 1"),
    ("PCR_PARA2", "Parainfluenza 2"),
    ("PCR_PARA3", "Parainfluenza 3"),
    ("PCR_PARA4", "Parainfluenza 4"),
    ("PCR_METAP", "Metapneumovirus"),
    ("PCR_BOCA", "Bocavirus"),
    ("PCR_RINO", "Rinovirus"),
)


def _is_positive(row: pd.Series, field: str) -> bool:
    return field in row.index and row.get(field) == 1


def normalize_detailed_etiology(row: pd.Series) -> str:
    for field, label in LAB_ETIOLOGY_FIELDS:
        if _is_positive(row, field):
            return label
    return "NAO_IDENTIFICADA"


def add_etiology_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "CLASSI_FIN" in result.columns:
        result["CLASSIFICACAO_FINAL_NORMALIZADA"] = result[
            "CLASSI_FIN"
        ].map(normalize_final_classification)
    else:
        result["CLASSIFICACAO_FINAL_NORMALIZADA"] = "AUSENTE"

    result["ETIOLOGIA_DETALHADA"] = result.apply(
        normalize_detailed_etiology,
        axis=1,
    )
    return result
```

Remove `normalize_etiology()` and `add_etiology_column()` only after all call sites are migrated in Task 3.

- [ ] **Step 4: Run unit tests**

```powershell
python -m pytest tests/unit/test_etiology.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/data/etiology.py tests/unit/test_etiology.py
git commit -m "feat: separa etiologia laboratorial da classificacao final"
```

---

### Task 3: Integrate the two columns into the ingest pipeline

**Files:**
- Modify: `guilherme/src/srag_api/data/ingest.py`
- Test: `guilherme/tests/unit/test_ingest.py`

**Interfaces:**
- Consumes: `add_etiology_columns(df: pd.DataFrame)`
- Produces transformed DataFrame with both analytical columns and intact source columns.

- [ ] **Step 1: Add integration assertions to `test_ingest.py`**

Add a focused test:

```python
import pandas as pd

from srag_api.data.ingest import transform_srag_dataframe


def test_transform_adds_final_classification_and_detailed_etiology():
    raw_df = pd.DataFrame(
        [
            {
                "TP_IDADE": 3,
                "NU_IDADE_N": 40,
                "SG_UF": "MT",
                "ID_MUNICIP": "CUIABA",
                "CO_MUN_RES": 510340,
                "EVOLUCAO": 1,
                "UTI": 2,
                "CS_SEXO": "F",
                "CLASSI_FIN": 4,
                "PCR_SARS2": 1,
            }
        ]
    )

    result = transform_srag_dataframe(raw_df, 2025)

    assert result.loc[0, "CLASSIFICACAO_FINAL_NORMALIZADA"] == "NAO_ESPECIFICADO"
    assert result.loc[0, "ETIOLOGIA_DETALHADA"] == "SARS-CoV-2"
    assert result.loc[0, "CLASSI_FIN"] == 4
    assert result.loc[0, "PCR_SARS2"] == 1
```

If current required schema needs temporal fields, include the same mandatory fields used by existing ingest fixtures rather than weakening schema validation.

- [ ] **Step 2: Run test and verify RED**

```powershell
python -m pytest tests/unit/test_ingest.py -v
```

Expected: FAIL because transform still calls `add_etiology_column`.

- [ ] **Step 3: Migrate ingest import and call**

Change:

```python
from srag_api.data.etiology import add_etiology_column
```

to:

```python
from srag_api.data.etiology import add_etiology_columns
```

Change:

```python
result = add_etiology_column(result)
```

to:

```python
result = add_etiology_columns(result)
```

- [ ] **Step 4: Remove old compatibility functions from `etiology.py`**

Delete:
- `normalize_etiology`
- `add_etiology_column`
- old `OTHER_RESPIRATORY_FLAGS` if no longer referenced.

- [ ] **Step 5: Run ingest + etiology tests**

```powershell
python -m pytest tests/unit/test_etiology.py tests/unit/test_ingest.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/data/etiology.py src/srag_api/data/ingest.py tests/unit/test_ingest.py
git commit -m "feat: integra novas colunas etiologicas ao pipeline"
```

---

### Task 4: Migrate repository filtering and distribution

**Files:**
- Modify: `guilherme/src/srag_api/data/repository.py`
- Test: `guilherme/tests/integration/test_repository.py`

**Interfaces:**
- `SragFilters.etiologia` remains unchanged for API compatibility.
- Internally `etiologia` now filters `ETIOLOGIA_DETALHADA`.
- `get_etiology_distribution()` returns the same response shape:
  `{"etiologia": <str>, "casos": <int>}`

- [ ] **Step 1: Add repository regression test**

Add:

```python
def test_repository_filters_by_detailed_etiology(tmp_path):
    raw_df = pd.DataFrame(
        [
            {
                "TP_IDADE": 3,
                "NU_IDADE_N": 50,
                "SG_UF": "PR",
                "ID_MUNICIP": "CURITIBA",
                "CO_MUN_RES": 410690,
                "EVOLUCAO": 1,
                "UTI": 2,
                "CS_SEXO": "M",
                "CLASSI_FIN": 2,
                "PCR_VSR": 1,
            },
            {
                "TP_IDADE": 3,
                "NU_IDADE_N": 51,
                "SG_UF": "PR",
                "ID_MUNICIP": "CURITIBA",
                "CO_MUN_RES": 410690,
                "EVOLUCAO": 1,
                "UTI": 2,
                "CS_SEXO": "F",
                "CLASSI_FIN": 1,
                "PCR_FLUAS": 1,
            },
        ]
    )

    parquet_root = tmp_path / "parquet"
    transformed = transform_srag_dataframe(raw_df, 2025)
    write_year_parquet(transformed, parquet_root, 2025)
    repository = SragRepository(parquet_root)

    filters = SragFilters(etiologia="VSR")

    assert repository.get_total_cases(filters) == 1
    assert repository.get_etiology_distribution() == [
        {"etiologia": "Influenza A", "casos": 1},
        {"etiologia": "VSR", "casos": 1},
    ]
```

If sort order differs because both counts equal 1, expected alphabetical order follows current SQL `ORDER BY casos DESC, etiologia`.

- [ ] **Step 2: Run focused test and verify RED**

```powershell
python -m pytest tests/integration/test_repository.py::test_repository_filters_by_detailed_etiology -v
```

Expected: FAIL because SQL still references `ETIOLOGIA_NORMALIZADA`.

- [ ] **Step 3: Change `_where`**

Change:

```python
clauses.append("ETIOLOGIA_NORMALIZADA = ?")
```

to:

```python
clauses.append("ETIOLOGIA_DETALHADA = ?")
```

- [ ] **Step 4: Change `get_etiology_distribution`**

Use:

```python
sql = f"""
    SELECT ETIOLOGIA_DETALHADA AS etiologia, COUNT(*) AS casos
    FROM {self.VIEW_NAME}
    {where}
    GROUP BY ETIOLOGIA_DETALHADA
    ORDER BY casos DESC, etiologia
"""
```

- [ ] **Step 5: Run repository suite**

```powershell
python -m pytest tests/integration/test_repository.py tests/unit/test_repository_comorbidities.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/data/repository.py tests/integration/test_repository.py
git commit -m "refactor: usa etiologia detalhada no repository"
```

---

### Task 5: Preserve public API behavior

**Files:**
- Test: `guilherme/tests/integration/test_api.py`
- Inspect only unless failing:
  - `guilherme/src/srag_api/services/epidemiology.py`
  - `guilherme/src/srag_api/api/routes/epidemiology.py`

**Interfaces:**
- `GET /api/v1/etiologia` remains available.
- `etiologia` query parameter remains named `etiologia`.
- Response continues to contain `dados` with items `{etiologia, casos}`.

- [ ] **Step 1: Add/adjust API test**

Ensure there is a test equivalent to:

```python
def test_etiology_endpoint_uses_detailed_lab_etiology(client):
    response = client.get("/api/v1/etiologia")

    assert response.status_code == 200
    payload = response.json()
    assert "dados" in payload
    assert all(
        {"etiologia", "casos"} <= item.keys()
        for item in payload["dados"]
    )
```

Where the fixture contains at least one PCR-specific positive record, additionally assert that `"VSR"`, `"Influenza A"`, or `"SARS-CoV-2"` appears.

- [ ] **Step 2: Run API tests**

```powershell
python -m pytest tests/integration/test_api.py -v
```

Expected: PASS without modifying service/routes because they delegate to repository.

- [ ] **Step 3: Only if the API test fails, make the minimum compatibility fix**

Do not rename the endpoint or query parameter. Do not add a new endpoint in this PR.

- [ ] **Step 4: Commit test changes**

```powershell
git add tests/integration/test_api.py
git commit -m "test: garante compatibilidade da API de etiologia"
```

---

### Task 6: Add historical schema guards for 2019-style rows

**Files:**
- Modify: `guilherme/tests/unit/test_etiology.py`
- Modify: `guilherme/tests/unit/test_ingest.py` only if necessary

**Interfaces:**
- Pipeline must not require SARS-CoV-2 fields to exist.
- `ETIOLOGIA_DETALHADA` returns `NAO_IDENTIFICADA` when no configured positive laboratory field exists.

- [ ] **Step 1: Add 2019-compatible etiology test**

```python
def test_2019_style_row_does_not_require_covid_fields():
    row = make_row(
        CLASSI_FIN=1,
        PCR_FLUAS=1,
    )

    assert normalize_final_classification(
        row["CLASSI_FIN"]
    ) == "INFLUENZA"
    assert normalize_detailed_etiology(row) == "Influenza A"
```

- [ ] **Step 2: Add zero-PCR-column DataFrame test**

```python
def test_add_etiology_columns_accepts_dataframe_without_pcr_columns():
    df = pd.DataFrame(
        {
            "CLASSI_FIN": [1, 4, None],
        }
    )

    result = add_etiology_columns(df)

    assert result["CLASSIFICACAO_FINAL_NORMALIZADA"].tolist() == [
        "INFLUENZA",
        "NAO_ESPECIFICADO",
        "AUSENTE",
    ]
    assert result["ETIOLOGIA_DETALHADA"].tolist() == [
        "NAO_IDENTIFICADA",
        "NAO_IDENTIFICADA",
        "NAO_IDENTIFICADA",
    ]
```

- [ ] **Step 3: Run tests**

```powershell
python -m pytest tests/unit/test_etiology.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add tests/unit/test_etiology.py
git commit -m "test: cobre schemas historicos de etiologia"
```

---

### Task 7: Document semantics and remove stale terminology

**Files:**
- Modify: `guilherme/README.md`
- Inspect/search:
  - `guilherme/docs/superpowers/specs/`
  - `guilherme/docs/superpowers/plans/`

**Interfaces:**
- README must define:
  - `CLASSIFICACAO_FINAL_NORMALIZADA`
  - `ETIOLOGIA_DETALHADA`
  - `/api/v1/etiologia` = distribuição da etiologia laboratorial detalhada.

- [ ] **Step 1: Search for stale terminology**

From `guilherme`:

```powershell
Get-ChildItem -Recurse -File | Select-String "ETIOLOGIA_NORMALIZADA|normalize_etiology|add_etiology_column"
```

Expected after code migration: only historical design/plan docs may still contain old terminology.

- [ ] **Step 2: Update README**

Add a concise section:

```markdown
### Classificação final e etiologia

O pipeline preserva `CLASSI_FIN` e gera duas variáveis distintas:

- `CLASSIFICACAO_FINAL_NORMALIZADA`: classificação final oficial do caso;
- `ETIOLOGIA_DETALHADA`: agente identificado pelos campos laboratoriais disponíveis.

A ausência histórica de uma coluna laboratorial não é interpretada como resultado negativo.

O endpoint `GET /api/v1/etiologia` agrega `ETIOLOGIA_DETALHADA`.
```

- [ ] **Step 3: Do not rewrite historical planning docs**

Historical specs/plans may describe the old design and should remain as project history unless they explicitly claim to be current normative documentation. The approved 2026-08-14 spec supersedes them.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: documenta classificacao e etiologia separadas"
```

---

### Task 8: Full verification and schema reconnaissance gate

**Files:**
- No production code changes expected.
- Optional generated local notes must remain outside Git unless intentionally documented.

**Interfaces:**
- Full suite must pass before PR.
- Real 2019/2026 data must only be inspected, not bulk-reprocessed yet.

- [ ] **Step 1: Run full test suite**

```powershell
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Check no old analytical column remains in active code**

```powershell
Get-ChildItem src,tests -Recurse -File | Select-String "ETIOLOGIA_NORMALIZADA|normalize_etiology|add_etiology_column"
```

Expected: no matches in active `src/` or `tests/`.

- [ ] **Step 3: Inspect Git diff**

```powershell
git status
git diff main...HEAD -- src tests README.md
```

Verify:
- no raw CSV/Parquet included;
- no unrelated files;
- no changes outside `guilherme`;
- source `CLASSI_FIN`/PCR fields remain preserved.

- [ ] **Step 4: Validate schema samples before bulk processing**

For one 2019 CSV and one 2026 CSV, run a local header inspection only:

```powershell
python -c "import pandas as pd; from pathlib import Path; p=Path(r'data/raw/2019/ARQUIVO.csv'); df=pd.read_csv(p, sep=';', encoding='latin-1', nrows=5, low_memory=False); print(sorted(df.columns))"
```

and:

```powershell
python -c "import pandas as pd; from pathlib import Path; p=Path(r'data/raw/2026/ARQUIVO.csv'); df=pd.read_csv(p, sep=';', encoding='latin-1', nrows=5, low_memory=False); print(sorted(df.columns))"
```

Replace only `ARQUIVO.csv` with the actual downloaded filenames.

Check specifically for:
- `CLASSI_FIN`
- `PCR_SARS2`
- `PCR_FLUAS`
- `PCR_FLUBS`
- `PCR_VSR`
- `PCR_ADENO`
- `PCR_PARA1`
- `PCR_PARA2`
- `PCR_PARA3`
- `PCR_PARA4`
- `PCR_METAP`
- `PCR_BOCA`
- `PCR_RINO`

Do **not** infer a negative result from absence.

- [ ] **Step 5: Re-run full tests after schema reconnaissance**

```powershell
python -m pytest -v
```

Expected: PASS.

- [ ] **Step 6: Final commit only if reconnaissance required documentation changes**

If no code/docs changed, do not create an empty commit.

- [ ] **Step 7: Push branch**

```powershell
git push -u origin feature/etiology-normalization
```

- [ ] **Step 8: Open PR**

Suggested title:

```text
feat: separa classificacao final e etiologia SRAG
```

Suggested scope in PR body:
- official `CLASSI_FIN` normalization;
- laboratory etiology separated;
- repository/API compatibility;
- historical schema guards;
- no bulk data reprocessing in this PR.

---

## Verification Checklist

Before considering the implementation complete:

- [ ] `python -m pytest -v` passes.
- [ ] `CLASSI_FIN=1..5` follows the approved mapping.
- [ ] unexpected `CLASSI_FIN` does not become `IGNORADO` automatically.
- [ ] `CLASSIFICACAO_FINAL_NORMALIZADA` depends only on `CLASSI_FIN`.
- [ ] `ETIOLOGIA_DETALHADA` depends only on laboratory fields.
- [ ] conflicting classification/lab values are preserved independently.
- [ ] missing PCR columns are accepted.
- [ ] original columns are preserved.
- [ ] repository filter `etiologia` uses `ETIOLOGIA_DETALHADA`.
- [ ] `/api/v1/etiologia` retains its response shape.
- [ ] README explains the new semantics.
- [ ] no CSV, Parquet, or generated quality report is committed.
- [ ] schema samples from 2019 and 2026 are inspected before bulk reprocessing.
