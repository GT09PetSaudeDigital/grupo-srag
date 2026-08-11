# SRAG Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir um pipeline reproduzível para validar, normalizar e transformar os CSVs do SIVEP-Gripe de 2019–2026 em Parquet particionado, com relatórios de qualidade por ano.

**Architecture:** A implementação ficará isolada em `guilherme/src/srag_api/data/`, com funções pequenas e testáveis para schema, idade, valores categóricos, etiologia, qualidade e escrita Parquet. Scripts em `guilherme/scripts/` apenas orquestram essas funções; dados RAW nunca são modificados.

**Tech Stack:** Python, pandas, pyarrow, pytest, pathlib, json, dataclasses/typing da biblioteca padrão.

## Global Constraints

- Alterações deste trabalho devem permanecer no repositório `GT09PetSaudeDigital/grupo-srag`.
- O escopo principal é a pasta `guilherme/`.
- Não modificar nenhum arquivo ou repositório relacionado a `infosetecinco/agrocifra`.
- Commits e pushes serão executados pelo usuário.
- A implementação não deve alterar dados RAW.
- A API não expõe registros individuais na V1.
- A API não aceita SQL arbitrário.
- MCP não consulta DuckDB diretamente.
- ML não deve compartilhar fitted preprocessing entre treino e teste.
- Dados suportados: 2019–2026.
- A ingestão deve ser incremental por ano.
- Valores SIM, NÃO, IGNORADO e AUSENTE devem ser distinguíveis sempre que possível.
- `NU_IDADE_N` não pode ser tratado isoladamente como idade em anos; a normalização deve usar `TP_IDADE`.
- A V1 deve gerar relatório de qualidade por ano.

---

## File Structure Locked for This Plan

**Create:**

```text
guilherme/
├── pyproject.toml
├── src/
│   └── srag_api/
│       ├── __init__.py
│       ├── config.py
│       └── data/
│           ├── __init__.py
│           ├── schema.py
│           ├── clean.py
│           ├── etiology.py
│           ├── quality.py
│           └── ingest.py
├── scripts/
│   ├── __init__.py
│   ├── ingest_year.py
│   └── ingest_all.py
├── tests/
│   ├── fixtures/
│   │   └── sample_srag.csv
│   └── unit/
│       ├── test_schema.py
│       ├── test_age_normalization.py
│       ├── test_clean.py
│       ├── test_etiology.py
│       ├── test_quality.py
│       ├── test_ingest.py
│       └── test_batch_ingest.py
└── data/
    ├── raw/.gitkeep
    ├── parquet/.gitkeep
    └── quality/.gitkeep
```

**Do not modify in this plan:**

```text
guilherme/analise_srag_pr.py
```

O script antigo permanece intacto nesta fase para preservar a análise histórica enquanto o novo pipeline é construído e testado.

---

### Task 1: Package Foundation and Test Harness

**Files:**
- Create: `guilherme/pyproject.toml`
- Create: `guilherme/src/srag_api/__init__.py`
- Create: `guilherme/src/srag_api/data/__init__.py`
- Create: `guilherme/tests/unit/test_schema.py`

**Interfaces:**
- Consumes: nada.
- Produces: pacote importável `srag_api`; configuração de pytest; dependências de runtime para o pipeline.

- [ ] **Step 1: Write the failing import test**

Create `guilherme/tests/unit/test_schema.py`:

```python
def test_package_imports():
    import srag_api

    assert srag_api.__name__ == "srag_api"
```

- [ ] **Step 2: Run the test to verify the package is not configured yet**

Run from `guilherme/`:

```powershell
python -m pytest tests/unit/test_schema.py::test_package_imports -v
```

Expected: FAIL with an import error if the package is not yet installed/configured.

- [ ] **Step 3: Add project configuration**

Create `guilherme/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "srag-api"
version = "0.1.0"
description = "Pipeline e API epidemiologica para dados SRAG/SIVEP-Gripe"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `guilherme/src/srag_api/__init__.py`:

```python
"""SRAG epidemiological data platform."""
```

Create `guilherme/src/srag_api/data/__init__.py`:

```python
"""Data ingestion and normalization utilities."""
```

- [ ] **Step 4: Install the project in editable mode**

Run from `guilherme/`:

```powershell
python -m pip install -e ".[dev]"
```

Expected: installation succeeds.

- [ ] **Step 5: Run the test again**

```powershell
python -m pytest tests/unit/test_schema.py::test_package_imports -v
```

Expected: PASS.

- [ ] **Step 6: Review before commit**

From repository root:

```powershell
git status
git diff -- guilherme/pyproject.toml guilherme/src/srag_api/__init__.py guilherme/src/srag_api/data/__init__.py guilherme/tests/unit/test_schema.py
```

- [ ] **Step 7: Commit manually**

```powershell
git add guilherme/pyproject.toml guilherme/src/srag_api/__init__.py guilherme/src/srag_api/data/__init__.py guilherme/tests/unit/test_schema.py
git commit -m "chore: cria estrutura inicial do pipeline SRAG"
```

---

### Task 2: Central Configuration and Required Schema

**Files:**
- Create: `guilherme/src/srag_api/config.py`
- Create: `guilherme/src/srag_api/data/schema.py`
- Modify: `guilherme/tests/unit/test_schema.py`

**Interfaces:**
- Consumes: pacote criado na Task 1.
- Produces:
  - `SUPPORTED_YEARS: tuple[int, ...]`
  - `AGE_BANDS: tuple[tuple[float, float | None, str], ...]`
  - `ESSENTIAL_COLUMNS: frozenset[str]`
  - `normalize_column_names(df: pd.DataFrame) -> pd.DataFrame`
  - `validate_required_columns(df: pd.DataFrame) -> None`

- [ ] **Step 1: Add failing tests for configuration and schema**

Replace `guilherme/tests/unit/test_schema.py` with:

```python
import pandas as pd
import pytest

from srag_api.config import AGE_BANDS, SUPPORTED_YEARS
from srag_api.data.schema import (
    ESSENTIAL_COLUMNS,
    normalize_column_names,
    validate_required_columns,
)


def test_supported_years_are_2019_through_2026():
    assert SUPPORTED_YEARS == tuple(range(2019, 2027))


def test_age_bands_have_expected_labels():
    labels = [label for _, _, label in AGE_BANDS]
    assert labels == ["<1", "1-4", "5-11", "12-17", "18-29", "30-44", "45-59", "60-74", "75+"]


def test_normalize_column_names_strips_and_uppercases():
    df = pd.DataFrame(columns=[" tp_idade ", "Nu_Idade_N", " evolucao"])
    result = normalize_column_names(df)
    assert list(result.columns) == ["TP_IDADE", "NU_IDADE_N", "EVOLUCAO"]


def test_validate_required_columns_accepts_minimum_schema():
    df = pd.DataFrame(columns=sorted(ESSENTIAL_COLUMNS))
    validate_required_columns(df)


def test_validate_required_columns_reports_missing_fields():
    df = pd.DataFrame(columns=["TP_IDADE", "NU_IDADE_N"])
    with pytest.raises(ValueError, match="Colunas essenciais ausentes"):
        validate_required_columns(df)
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
python -m pytest tests/unit/test_schema.py -v
```

Expected: FAIL because `config.py` and `schema.py` do not exist.

- [ ] **Step 3: Implement configuration**

Create `guilherme/src/srag_api/config.py`:

```python
SUPPORTED_YEARS = tuple(range(2019, 2027))

AGE_BANDS = (
    (0.0, 1.0, "<1"),
    (1.0, 5.0, "1-4"),
    (5.0, 12.0, "5-11"),
    (12.0, 18.0, "12-17"),
    (18.0, 30.0, "18-29"),
    (30.0, 45.0, "30-44"),
    (45.0, 60.0, "45-59"),
    (60.0, 75.0, "60-74"),
    (75.0, None, "75+"),
)
```

- [ ] **Step 4: Implement schema validation**

Create `guilherme/src/srag_api/data/schema.py`:

```python
import pandas as pd


ESSENTIAL_COLUMNS = frozenset(
    {
        "TP_IDADE",
        "NU_IDADE_N",
        "SG_UF",
        "ID_MUNICIP",
        "EVOLUCAO",
        "UTI",
    }
)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(column).strip().upper() for column in result.columns]
    return result


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(ESSENTIAL_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "Colunas essenciais ausentes: " + ", ".join(missing)
        )
```

- [ ] **Step 5: Run tests**

```powershell
python -m pytest tests/unit/test_schema.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit manually**

```powershell
git add guilherme/src/srag_api/config.py guilherme/src/srag_api/data/schema.py guilherme/tests/unit/test_schema.py
git commit -m "feat: adiciona configuracao e validacao de schema SRAG"
```

---

### Task 3: Correct Age Normalization

**Files:**
- Create: `guilherme/src/srag_api/data/clean.py`
- Create: `guilherme/tests/unit/test_age_normalization.py`

**Interfaces:**
- Consumes: `AGE_BANDS`.
- Produces:
  - `normalize_age(tp_idade: object, nu_idade_n: object) -> float | None`
  - `classify_age_band(age_years: float | None) -> str | None`
  - `add_normalized_age_columns(df: pd.DataFrame) -> pd.DataFrame`

**Domain rule:**
- `TP_IDADE == 1`: idade informada em dias.
- `TP_IDADE == 2`: idade informada em meses.
- `TP_IDADE == 3`: idade informada em anos.
- valores inválidos, negativos ou idade equivalente acima de 120 anos retornam `None`.

- [ ] **Step 1: Write failing age tests**

Create `guilherme/tests/unit/test_age_normalization.py`:

```python
import math

import pandas as pd

from srag_api.data.clean import (
    add_normalized_age_columns,
    classify_age_band,
    normalize_age,
)


def test_normalize_age_from_days():
    assert math.isclose(normalize_age(1, 20), 20 / 365.25, rel_tol=1e-6)


def test_normalize_age_from_months():
    assert math.isclose(normalize_age(2, 18), 1.5, rel_tol=1e-6)


def test_normalize_age_from_years():
    assert normalize_age(3, 67) == 67.0


def test_normalize_age_rejects_negative_value():
    assert normalize_age(3, -1) is None


def test_normalize_age_rejects_unknown_unit():
    assert normalize_age(9, 30) is None


def test_normalize_age_rejects_more_than_120_years():
    assert normalize_age(3, 121) is None


def test_classify_age_band():
    assert classify_age_band(0.5) == "<1"
    assert classify_age_band(4.0) == "1-4"
    assert classify_age_band(17.0) == "12-17"
    assert classify_age_band(60.0) == "60-74"
    assert classify_age_band(80.0) == "75+"


def test_add_normalized_age_columns():
    df = pd.DataFrame(
        {
            "TP_IDADE": [1, 2, 3],
            "NU_IDADE_N": [30, 18, 75],
        }
    )
    result = add_normalized_age_columns(df)

    assert "IDADE_ANOS" in result.columns
    assert "FAIXA_ETARIA" in result.columns
    assert result["FAIXA_ETARIA"].tolist() == ["<1", "1-4", "75+"]
```

- [ ] **Step 2: Run the tests and verify failure**

```powershell
python -m pytest tests/unit/test_age_normalization.py -v
```

Expected: FAIL because `clean.py` does not exist.

- [ ] **Step 3: Implement age normalization**

Create `guilherme/src/srag_api/data/clean.py`:

```python
from __future__ import annotations

import math

import pandas as pd

from srag_api.config import AGE_BANDS


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def normalize_age(tp_idade: object, nu_idade_n: object) -> float | None:
    unit = _to_float(tp_idade)
    value = _to_float(nu_idade_n)

    if unit is None or value is None or value < 0:
        return None

    if unit == 1:
        age_years = value / 365.25
    elif unit == 2:
        age_years = value / 12.0
    elif unit == 3:
        age_years = value
    else:
        return None

    if age_years > 120:
        return None

    return age_years


def classify_age_band(age_years: float | None) -> str | None:
    if age_years is None:
        return None

    for lower, upper, label in AGE_BANDS:
        if age_years >= lower and (upper is None or age_years < upper):
            return label

    return None


def add_normalized_age_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["IDADE_ANOS"] = [
        normalize_age(tp_idade, nu_idade_n)
        for tp_idade, nu_idade_n in zip(
            result["TP_IDADE"],
            result["NU_IDADE_N"],
        )
    ]
    result["FAIXA_ETARIA"] = result["IDADE_ANOS"].map(classify_age_band)
    return result
```

- [ ] **Step 4: Run the age tests**

```powershell
python -m pytest tests/unit/test_age_normalization.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit manually**

```powershell
git add guilherme/src/srag_api/data/clean.py guilherme/tests/unit/test_age_normalization.py
git commit -m "feat: normaliza idade SRAG usando TP_IDADE"
```

---

### Task 4: Normalize Categorical Values Without Losing Ignored Status

**Files:**
- Modify: `guilherme/src/srag_api/data/clean.py`
- Create: `guilherme/tests/unit/test_clean.py`

**Interfaces:**
- Consumes: dataframe with SIVEP-Gripe codes.
- Produces:
  - `normalize_yes_no(value: object) -> str`
  - `normalize_outcome(value: object) -> str`
  - `add_core_normalized_columns(df: pd.DataFrame) -> pd.DataFrame`

**Canonical values:**
- yes/no field: `"SIM"`, `"NAO"`, `"IGNORADO"`, `"AUSENTE"`.
- evolution field: `"CURA"`, `"OBITO_SRAG"`, `"OBITO_OUTRAS_CAUSAS"`, `"IGNORADO"`, `"AUSENTE"`, `"OUTRO"`.

- [ ] **Step 1: Write failing tests**

Create `guilherme/tests/unit/test_clean.py`:

```python
import pandas as pd

from srag_api.data.clean import (
    add_core_normalized_columns,
    normalize_outcome,
    normalize_yes_no,
)


def test_normalize_yes_no_codes():
    assert normalize_yes_no(1) == "SIM"
    assert normalize_yes_no(2) == "NAO"
    assert normalize_yes_no(9) == "IGNORADO"
    assert normalize_yes_no(None) == "AUSENTE"


def test_normalize_outcome_codes():
    assert normalize_outcome(1) == "CURA"
    assert normalize_outcome(2) == "OBITO_SRAG"
    assert normalize_outcome(3) == "OBITO_OUTRAS_CAUSAS"
    assert normalize_outcome(9) == "IGNORADO"
    assert normalize_outcome(None) == "AUSENTE"
    assert normalize_outcome(7) == "OUTRO"


def test_add_core_normalized_columns_preserves_source_fields():
    df = pd.DataFrame(
        {
            "EVOLUCAO": [1, 2, 9],
            "UTI": [2, 1, 9],
        }
    )
    result = add_core_normalized_columns(df)

    assert result["EVOLUCAO"].tolist() == [1, 2, 9]
    assert result["UTI"].tolist() == [2, 1, 9]
    assert result["DESFECHO_NORMALIZADO"].tolist() == ["CURA", "OBITO_SRAG", "IGNORADO"]
    assert result["FOI_UTI"].tolist() == ["NAO", "SIM", "IGNORADO"]
    assert result["OBITO_SRAG"].tolist() == [False, True, False]
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest tests/unit/test_clean.py -v
```

Expected: FAIL because the functions are not defined.

- [ ] **Step 3: Add categorical normalization**

Append to `guilherme/src/srag_api/data/clean.py`:

```python
def normalize_yes_no(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "AUSENTE"
    if number == 1:
        return "SIM"
    if number == 2:
        return "NAO"
    if number == 9:
        return "IGNORADO"
    return "IGNORADO"


def normalize_outcome(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "AUSENTE"
    if number == 1:
        return "CURA"
    if number == 2:
        return "OBITO_SRAG"
    if number == 3:
        return "OBITO_OUTRAS_CAUSAS"
    if number == 9:
        return "IGNORADO"
    return "OUTRO"


def add_core_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["DESFECHO_NORMALIZADO"] = result["EVOLUCAO"].map(normalize_outcome)
    result["FOI_UTI"] = result["UTI"].map(normalize_yes_no)
    result["OBITO_SRAG"] = result["DESFECHO_NORMALIZADO"].eq("OBITO_SRAG")
    return result
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/unit/test_clean.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit manually**

```powershell
git add guilherme/src/srag_api/data/clean.py guilherme/tests/unit/test_clean.py
git commit -m "feat: preserva estados ignorados na normalizacao SRAG"
```

---

### Task 5: Etiology Normalization

**Files:**
- Create: `guilherme/src/srag_api/data/etiology.py`
- Create: `guilherme/tests/unit/test_etiology.py`

**Interfaces:**
- Consumes: colunas laboratoriais/classificatórias existentes no dataframe.
- Produces:
  - `normalize_etiology(row: pd.Series) -> str`
  - `add_etiology_column(df: pd.DataFrame) -> pd.DataFrame`

**Precedence rule for the first implementation:**
1. `CLASSI_FIN == 5` → `"COVID-19"`.
2. `PCR_SARS2 == 1` → `"COVID-19"`.
3. `PCR_FLUAS == 1` → `"Influenza A"`.
4. `PCR_FLUBS == 1` → `"Influenza B"`.
5. `PCR_VSR == 1` → `"VSR"`.
6. any respiratory-virus positive flag from the configured list → `"Outros virus respiratorios"`.
7. `CLASSI_FIN` present with another known non-viral final classification → `"Outro agente"`.
8. explicit ignored final classification → `"Ignorado"`.
9. otherwise → `"Nao identificado"`.

This mapping must remain isolated in this module so it can be revised against the official SIVEP-Gripe dictionary without changing API code.

- [ ] **Step 1: Write failing etiology tests**

Create `guilherme/tests/unit/test_etiology.py`:

```python
import pandas as pd

from srag_api.data.etiology import add_etiology_column, normalize_etiology


def make_row(**kwargs):
    return pd.Series(kwargs)


def test_covid_has_priority_when_final_classification_is_covid():
    row = make_row(CLASSI_FIN=5, PCR_FLUAS=1)
    assert normalize_etiology(row) == "COVID-19"


def test_influenza_a():
    row = make_row(CLASSI_FIN=None, PCR_FLUAS=1)
    assert normalize_etiology(row) == "Influenza A"


def test_influenza_b():
    row = make_row(CLASSI_FIN=None, PCR_FLUBS=1)
    assert normalize_etiology(row) == "Influenza B"


def test_vsr():
    row = make_row(CLASSI_FIN=None, PCR_VSR=1)
    assert normalize_etiology(row) == "VSR"


def test_unknown_is_not_identified():
    row = make_row(CLASSI_FIN=None)
    assert normalize_etiology(row) == "Nao identificado"


def test_add_etiology_column_keeps_original_columns():
    df = pd.DataFrame(
        {
            "CLASSI_FIN": [5, None],
            "PCR_FLUAS": [None, 1],
        }
    )
    result = add_etiology_column(df)

    assert "CLASSI_FIN" in result.columns
    assert result["ETIOLOGIA_NORMALIZADA"].tolist() == ["COVID-19", "Influenza A"]
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest tests/unit/test_etiology.py -v
```

Expected: FAIL because `etiology.py` does not exist.

- [ ] **Step 3: Implement isolated mapping**

Create `guilherme/src/srag_api/data/etiology.py`:

```python
from __future__ import annotations

import pandas as pd


OTHER_RESPIRATORY_FLAGS = (
    "PCR_ADENO",
    "PCR_PARA1",
    "PCR_PARA2",
    "PCR_PARA3",
    "PCR_PARA4",
    "PCR_METAP",
    "PCR_BOCA",
    "PCR_RINO",
)


def _is_positive(row: pd.Series, field: str) -> bool:
    return field in row.index and row.get(field) == 1


def normalize_etiology(row: pd.Series) -> str:
    final_classification = row.get("CLASSI_FIN")

    if final_classification == 5 or _is_positive(row, "PCR_SARS2"):
        return "COVID-19"
    if _is_positive(row, "PCR_FLUAS"):
        return "Influenza A"
    if _is_positive(row, "PCR_FLUBS"):
        return "Influenza B"
    if _is_positive(row, "PCR_VSR"):
        return "VSR"
    if any(_is_positive(row, field) for field in OTHER_RESPIRATORY_FLAGS):
        return "Outros virus respiratorios"
    if final_classification == 9:
        return "Ignorado"
    if pd.notna(final_classification):
        return "Outro agente"
    return "Nao identificado"


def add_etiology_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ETIOLOGIA_NORMALIZADA"] = result.apply(normalize_etiology, axis=1)
    return result
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/unit/test_etiology.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit manually**

```powershell
git add guilherme/src/srag_api/data/etiology.py guilherme/tests/unit/test_etiology.py
git commit -m "feat: adiciona normalizacao etiologica SRAG"
```

---

### Task 6: Quality Report

**Files:**
- Create: `guilherme/src/srag_api/data/quality.py`
- Create: `guilherme/tests/unit/test_quality.py`

**Interfaces:**
- Consumes: dataframe raw e dataframe processado.
- Produces:
  - `build_quality_report(raw_df: pd.DataFrame, processed_df: pd.DataFrame, year: int) -> dict[str, object]`
  - `write_quality_report(report: dict[str, object], output_path: Path) -> None`

- [ ] **Step 1: Write failing quality tests**

Create `guilherme/tests/unit/test_quality.py`:

```python
import json

import pandas as pd

from srag_api.data.quality import build_quality_report, write_quality_report


def test_quality_report_counts_missing_and_duplicates(tmp_path):
    raw = pd.DataFrame(
        {
            "TP_IDADE": [3, 3, 3],
            "NU_IDADE_N": [70, 70, None],
            "SG_UF": ["PR", "PR", None],
            "ID_MUNICIP": ["CURITIBA", "CURITIBA", None],
            "EVOLUCAO": [2, 2, 9],
            "UTI": [1, 1, None],
        }
    )
    processed = raw.drop_duplicates().copy()
    processed["IDADE_ANOS"] = [70.0, None]
    processed["ETIOLOGIA_NORMALIZADA"] = ["COVID-19", "Nao identificado"]

    report = build_quality_report(raw, processed, 2025)

    assert report["ano"] == 2025
    assert report["registros_recebidos"] == 3
    assert report["registros_processados"] == 2
    assert report["duplicados"] == 1
    assert report["idade_ausente"] == 1
    assert report["municipio_ausente"] == 1
    assert report["etiologia_nao_identificada"] == 1

    output = tmp_path / "quality_2025.json"
    write_quality_report(report, output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["ano"] == 2025
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest tests/unit/test_quality.py -v
```

Expected: FAIL because `quality.py` does not exist.

- [ ] **Step 3: Implement report generation**

Create `guilherme/src/srag_api/data/quality.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _missing_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return len(df)
    return int(df[column].isna().sum())


def build_quality_report(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    year: int,
) -> dict[str, object]:
    etiology = processed_df.get(
        "ETIOLOGIA_NORMALIZADA",
        pd.Series(index=processed_df.index, dtype="object"),
    )

    return {
        "ano": year,
        "registros_recebidos": int(len(raw_df)),
        "registros_processados": int(len(processed_df)),
        "duplicados": int(raw_df.duplicated().sum()),
        "idade_ausente": _missing_count(processed_df, "IDADE_ANOS"),
        "sexo_ausente": _missing_count(processed_df, "CS_SEXO"),
        "municipio_ausente": _missing_count(processed_df, "ID_MUNICIP"),
        "evolucao_ausente": _missing_count(processed_df, "EVOLUCAO"),
        "uti_ausente": _missing_count(processed_df, "UTI"),
        "etiologia_nao_identificada": int(
            (etiology == "Nao identificado").sum()
        ),
    }


def write_quality_report(
    report: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/unit/test_quality.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit manually**

```powershell
git add guilherme/src/srag_api/data/quality.py guilherme/tests/unit/test_quality.py
git commit -m "feat: gera relatorio de qualidade por ano"
```

---

### Task 7: End-to-End Dataframe Transformation

**Files:**
- Modify: `guilherme/src/srag_api/data/clean.py`
- Create: `guilherme/tests/fixtures/sample_srag.csv`
- Create: `guilherme/tests/unit/test_ingest.py`
- Create: `guilherme/src/srag_api/data/ingest.py`

**Interfaces:**
- Consumes:
  - `normalize_column_names`
  - `validate_required_columns`
  - `add_normalized_age_columns`
  - `add_core_normalized_columns`
  - `add_etiology_column`
  - `build_quality_report`
- Produces:
  - `read_srag_csv(path: Path) -> pd.DataFrame`
  - `transform_srag_dataframe(df: pd.DataFrame, year: int) -> pd.DataFrame`
  - `write_year_parquet(df: pd.DataFrame, base_dir: Path, year: int) -> Path`
  - `ingest_year(input_path: Path, parquet_root: Path, quality_root: Path, year: int, force: bool = False) -> Path`

- [ ] **Step 1: Create a deterministic fixture**

Create `guilherme/tests/fixtures/sample_srag.csv`:

```csv
TP_IDADE;NU_IDADE_N;SG_UF;ID_MUNICIP;CO_MUN_RES;EVOLUCAO;UTI;CS_SEXO;CLASSI_FIN;PCR_FLUAS;PCR_FLUBS;PCR_VSR;PCR_SARS2
3;67;PR;CURITIBA;410690;1;2;M;;1;;;
2;18;PR;LONDRINA;411370;2;1;F;5;;;;1
1;30;MT;PRIMAVERA DO LESTE;510704;9;9;M;;;;1;
```

- [ ] **Step 2: Write failing ingest tests**

Create `guilherme/tests/unit/test_ingest.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from srag_api.data.ingest import (
    ingest_year,
    read_srag_csv,
    transform_srag_dataframe,
    write_year_parquet,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "sample_srag.csv"


def test_read_srag_csv_reads_semicolon_file():
    df = read_srag_csv(FIXTURE)
    assert len(df) == 3
    assert "TP_IDADE" in df.columns


def test_transform_srag_dataframe_adds_analytic_columns():
    df = read_srag_csv(FIXTURE)
    result = transform_srag_dataframe(df, 2025)

    assert result["ANO"].tolist() == [2025, 2025, 2025]
    assert "IDADE_ANOS" in result.columns
    assert "FAIXA_ETARIA" in result.columns
    assert "ETIOLOGIA_NORMALIZADA" in result.columns
    assert "DESFECHO_NORMALIZADO" in result.columns
    assert "FOI_UTI" in result.columns
    assert "OBITO_SRAG" in result.columns
    assert "CODIGO_MUNICIPIO" in result.columns
    assert "MUNICIPIO" in result.columns
    assert "UF" in result.columns


def test_write_year_parquet_uses_partition_directory(tmp_path):
    df = transform_srag_dataframe(read_srag_csv(FIXTURE), 2025)
    path = write_year_parquet(df, tmp_path / "parquet", 2025)

    assert path == tmp_path / "parquet" / "srag" / "ano=2025" / "srag.parquet"
    assert path.exists()

    loaded = pd.read_parquet(path)
    assert len(loaded) == 3


def test_ingest_year_writes_parquet_and_quality_report(tmp_path):
    path = ingest_year(
        input_path=FIXTURE,
        parquet_root=tmp_path / "parquet",
        quality_root=tmp_path / "quality",
        year=2025,
    )

    assert path.exists()
    assert (tmp_path / "quality" / "quality_2025.json").exists()


def test_ingest_year_is_incremental_without_force(tmp_path):
    ingest_year(
        input_path=FIXTURE,
        parquet_root=tmp_path / "parquet",
        quality_root=tmp_path / "quality",
        year=2025,
    )

    with pytest.raises(FileExistsError, match="2025"):
        ingest_year(
            input_path=FIXTURE,
            parquet_root=tmp_path / "parquet",
            quality_root=tmp_path / "quality",
            year=2025,
            force=False,
        )
```

- [ ] **Step 3: Run tests and verify failure**

```powershell
python -m pytest tests/unit/test_ingest.py -v
```

Expected: FAIL because ingestion functions are not defined.

- [ ] **Step 4: Add geographic normalization helper**

Append to `guilherme/src/srag_api/data/clean.py`:

```python
def add_geography_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["UF"] = result["SG_UF"].astype("string").str.strip().str.upper()
    result["MUNICIPIO"] = result["ID_MUNICIP"].astype("string").str.strip().str.upper()

    if "CO_MUN_RES" in result.columns:
        result["CODIGO_MUNICIPIO"] = (
            pd.to_numeric(result["CO_MUN_RES"], errors="coerce")
            .astype("Int64")
        )
    else:
        result["CODIGO_MUNICIPIO"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

    return result
```

- [ ] **Step 5: Implement ingestion pipeline**

Create `guilherme/src/srag_api/data/ingest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from srag_api.config import SUPPORTED_YEARS
from srag_api.data.clean import (
    add_core_normalized_columns,
    add_geography_columns,
    add_normalized_age_columns,
)
from srag_api.data.etiology import add_etiology_column
from srag_api.data.quality import build_quality_report, write_quality_report
from srag_api.data.schema import normalize_column_names, validate_required_columns


def read_srag_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        low_memory=False,
    )


def transform_srag_dataframe(
    df: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    if year not in SUPPORTED_YEARS:
        raise ValueError(f"Ano nao suportado: {year}")

    result = normalize_column_names(df)
    validate_required_columns(result)
    result = result.drop_duplicates().copy()
    result = add_normalized_age_columns(result)
    result = add_core_normalized_columns(result)
    result = add_geography_columns(result)
    result = add_etiology_column(result)
    result["ANO"] = year

    return result


def write_year_parquet(
    df: pd.DataFrame,
    base_dir: Path,
    year: int,
) -> Path:
    output_path = (
        base_dir
        / "srag"
        / f"ano={year}"
        / "srag.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def ingest_year(
    input_path: Path,
    parquet_root: Path,
    quality_root: Path,
    year: int,
    force: bool = False,
) -> Path:
    output_path = (
        parquet_root
        / "srag"
        / f"ano={year}"
        / "srag.parquet"
    )

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Ano {year} ja foi processado. Use force=True para reprocessar."
        )

    raw_df = read_srag_csv(input_path)
    processed_df = transform_srag_dataframe(raw_df, year)

    parquet_path = write_year_parquet(
        processed_df,
        parquet_root,
        year,
    )

    report = build_quality_report(
        raw_df,
        processed_df,
        year,
    )
    write_quality_report(
        report,
        quality_root / f"quality_{year}.json",
    )

    return parquet_path
```

- [ ] **Step 6: Run ingestion tests**

```powershell
python -m pytest tests/unit/test_ingest.py -v
```

Expected: all PASS.

- [ ] **Step 7: Run all pipeline tests**

```powershell
python -m pytest tests/unit -v
```

Expected: all PASS.

- [ ] **Step 8: Commit manually**

```powershell
git add guilherme/src/srag_api/data/clean.py guilherme/src/srag_api/data/ingest.py guilherme/tests/fixtures/sample_srag.csv guilherme/tests/unit/test_ingest.py
git commit -m "feat: implementa pipeline CSV para Parquet"
```

---

### Task 8: Year CLI

**Files:**
- Create: `guilherme/scripts/ingest_year.py`
- Modify: `guilherme/tests/unit/test_ingest.py`

**Interfaces:**
- Consumes: `ingest_year(...)`.
- Produces: CLI `python scripts/ingest_year.py --year YEAR --input PATH [--force]`.

- [ ] **Step 1: Add validation test for unsupported years**

Append to `guilherme/tests/unit/test_ingest.py`:

```python
def test_transform_rejects_unsupported_year():
    df = read_srag_csv(FIXTURE)

    with pytest.raises(ValueError, match="Ano nao suportado"):
        transform_srag_dataframe(df, 2018)
```

- [ ] **Step 2: Run test**

```powershell
python -m pytest tests/unit/test_ingest.py::test_transform_rejects_unsupported_year -v
```

Expected: PASS because the domain validation already exists.

- [ ] **Step 3: Implement CLI**

Create `guilherme/scripts/ingest_year.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from srag_api.data.ingest import ingest_year


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processa um ano do SIVEP-Gripe para Parquet."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output = ingest_year(
        input_path=args.input,
        parquet_root=Path("data/parquet"),
        quality_root=Path("data/quality"),
        year=args.year,
        force=args.force,
    )
    print(f"Parquet gerado: {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-test CLI using fixture**

Run from `guilherme/`:

```powershell
python scripts/ingest_year.py --year 2025 --input tests/fixtures/sample_srag.csv --force
```

Expected output contains:

```text
Parquet gerado:
```

Verify:

```powershell
Get-ChildItem data\parquet\sragno=2025
Get-Content data\quality\quality_2025.json
```

- [ ] **Step 5: Remove generated fixture output from the working tree if it is not intended for Git**

```powershell
Remove-Item -Recurse -Force data\parquet\sragno=2025
Remove-Item -Force data\quality\quality_2025.json
```

- [ ] **Step 6: Commit manually**

```powershell
git add guilherme/scripts/ingest_year.py guilherme/tests/unit/test_ingest.py
git commit -m "feat: adiciona CLI de ingestao por ano"
```

---

### Task 9: Batch CLI for 2019–2026

**Files:**
- Create: `guilherme/scripts/__init__.py`
- Create: `guilherme/scripts/ingest_all.py`
- Create: `guilherme/tests/unit/test_batch_ingest.py`

**Interfaces:**
- Consumes: `SUPPORTED_YEARS`, `ingest_year(...)`.
- Produces:
  - `discover_year_file(raw_root: Path, year: int) -> Path | None`
  - CLI `python scripts/ingest_all.py --raw-root data/raw [--force]`.

**File discovery rule:**
For each year, search only inside `data/raw/<year>/` and require exactly one `.csv` file. Zero files → skip with a clear message. More than one `.csv` → raise an error so the pipeline never guesses which source file to use.

- [ ] **Step 1: Write failing discovery tests**

Create `guilherme/tests/unit/test_batch_ingest.py`:

```python
import pytest

from scripts.ingest_all import discover_year_file


def test_discover_year_file_returns_only_csv(tmp_path):
    year_dir = tmp_path / "2025"
    year_dir.mkdir()
    expected = year_dir / "INFLUD25.csv"
    expected.write_text("a;b\n1;2\n", encoding="utf-8")

    assert discover_year_file(tmp_path, 2025) == expected


def test_discover_year_file_returns_none_when_missing(tmp_path):
    assert discover_year_file(tmp_path, 2025) is None


def test_discover_year_file_refuses_ambiguous_sources(tmp_path):
    year_dir = tmp_path / "2025"
    year_dir.mkdir()
    (year_dir / "a.csv").write_text("a\n1\n", encoding="utf-8")
    (year_dir / "b.csv").write_text("a\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Mais de um CSV"):
        discover_year_file(tmp_path, 2025)
```

- [ ] **Step 2: Run and verify failure**

```powershell
python -m pytest tests/unit/test_batch_ingest.py -v
```

Expected: FAIL because `scripts/ingest_all.py` does not exist.

- [ ] **Step 3: Make scripts importable for tests**

Create `guilherme/scripts/__init__.py`:

```python
"""Command-line orchestration scripts."""
```

- [ ] **Step 4: Implement batch CLI**

Create `guilherme/scripts/ingest_all.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from srag_api.config import SUPPORTED_YEARS
from srag_api.data.ingest import ingest_year


def discover_year_file(
    raw_root: Path,
    year: int,
) -> Path | None:
    year_dir = raw_root / str(year)
    if not year_dir.exists():
        return None

    csv_files = sorted(year_dir.glob("*.csv"))

    if not csv_files:
        return None

    if len(csv_files) > 1:
        raise ValueError(
            f"Mais de um CSV encontrado para {year}: "
            + ", ".join(path.name for path in csv_files)
        )

    return csv_files[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processa todos os anos disponiveis do SIVEP-Gripe."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    for year in SUPPORTED_YEARS:
        input_path = discover_year_file(args.raw_root, year)

        if input_path is None:
            print(f"[SKIP] {year}: nenhum CSV encontrado")
            continue

        try:
            output = ingest_year(
                input_path=input_path,
                parquet_root=Path("data/parquet"),
                quality_root=Path("data/quality"),
                year=year,
                force=args.force,
            )
        except FileExistsError as exc:
            print(f"[SKIP] {exc}")
            continue

        print(f"[OK] {year}: {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

```powershell
python -m pytest tests/unit/test_batch_ingest.py -v
```

Expected: all PASS.

- [ ] **Step 6: Run the full suite**

```powershell
python -m pytest -v
```

Expected: all PASS.

- [ ] **Step 7: Commit manually**

```powershell
git add guilherme/scripts/__init__.py guilherme/scripts/ingest_all.py guilherme/tests/unit/test_batch_ingest.py
git commit -m "feat: adiciona ingestao incremental de 2019 a 2026"
```

---

### Task 10: Protect Generated and Raw Data From Accidental Commits

**Files:**
- Create or Modify: `guilherme/.gitignore`
- Create: `guilherme/data/raw/.gitkeep`
- Create: `guilherme/data/parquet/.gitkeep`
- Create: `guilherme/data/quality/.gitkeep`

**Interfaces:**
- Consumes: directory layout from the design.
- Produces: safe Git behavior where raw and generated datasets stay local while directory intent remains documented.

- [ ] **Step 1: Create ignore rules**

Create `guilherme/.gitignore`:

```gitignore
# Source datasets
data/raw/**/*.csv

# Generated analytical datasets
data/parquet/**/*.parquet

# Generated quality reports
data/quality/*.json

# Python
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
.venv/
venv/
```

- [ ] **Step 2: Create directory markers**

Create empty files:

```text
guilherme/data/raw/.gitkeep
guilherme/data/parquet/.gitkeep
guilherme/data/quality/.gitkeep
```

- [ ] **Step 3: Verify ignore behavior**

From repository root:

```powershell
git status
git check-ignore -v guilherme/data/raw/example.csv
git check-ignore -v guilherme/data/parquet/example.parquet
git check-ignore -v guilherme/data/quality/quality_2025.json
```

Expected: each generated/data path is matched by `guilherme/.gitignore`.

- [ ] **Step 4: Run full tests once more**

From `guilherme/`:

```powershell
python -m pytest -v
```

Expected: all PASS.

- [ ] **Step 5: Commit manually**

From repository root:

```powershell
git add guilherme/.gitignore guilherme/data/raw/.gitkeep guilherme/data/parquet/.gitkeep guilherme/data/quality/.gitkeep
git commit -m "chore: protege dados brutos e artefatos gerados"
```

---

### Task 11: Pipeline README Documentation

**Files:**
- Create or Modify: `guilherme/README.md`

**Interfaces:**
- Consumes: completed pipeline commands and directory structure.
- Produces: reproducible instructions for another researcher/developer.

- [ ] **Step 1: Document the implemented pipeline**

The README must contain these sections:

```text
# SRAG Epidemiological Data Platform
## Estado atual
## Instalação
## Estrutura dos dados locais
## Processar um ano
## Processar todos os anos disponíveis
## Testes
## Saídas
## Próximas fases
```

It must document these exact commands:

```powershell
cd guilherme
python -m pip install -e ".[dev]"
python scripts/ingest_year.py --year 2025 --input data/raw/2025/INFLUD25.csv
python scripts/ingest_year.py --year 2025 --input data/raw/2025/INFLUD25.csv --force
python scripts/ingest_all.py --raw-root data/raw
python -m pytest -v
```

It must also state that raw CSVs and generated Parquet files are ignored by Git.

- [ ] **Step 2: Verify README commands against the implemented CLI**

Run:

```powershell
python scripts/ingest_year.py --help
python scripts/ingest_all.py --help
python -m pytest -q
```

Expected:
- both help commands exit successfully;
- tests PASS.

- [ ] **Step 3: Commit manually**

From repository root:

```powershell
git add guilherme/README.md
git commit -m "docs: documenta pipeline de dados SRAG"
```

---

## Final Verification Gate

Before starting the DuckDB/Repository plan, run from `guilherme/`:

```powershell
python -m pytest -v
```

Expected: all tests PASS.

Then run a fixture ingestion:

```powershell
python scripts/ingest_year.py --year 2025 --input tests/fixtures/sample_srag.csv --force
```

Verify these files exist:

```powershell
Test-Path data\parquet\srag\ano=2025\srag.parquet
Test-Path data\quality\quality_2025.json
```

Expected:

```text
True
True
```

Inspect the Parquet:

```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/parquet/srag/ano=2025/srag.parquet'); print(df[['ANO','UF','MUNICIPIO','IDADE_ANOS','FAIXA_ETARIA','ETIOLOGIA_NORMALIZADA','DESFECHO_NORMALIZADO','FOI_UTI']].to_string(index=False))"
```

Expected: three fixture rows with normalized year, geography, age, etiology, outcome and ICU fields.

Clean generated fixture output:

```powershell
Remove-Item -Recurse -Force data\parquet\srag\ano=2025
Remove-Item -Force data\quality\quality_2025.json
```

From repository root:

```powershell
git status
```

Expected: clean working tree after the user's final commit.

## Phase 1 Definition of Done

Phase 1 is complete only when:

1. CSV semicolon-separated can be read.
2. Required columns are validated explicitly.
3. `TP_IDADE` + `NU_IDADE_N` produce normalized age.
4. Age bands are centralized and tested.
5. ignored and absent values remain distinguishable.
6. etiological category is produced in an isolated mapping module.
7. source columns remain available after normalization.
8. geography produces UF, municipality and municipality code.
9. duplicate rows are removed from the analytical output.
10. Parquet is written under `ano=<YEAR>/`.
11. a quality JSON is generated per processed year.
12. reprocessing an existing year requires `--force`.
13. batch ingestion handles 2019–2026 without guessing between multiple CSVs.
14. raw CSV and generated Parquet/quality files are ignored by Git.
15. the full pytest suite passes.
16. `guilherme/analise_srag_pr.py` remains untouched.
