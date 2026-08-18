# SRAG ML V1 — Predição de Óbito na Admissão Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Criar um subsistema de preparação de dados para Machine Learning que gere um dataset nacional de SRAG 2019–2026 para predição de óbito na admissão/notificação, com proteção explícita contra leakage e validação temporal.

**Architecture:** O novo pacote `srag_api.ml` ficará desacoplado do treinamento de modelos. Ele receberá dados já normalizados pelo pipeline existente, definirá features permitidas/proibidas, construirá o alvo binário, montará o dataset, fará split temporal e aplicará pré-processamento ajustado somente no treino.

**Tech Stack:** Python 3, pandas, scikit-learn, pytest; imbalanced-learn somente se já disponível ou quando a etapa de balanceamento for implementada.

**Spec:** `docs/superpowers/specs/2026-08-18-srag-ml-admission-mortality-design.md`

## Global Constraints

- Escopo nacional, todos os casos de SRAG de 2019 a 2026.
- Alvo: `CURA -> 0`, `OBITO_SRAG -> 1`.
- Excluir do treino: `OBITO_OUTRAS_CAUSAS`, `AUSENTE`, `IGNORADO` e demais desfechos não elegíveis.
- Usar somente informações disponíveis até a notificação/admissão.
- Município fica fora da V1; UF/região podem entrar.
- 2026 é teste final fora do tempo e nunca participa de `fit`.
- Coluna ausente por schema não pode ser inventada como resultado negativo.
- Imputação, encoding e scaler são ajustados somente no treino.
- Balanceamento, quando utilizado, ocorre somente no treino.
- A suíte completa do projeto deve continuar passando.

---

## File Structure

### Criar

- `src/srag_api/ml/__init__.py` — superfície pública mínima do pacote.
- `src/srag_api/ml/features.py` — catálogo de features permitidas, grupos e bloqueio de leakage.
- `src/srag_api/ml/target.py` — construção do alvo binário.
- `src/srag_api/ml/dataset.py` — montagem de `X`, `y` e metadados.
- `src/srag_api/ml/split.py` — divisão temporal treino/validação/teste.
- `src/srag_api/ml/preprocessing.py` — pré-processamento ajustado apenas no treino.
- `tests/unit/ml/test_features.py`
- `tests/unit/ml/test_target.py`
- `tests/unit/ml/test_dataset.py`
- `tests/unit/ml/test_split.py`
- `tests/unit/ml/test_preprocessing.py`

### Modificar somente se necessário

- `pyproject.toml` — apenas se faltar dependência já exigida pelo código implementado.
- `README.md` — documentação final do novo módulo.
- `docs/superpowers/plans/2026-08-18-srag-ml-admission-mortality-implementation-plan.md` — este plano.

---

### Task 1: Catálogo de features e proteção contra leakage

**Files:**
- Create: `src/srag_api/ml/__init__.py`
- Create: `src/srag_api/ml/features.py`
- Test: `tests/unit/ml/test_features.py`

**Interfaces:**
- Produces:
  - `DEMOGRAPHIC_FEATURES: tuple[str, ...]`
  - `SYMPTOM_FEATURES: tuple[str, ...]`
  - `COMORBIDITY_FEATURES: tuple[str, ...]`
  - `GEOGRAPHIC_FEATURES: tuple[str, ...]`
  - `TEMPORAL_FEATURES: tuple[str, ...]`
  - `ADMISSION_FEATURES: tuple[str, ...]`
  - `LEAKAGE_FEATURES: frozenset[str]`
  - `validate_feature_registry() -> None`

- [ ] **Step 1: Write the failing leakage registry tests**

```python
from srag_api.ml.features import (
    ADMISSION_FEATURES,
    COMORBIDITY_FEATURES,
    LEAKAGE_FEATURES,
    validate_feature_registry,
)


def test_admission_features_do_not_contain_known_leakage():
    assert set(ADMISSION_FEATURES).isdisjoint(LEAKAGE_FEATURES)


def test_extended_comorbidities_are_available_as_candidates():
    expected = {
        "CARDIOPATI",
        "DIABETES",
        "PNEUMOPATI",
        "RENAL",
        "HEPATICA",
        "IMUNODEPRE",
        "OBESIDADE",
        "OUT_MORBI",
    }
    assert expected.issubset(COMORBIDITY_FEATURES)


def test_validate_feature_registry_accepts_current_registry():
    validate_feature_registry()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/unit/ml/test_features.py -v
```

Expected: import failure because `srag_api.ml.features` does not exist.

- [ ] **Step 3: Implement minimal feature registry**

Create `features.py` with explicit immutable groups.

Minimum leakage block:

```python
LEAKAGE_FEATURES = frozenset(
    {
        "EVOLUCAO",
        "DESFECHO_NORMALIZADO",
        "OBITO_SRAG",
        "DT_EVOLUCA",
        "UTI",
        "SUPORT_VEN",
        "QTD_DIAS",
        "DIAS_INTERNA",
        "PCR_EVOLUCAO",
    }
)
```

Admission groups should include the approved V1 candidates, including:

```python
DEMOGRAPHIC_FEATURES = (
    "CS_SEXO",
    "NU_IDADE_N",
    "CS_GESTANT",
)

SYMPTOM_FEATURES = (
    "FEBRE",
    "TOSSE",
    "GARGANTA",
    "DISPNEIA",
    "DESC_RESP",
    "SATURACAO",
    "DIARREIA",
    "VOMITO",
    "DOR_ABD",
    "FADIGA",
    "PERD_OLFT",
    "PERD_PALA",
    "OUTRO_SIN",
)

COMORBIDITY_FEATURES = (
    "CARDIOPATI",
    "DIABETES",
    "PNEUMOPATI",
    "RENAL",
    "HEPATICA",
    "IMUNODEPRE",
    "OBESIDADE",
    "OUT_MORBI",
    "FATOR_RISC",
)

GEOGRAPHIC_FEATURES = (
    "SG_UF",
    "REGIAO",
)

TEMPORAL_FEATURES = (
    "SINT_ATE_NOTIF",
)
```

`ADMISSION_FEATURES` is the concatenation of these groups.

`validate_feature_registry()` must raise `ValueError` if there is intersection with `LEAKAGE_FEATURES`.

- [ ] **Step 4: Run tests to verify GREEN**

```powershell
pytest tests/unit/ml/test_features.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/__init__.py src/srag_api/ml/features.py tests/unit/ml/test_features.py
git commit -m "feat: define features seguras para ml de admissao"
```

---

### Task 2: Construção do alvo binário

**Files:**
- Create: `src/srag_api/ml/target.py`
- Test: `tests/unit/ml/test_target.py`

**Interfaces:**
- Produces:
  - `build_mortality_target(df: pd.DataFrame, source_column: str = "DESFECHO_NORMALIZADO") -> pd.Series`
  - `eligible_outcome_mask(df: pd.DataFrame, source_column: str = "DESFECHO_NORMALIZADO") -> pd.Series`

- [ ] **Step 1: Write failing target tests**

```python
import pandas as pd

from srag_api.ml.target import build_mortality_target, eligible_outcome_mask


def test_target_maps_cure_and_srag_death():
    df = pd.DataFrame(
        {"DESFECHO_NORMALIZADO": ["CURA", "OBITO_SRAG"]}
    )

    target = build_mortality_target(df)

    assert target.tolist() == [0, 1]


def test_other_cause_death_is_not_eligible():
    df = pd.DataFrame(
        {"DESFECHO_NORMALIZADO": ["OBITO_OUTRAS_CAUSAS"]}
    )

    assert eligible_outcome_mask(df).tolist() == [False]


def test_missing_and_ignored_are_not_eligible():
    df = pd.DataFrame(
        {"DESFECHO_NORMALIZADO": ["AUSENTE", "IGNORADO", None]}
    )

    assert eligible_outcome_mask(df).tolist() == [False, False, False]
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/unit/ml/test_target.py -v
```

Expected: module/function missing.

- [ ] **Step 3: Implement target logic**

Use only explicit mapping:

```python
TARGET_MAPPING = {
    "CURA": 0,
    "OBITO_SRAG": 1,
}
```

`eligible_outcome_mask()` should be based on `.isin(TARGET_MAPPING)`.

`build_mortality_target()` should return nullable integer or integer series for already-filtered rows; if ineligible values are present, return `NA` for them rather than silently converting.

- [ ] **Step 4: Run GREEN**

```powershell
pytest tests/unit/ml/test_target.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/target.py tests/unit/ml/test_target.py
git commit -m "feat: define alvo de mortalidade srag"
```

---

### Task 3: Montagem segura do dataset de admissão

**Files:**
- Create: `src/srag_api/ml/dataset.py`
- Test: `tests/unit/ml/test_dataset.py`

**Interfaces:**
- Consumes:
  - `ADMISSION_FEATURES`
  - `LEAKAGE_FEATURES`
  - `eligible_outcome_mask()`
  - `build_mortality_target()`
- Produces:
  - `AdmissionDataset` dataclass with:
    - `X: pd.DataFrame`
    - `y: pd.Series`
    - `metadata: pd.DataFrame`
  - `build_admission_dataset(df: pd.DataFrame) -> AdmissionDataset`

- [ ] **Step 1: Write failing dataset tests**

```python
import pandas as pd

from srag_api.ml.dataset import build_admission_dataset


def test_dataset_keeps_only_eligible_outcomes():
    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": [
                "CURA",
                "OBITO_SRAG",
                "OBITO_OUTRAS_CAUSAS",
                "AUSENTE",
            ],
            "NU_IDADE_N": [20, 70, 50, 40],
            "CS_SEXO": ["F", "M", "M", "F"],
            "ANO": [2024, 2024, 2024, 2024],
        }
    )

    result = build_admission_dataset(df)

    assert result.y.tolist() == [0, 1]
    assert len(result.X) == 2


def test_missing_optional_feature_is_not_fabricated():
    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": ["CURA"],
            "NU_IDADE_N": [30],
            "ANO": [2024],
        }
    )

    result = build_admission_dataset(df)

    assert "CARDIOPATI" not in result.X.columns


def test_dataset_never_exports_leakage_columns():
    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": ["CURA"],
            "NU_IDADE_N": [30],
            "UTI": [1],
            "SUPORT_VEN": [2],
            "DT_EVOLUCA": ["2024-02-01"],
            "ANO": [2024],
        }
    )

    result = build_admission_dataset(df)

    assert "UTI" not in result.X.columns
    assert "SUPORT_VEN" not in result.X.columns
    assert "DT_EVOLUCA" not in result.X.columns
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/unit/ml/test_dataset.py -v
```

- [ ] **Step 3: Implement dataset builder**

Rules:

1. Apply `eligible_outcome_mask`.
2. Build `y` after filtering.
3. Select intersection between `ADMISSION_FEATURES` and actual dataframe columns.
4. Never create absent source columns with 0.
5. Raise `ValueError` if selected features intersect `LEAKAGE_FEATURES`.
6. Preserve metadata separately. Minimum metadata fields when present:
   - `ANO`
   - `SG_UF`
   - `DT_NOTIFIC`

Do not include metadata-only fields in `X` unless they are explicitly in `ADMISSION_FEATURES`.

- [ ] **Step 4: Run GREEN**

```powershell
pytest tests/unit/ml/test_dataset.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/dataset.py tests/unit/ml/test_dataset.py
git commit -m "feat: monta dataset seguro para ml de admissao"
```

---

### Task 4: Split temporal com 2026 fora do treino

**Files:**
- Create: `src/srag_api/ml/split.py`
- Test: `tests/unit/ml/test_split.py`

**Interfaces:**
- Produces:
  - `TemporalSplit` dataclass:
    - `train_idx`
    - `validation_idx`
    - `test_idx`
  - `temporal_split(years: pd.Series, validation_year: int = 2025, test_year: int = 2026) -> TemporalSplit`

- [ ] **Step 1: Write failing temporal split tests**

```python
import pandas as pd

from srag_api.ml.split import temporal_split


def test_temporal_split_reserves_2026_for_test():
    years = pd.Series([2019, 2020, 2024, 2025, 2026])

    split = temporal_split(years)

    assert years.iloc[split.test_idx].tolist() == [2026]
    assert 2026 not in years.iloc[split.train_idx].tolist()


def test_validation_is_more_recent_than_training():
    years = pd.Series([2021, 2023, 2024, 2025, 2026])

    split = temporal_split(years)

    assert max(years.iloc[split.train_idx]) < min(years.iloc[split.validation_idx])


def test_split_has_no_overlapping_indices():
    years = pd.Series([2023, 2024, 2025, 2026])

    split = temporal_split(years)

    train = set(split.train_idx)
    validation = set(split.validation_idx)
    test = set(split.test_idx)

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/unit/ml/test_split.py -v
```

- [ ] **Step 3: Implement temporal split**

Initial default:

- train: `year < 2025`
- validation: `year == 2025`
- test: `year == 2026`

Validate:
- no missing year for rows entering the split;
- no overlap;
- each requested partition exists for the observed data, otherwise raise informative `ValueError`.

The defaults may be made configurable but 2026 remains the default test year.

- [ ] **Step 4: Run GREEN**

```powershell
pytest tests/unit/ml/test_split.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/split.py tests/unit/ml/test_split.py
git commit -m "feat: adiciona validacao temporal para ml srag"
```

---

### Task 5: Pré-processamento ajustado somente no treino

**Files:**
- Create: `src/srag_api/ml/preprocessing.py`
- Test: `tests/unit/ml/test_preprocessing.py`
- Modify: `pyproject.toml` only if required by missing dependencies.

**Interfaces:**
- Produces:
  - `build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer`
  - `fit_preprocessor_on_train(preprocessor, X_train) -> fitted preprocessor`
  - `transform_partitions(preprocessor, X_train, X_validation, X_test) -> tuple`

- [ ] **Step 1: Write failing preprocessing tests**

```python
import pandas as pd

from srag_api.ml.preprocessing import (
    build_preprocessor,
    fit_preprocessor_on_train,
    transform_partitions,
)


def test_numeric_imputation_is_learned_from_train_only():
    X_train = pd.DataFrame({"NU_IDADE_N": [20.0, 40.0, None]})
    X_validation = pd.DataFrame({"NU_IDADE_N": [1000.0]})
    X_test = pd.DataFrame({"NU_IDADE_N": [2000.0]})

    preprocessor = build_preprocessor(
        numeric_features=["NU_IDADE_N"],
        categorical_features=[],
    )
    fitted = fit_preprocessor_on_train(preprocessor, X_train)

    imputer = fitted.named_transformers_["numeric"].named_steps["imputer"]

    assert imputer.statistics_[0] == 30.0


def test_unknown_category_in_future_partition_does_not_refit_encoder():
    X_train = pd.DataFrame({"CS_SEXO": ["F", "M"]})
    X_validation = pd.DataFrame({"CS_SEXO": ["I"]})
    X_test = pd.DataFrame({"CS_SEXO": ["X"]})

    preprocessor = build_preprocessor(
        numeric_features=[],
        categorical_features=["CS_SEXO"],
    )
    fitted = fit_preprocessor_on_train(preprocessor, X_train)

    train, validation, test = transform_partitions(
        fitted,
        X_train,
        X_validation,
        X_test,
    )

    assert train.shape[0] == 2
    assert validation.shape[0] == 1
    assert test.shape[0] == 1
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/unit/ml/test_preprocessing.py -v
```

- [ ] **Step 3: Implement preprocessing pipeline**

Recommended minimal implementation:

```python
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)
```

Use `ColumnTransformer`.

Important:
- `fit_preprocessor_on_train()` is the only function that calls `.fit()`.
- `transform_partitions()` calls only `.transform()`.
- Do not fit on concatenated train/validation/test.

If a partition has no numeric or categorical features, handle it without invalid empty transformers.

- [ ] **Step 4: Run GREEN**

```powershell
pytest tests/unit/ml/test_preprocessing.py -v
```

- [ ] **Step 5: Run all ML unit tests**

```powershell
pytest tests/unit/ml -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/ml/preprocessing.py tests/unit/ml/test_preprocessing.py pyproject.toml
git commit -m "feat: adiciona preprocessamento sem leakage"
```

If `pyproject.toml` did not change, omit it from `git add`.

---

### Task 6: Balanceamento restrito ao treino

**Files:**
- Modify: `src/srag_api/ml/preprocessing.py`
- Modify: `tests/unit/ml/test_preprocessing.py`
- Modify: `pyproject.toml` only if `imbalanced-learn` is not already a dependency.

**Interfaces:**
- Produces:
  - `balance_training_data(X_train, y_train, *, strategy: str = "none", random_state: int = 42)`

- [ ] **Step 1: Write failing balance tests**

```python
import pandas as pd

from srag_api.ml.preprocessing import balance_training_data


def test_none_strategy_keeps_training_data_unchanged():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([0, 0, 1])

    X_out, y_out = balance_training_data(X, y, strategy="none")

    assert len(X_out) == 3
    assert y_out.tolist() == [0, 0, 1]


def test_balance_api_only_accepts_training_partition():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([0, 0, 1])

    X_out, y_out = balance_training_data(
        X,
        y,
        strategy="none",
        random_state=42,
    )

    assert len(X_out) == len(y_out)
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/unit/ml/test_preprocessing.py -v
```

- [ ] **Step 3: Implement minimal balance API**

Start with:
- `strategy="none"` always supported.
- Add `strategy="smote"` only if dependency and feature representation are appropriate at this point.

If implementing SMOTE:
- apply only after train preprocessing;
- never expose a function that receives validation/test together with the training balance call;
- use deterministic `random_state=42`.

Do not apply SMOTE to sparse/object data that the installed version cannot safely process; in that case keep V1 API with `none` and defer actual SMOTE to the model-experiment phase.

- [ ] **Step 4: Run GREEN**

```powershell
pytest tests/unit/ml/test_preprocessing.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/preprocessing.py tests/unit/ml/test_preprocessing.py pyproject.toml
git commit -m "feat: restringe balanceamento ao treino"
```

---

### Task 7: Integração ponta a ponta do dataset V1

**Files:**
- Create: `tests/integration/test_ml_dataset_pipeline.py`
- Modify: `src/srag_api/ml/__init__.py`

**Interfaces:**
- Consumes all previous tasks.
- Produces a stable public import path for dataset construction and split.

- [ ] **Step 1: Write failing integration test**

```python
import pandas as pd

from srag_api.ml import build_admission_dataset, temporal_split


def test_ml_dataset_pipeline_builds_target_without_leakage_and_splits_time():
    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": [
                "CURA",
                "OBITO_SRAG",
                "CURA",
                "OBITO_SRAG",
                "OBITO_OUTRAS_CAUSAS",
            ],
            "ANO": [2023, 2024, 2025, 2026, 2026],
            "NU_IDADE_N": [20, 70, 35, 80, 50],
            "CS_SEXO": ["F", "M", "F", "M", "M"],
            "DISPNEIA": [2, 1, 2, 1, 1],
            "UTI": [2, 1, 2, 1, 1],
        }
    )

    dataset = build_admission_dataset(df)
    split = temporal_split(dataset.metadata["ANO"])

    assert dataset.y.tolist() == [0, 1, 0, 1]
    assert "UTI" not in dataset.X.columns
    assert dataset.metadata["ANO"].iloc[split.test_idx].tolist() == [2026]
```

- [ ] **Step 2: Run RED if public exports are missing**

```powershell
pytest tests/integration/test_ml_dataset_pipeline.py -v
```

- [ ] **Step 3: Export stable public interfaces**

In `src/srag_api/ml/__init__.py` expose only the intended API:

```python
from .dataset import AdmissionDataset, build_admission_dataset
from .split import TemporalSplit, temporal_split

__all__ = [
    "AdmissionDataset",
    "TemporalSplit",
    "build_admission_dataset",
    "temporal_split",
]
```

- [ ] **Step 4: Run integration test**

```powershell
pytest tests/integration/test_ml_dataset_pipeline.py -v
```

- [ ] **Step 5: Run complete test suite**

```powershell
pytest -q
```

Expected: current project suite plus new ML tests all pass.

- [ ] **Step 6: Commit**

```powershell
git add src/srag_api/ml/__init__.py tests/integration/test_ml_dataset_pipeline.py
git commit -m "test: valida pipeline ml srag ponta a ponta"
```

---

### Task 8: Documentação e auditoria final

**Files:**
- Modify: `README.md`
- Optional create: `docs/relatorios/` artifact later, outside this implementation plan.

- [ ] **Step 1: Add README section**

Document:
- research question;
- target definition;
- admission-only feature policy;
- known leakage block;
- 2026 out-of-time test;
- preprocessing fit only on train;
- path `src/srag_api/ml/`;
- note that model training is a later phase.

- [ ] **Step 2: Audit forbidden fields in ML source/tests**

Run:

```powershell
git grep -n -E "EVOLUCAO|DT_EVOLUCA|UTI|SUPORT_VEN|QTD_DIAS|DIAS_INTERNA|PCR_EVOLUCAO" -- src/srag_api/ml tests/unit/ml tests/integration/test_ml_dataset_pipeline.py
```

Expected:
- matches are allowed in `LEAKAGE_FEATURES` and tests that assert blocking;
- no forbidden field appears inside `ADMISSION_FEATURES`.

- [ ] **Step 3: Run ML suite**

```powershell
pytest tests/unit/ml tests/integration/test_ml_dataset_pipeline.py -q
```

Expected: all pass.

- [ ] **Step 4: Run full project suite**

```powershell
pytest -q
```

Expected: all pass; pre-existing external deprecation warnings may remain but no new project failure.

- [ ] **Step 5: Inspect git diff**

```powershell
git status
git diff --stat
git diff
```

Confirm there are no unrelated changes.

- [ ] **Step 6: Commit documentation**

```powershell
git add README.md
git commit -m "docs: documenta dataset ml de admissao"
```

---

## Final Verification Checklist

- [ ] `ADMISSION_FEATURES` and `LEAKAGE_FEATURES` are disjoint.
- [ ] Extended comorbidities are included as candidates.
- [ ] `CURA -> 0`.
- [ ] `OBITO_SRAG -> 1`.
- [ ] Other-cause death is excluded from the ML population.
- [ ] Undefined outcomes are excluded.
- [ ] Missing optional schema columns are not fabricated.
- [ ] Municipality is not an admission feature.
- [ ] 2026 never appears in training.
- [ ] Imputer/scaler/encoder are fit only on train.
- [ ] Validation/test use transform only.
- [ ] Balancing API is train-only.
- [ ] Integration test covers target + leakage + temporal split.
- [ ] Full project suite passes.
- [ ] README documents methodological decisions.

## Expected Commit Sequence

```text
feat: define features seguras para ml de admissao
feat: define alvo de mortalidade srag
feat: monta dataset seguro para ml de admissao
feat: adiciona validacao temporal para ml srag
feat: adiciona preprocessamento sem leakage
feat: restringe balanceamento ao treino
test: valida pipeline ml srag ponta a ponta
docs: documenta dataset ml de admissao
```

## Out of Scope for This Plan

Do not implement yet:
- Random Forest, XGBoost, CNN-1D or other estimators;
- hyperparameter tuning;
- SHAP/LIME;
- API endpoint for prediction;
- model persistence/registry;
- deployment;
- Model B with laboratory data;
- Model C with inpatient progression data.

Those belong to the next implementation cycle after the V1 dataset pipeline is validated.
