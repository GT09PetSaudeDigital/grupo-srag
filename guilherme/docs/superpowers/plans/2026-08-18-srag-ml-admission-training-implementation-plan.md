# SRAG ML Admission Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o treinamento, comparação, seleção de limiar, avaliação out-of-time e persistência de artefatos para quatro modelos clássicos de Machine Learning de mortalidade por SRAG na admissão.

**Architecture:** A lógica científica ficará modularizada em `srag_api.ml`, reutilizando `build_admission_dataset`, `temporal_split` e o preprocessamento já existentes. Os quatro modelos serão treinados apenas em 2019–2024, comparados em 2025 por AUC-PR, terão o limiar escolhido somente em 2025 e serão avaliados uma única vez em 2026, com persistência dos resultados e do melhor pipeline.

**Tech Stack:** Python 3.10+, pandas, scikit-learn >=1.5,<2, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-srag-ml-admission-training-design.md`

## Global Constraints

- Treino padrão: 2019–2024.
- Validação padrão: 2025.
- Teste out-of-time padrão: 2026.
- AUC-PR / Average Precision é a métrica principal de seleção.
- O limiar é escolhido somente em 2025.
- Política principal do limiar: maximizar recall com `precision >= 0.50`.
- Fallback do limiar: maximizar F1 e registrar `fallback_max_f1`.
- 2026 não pode participar de seleção de modelo, limiar, preprocessing, tuning ou definição de features.
- Sem PCA na V1.
- Sem SMOTE na V1.
- Hiperparâmetros fixos na V1.
- Pesos de classe ou `sample_weight` devem ser derivados somente do treino.
- Artefatos experimentais não entram no Git.
- Usar `python -m pytest` no Windows.
- Antes de cada commit, executar os testes específicos da tarefa.
- Antes de encerrar a fase, executar a suíte completa.

---

## File Map

### Novos arquivos

- `src/srag_api/ml/models.py`
  - Registro dos quatro modelos e política de pesos.
- `src/srag_api/ml/metrics.py`
  - Cálculo estruturado de métricas e matriz de confusão.
- `src/srag_api/ml/threshold.py`
  - Seleção do limiar em validação.
- `src/srag_api/ml/training.py`
  - Orquestração ponta a ponta de treino, validação, seleção e teste.
- `src/srag_api/ml/artifacts.py`
  - Persistência do melhor pipeline, métricas e metadados.
- `scripts/train_ml_admission.py`
  - CLI fina para executar o treinamento.
- `tests/unit/ml/test_models.py`
- `tests/unit/ml/test_metrics.py`
- `tests/unit/ml/test_threshold.py`
- `tests/unit/ml/test_training.py`
- `tests/unit/ml/test_artifacts.py`
- `tests/integration/test_ml_training_pipeline.py`

### Arquivos modificados

- `src/srag_api/ml/__init__.py`
  - Exportar interfaces públicas do novo treinamento.
- `src/srag_api/ml/preprocessing.py`
  - Apenas se necessário para expor informação usada pelo training, sem alterar a política train-only já validada.
- `.gitignore` ou `guilherme/.gitignore`
  - Ignorar `artifacts/ml-admission/`.
- `README.md`
  - Documentar execução do treinamento e artefatos gerados.

---

### Task 1: Registry dos quatro modelos

**Files:**
- Create: `src/srag_api/ml/models.py`
- Test: `tests/unit/ml/test_models.py`

**Interfaces:**
- Produces:
  - `RANDOM_STATE: int`
  - `build_models(random_state: int = 42) -> dict[str, BaseEstimator]`
  - `build_gradient_boosting_sample_weight(y_train: pd.Series) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Criar testes que garantam:

```python
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from srag_api.ml.models import build_gradient_boosting_sample_weight, build_models


def test_build_models_registers_exactly_four_v1_models():
    models = build_models()
    assert list(models) == [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
    ]


def test_model_types_are_expected():
    models = build_models()
    assert isinstance(models["logistic_regression"], LogisticRegression)
    assert isinstance(models["random_forest"], RandomForestClassifier)
    assert isinstance(models["gradient_boosting"], GradientBoostingClassifier)
    assert isinstance(models["hist_gradient_boosting"], HistGradientBoostingClassifier)


def test_supported_models_use_class_balancing_when_available():
    models = build_models()
    assert models["logistic_regression"].class_weight == "balanced"
    assert models["random_forest"].class_weight in {"balanced", "balanced_subsample"}
    assert models["hist_gradient_boosting"].class_weight == "balanced"


def test_gradient_boosting_sample_weight_is_derived_from_training_labels():
    y = pd.Series([0, 0, 0, 1])
    weights = build_gradient_boosting_sample_weight(y)

    assert len(weights) == len(y)
    assert weights[y == 1][0] > weights[y == 0][0]
```

- [ ] **Step 2: Run tests to verify RED**

```powershell
python -m pytest tests\unit\ml\test_models.py -v
```

Expected: FAIL because `srag_api.ml.models` does not exist.

- [ ] **Step 3: Implement minimal model registry**

Implementar `models.py` com:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42


def build_models(random_state: int = RANDOM_STATE):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def build_gradient_boosting_sample_weight(y_train: pd.Series) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=y_train)
```

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests\unit\ml\test_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/models.py tests/unit/ml/test_models.py
git commit -m "feat: adiciona modelos baseline para ml de admissao"
```

---

### Task 2: Métricas estruturadas

**Files:**
- Create: `src/srag_api/ml/metrics.py`
- Test: `tests/unit/ml/test_metrics.py`

**Interfaces:**
- Produces:
  - `BinaryMetrics`
  - `evaluate_binary_predictions(y_true, probabilities, threshold=0.5) -> BinaryMetrics`

- [ ] **Step 1: Write the failing tests**

Cobrir:

```python
import numpy as np
import pandas as pd

from srag_api.ml.metrics import evaluate_binary_predictions


def test_metrics_return_auc_pr_roc_auc_and_threshold_metrics():
    y = pd.Series([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])

    result = evaluate_binary_predictions(y, p, threshold=0.5)

    assert 0 <= result.auc_pr <= 1
    assert 0 <= result.roc_auc <= 1
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.f1 == 1.0


def test_metrics_include_2x2_confusion_matrix():
    result = evaluate_binary_predictions(
        pd.Series([0, 0, 1, 1]),
        np.array([0.2, 0.8, 0.7, 0.1]),
        threshold=0.5,
    )

    assert result.confusion_matrix.shape == (2, 2)


def test_metrics_reject_single_class_partition():
    with pytest.raises(ValueError, match="duas classes"):
        evaluate_binary_predictions(
            pd.Series([0, 0, 0]),
            np.array([0.1, 0.2, 0.3]),
        )
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests\unit\ml\test_metrics.py -v
```

- [ ] **Step 3: Implement `BinaryMetrics` and evaluator**

Usar:

- `average_precision_score`
- `roc_auc_score`
- `precision_score`
- `recall_score`
- `f1_score`
- `confusion_matrix`

`BinaryMetrics` deve ser um `@dataclass(frozen=True)` e conter:

```python
auc_pr: float
roc_auc: float
recall: float
precision: float
f1: float
threshold: float
confusion_matrix: np.ndarray
```

Validar que `y_true` possui as classes `{0, 1}`.

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests\unit\ml\test_metrics.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/metrics.py tests/unit/ml/test_metrics.py
git commit -m "feat: adiciona metricas para avaliacao ml"
```

---

### Task 3: Seleção de limiar usando somente validação

**Files:**
- Create: `src/srag_api/ml/threshold.py`
- Test: `tests/unit/ml/test_threshold.py`

**Interfaces:**
- Produces:
  - `ThresholdSelection`
  - `select_decision_threshold(y_validation, probabilities, min_precision=0.50) -> ThresholdSelection`

- [ ] **Step 1: Write failing tests**

Criar casos determinísticos para:

```python
def test_threshold_prefers_max_recall_with_minimum_precision():
    ...


def test_threshold_never_uses_default_05_when_better_valid_threshold_exists():
    ...


def test_threshold_falls_back_to_max_f1_when_precision_constraint_is_impossible():
    ...


def test_threshold_selection_rejects_single_class_validation():
    ...
```

`ThresholdSelection` deverá expor:

```python
threshold: float
policy: str
precision: float
recall: float
f1: float
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests\unit\ml\test_threshold.py -v
```

- [ ] **Step 3: Implement selection**

Usar `precision_recall_curve`.

Política:

```text
1. construir candidatos válidos com precision >= min_precision
2. entre eles escolher maior recall
3. em empate, escolher maior precision
4. em novo empate, escolher maior threshold para comportamento determinístico
5. policy = "max_recall_precision_ge_0_50"
```

Fallback:

```text
1. calcular F1 de cada threshold
2. escolher maior F1
3. usar desempate determinístico por maior recall e maior threshold
4. policy = "fallback_max_f1"
```

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests\unit\ml\test_threshold.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/threshold.py tests/unit/ml/test_threshold.py
git commit -m "feat: adiciona selecao de limiar na validacao"
```

---

### Task 4: Unidade de treino por modelo

**Files:**
- Create: `src/srag_api/ml/training.py`
- Test: `tests/unit/ml/test_training.py`

**Interfaces:**
- Consumes:
  - `build_preprocessor(...)`
  - `fit_preprocessor_on_train(...)`
  - `build_models(...)`
  - `build_gradient_boosting_sample_weight(...)`
  - `evaluate_binary_predictions(...)`
- Produces:
  - `TrainedCandidate`
  - `split_preprocessing_features(X: pd.DataFrame) -> tuple[list[str], list[str]]`
  - `train_candidate_model(...)`

- [ ] **Step 1: Write failing tests**

Testar uma função de baixo nível:

```python
def train_candidate_model(
    *,
    name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    preprocessor,
) -> TrainedCandidate:
    ...
```

`TrainedCandidate` deve conter:

```python
name: str
pipeline: object
validation_probabilities: np.ndarray
validation_metrics: BinaryMetrics
```

Cobrir:

- `NU_IDADE_N` e `SINT_ATE_NOTIF`, quando presentes, são tratadas como numéricas;
- as demais features de admissão presentes são tratadas como categóricas;
- a classificação usa apenas colunas realmente presentes, sem fabricar features ausentes;
- fit do preprocessador ocorre só em `X_train`;
- validação usa somente `transform`;
- probabilidades são produzidas para validação;
- Gradient Boosting recebe `sample_weight`;
- demais modelos não recebem sample weights externos;
- leakage features não são aceitas se aparecerem em `X_train`.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests\unit\ml\test_training.py -v
```

- [ ] **Step 3: Implement**

Antes do pipeline, implementar uma regra explícita de tipagem:

```python
NUMERIC_MODEL_FEATURES = frozenset({"NU_IDADE_N", "SINT_ATE_NOTIF"})

def split_preprocessing_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in X.columns if c in NUMERIC_MODEL_FEATURES]
    categorical = [c for c in X.columns if c not in NUMERIC_MODEL_FEATURES]
    return numeric, categorical
```

Passar essas listas para `build_preprocessor(...)`.

Usar `sklearn.pipeline.Pipeline` ou equivalente que permita persistir:

```text
preprocessor ajustado + modelo
```

Como `GradientBoostingClassifier` precisa de `sample_weight`, o fit pode usar:

```python
pipeline.fit(X_train, y_train, modelo__sample_weight=weights)
```

caso a estrutura seja:

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("model", estimator),
])
```

Para demais modelos:

```python
pipeline.fit(X_train, y_train)
```

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests\unit\ml\test_training.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/training.py tests/unit/ml/test_training.py
git commit -m "feat: adiciona treino isolado de candidatos ml"
```

---

### Task 5: Seleção do vencedor em 2025

**Files:**
- Modify: `src/srag_api/ml/training.py`
- Modify: `tests/unit/ml/test_training.py`

**Interfaces:**
- Produces:
  - `ValidationComparison`
  - `select_best_candidate(candidates) -> TrainedCandidate`

- [ ] **Step 1: Add failing tests**

Cobrir:

```python
def test_best_candidate_is_selected_by_auc_pr():
    ...


def test_auc_pr_tie_is_resolved_by_model_registry_order():
    ...
```

A ordem do registro deve permanecer:

```text
logistic_regression
random_forest
gradient_boosting
hist_gradient_boosting
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests\unit\ml\test_training.py -v
```

- [ ] **Step 3: Implement minimal deterministic selection**

Selecionar somente por `validation_metrics.auc_pr`.

Não usar 2026 em qualquer ponto desta função.

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests\unit\ml\test_training.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/training.py tests/unit/ml/test_training.py
git commit -m "feat: seleciona melhor modelo por auc pr"
```

---

### Task 6: Orquestração temporal completa

**Files:**
- Modify: `src/srag_api/ml/training.py`
- Modify: `tests/unit/ml/test_training.py`

**Interfaces:**
- Consumes:
  - `AdmissionDataset`
  - `TemporalSplit`
  - model registry
  - threshold selection
  - metrics
- Produces:
  - `TrainingRunResult`
  - `run_admission_training(dataset, split, ...) -> TrainingRunResult`

`TrainingRunResult` deve conter pelo menos:

```python
candidates: dict[str, TrainedCandidate]
best_model_name: str
best_pipeline: object
threshold: float
threshold_policy: str
validation_metrics: BinaryMetrics
test_metrics: BinaryMetrics
train_size: int
validation_size: int
test_size: int
```

- [ ] **Step 1: Add failing tests**

Cobrir:

- usa os índices temporais corretos;
- todos os quatro modelos treinam somente no treino;
- vencedor vem de 2025;
- limiar é escolhido com probabilidades de 2025;
- teste de 2026 só é avaliado depois da seleção;
- erro se qualquer partição estiver sem as duas classes;
- `OBITO_OUTRAS_CAUSAS` nunca reaparece, pois a função recebe o `AdmissionDataset` já filtrado;
- nenhuma coluna de `LEAKAGE_FEATURES` entra no treinamento.

Criar spies/fakes onde necessário para provar que dados de 2026 não chegam às funções de seleção.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests\unit\ml\test_training.py -v
```

- [ ] **Step 3: Implement orchestration**

Fluxo obrigatório:

```python
X_train = dataset.X.iloc[split.train_idx]
X_validation = dataset.X.iloc[split.validation_idx]
X_test = dataset.X.iloc[split.test_idx]

y_train = dataset.y.iloc[split.train_idx]
y_validation = dataset.y.iloc[split.validation_idx]
y_test = dataset.y.iloc[split.test_idx]
```

Depois:

```text
train candidates
→ compare validation AUC-PR
→ select best
→ select threshold on validation
→ evaluate frozen winner on test
```

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests\unit\ml\test_training.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/training.py tests/unit/ml/test_training.py
git commit -m "feat: orquestra treino e teste temporal do ml"
```

---

### Task 7: Persistência de artefatos

**Files:**
- Create: `src/srag_api/ml/artifacts.py`
- Test: `tests/unit/ml/test_artifacts.py`

**Interfaces:**
- Produces:
  - `ArtifactPaths`
  - `save_training_artifacts(result, output_dir, metadata) -> ArtifactPaths`

- [ ] **Step 1: Write failing tests**

Usar `tmp_path`.

Cobrir existência e conteúdo de:

```text
best_model.joblib
metrics.json
metrics.csv
validation_comparison.csv
confusion_matrix_validation.csv
confusion_matrix_test.csv
run_metadata.json
```

Testar também que:

```python
joblib.load(best_model_path)
```

retorna estrutura com:

```text
pipeline
threshold
features
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests\unit\ml\test_artifacts.py -v
```

- [ ] **Step 3: Implement serialization**

`metrics.json` deve conter:

```text
selection_metric = average_precision
validation_year = 2025
test_year = 2026
best_model
threshold
threshold_policy
validation
test
```

`validation_comparison.csv` deve conter uma linha por candidato.

`run_metadata.json` deve conter:

```text
timestamp
random_state
train_years
validation_year
test_year
features_used
features_missing
partition_sizes
class_prevalence
best_model
best_model_params
selection_metric
threshold_policy
python_version
pandas_version
scikit_learn_version
```

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests\unit\ml\test_artifacts.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/artifacts.py tests/unit/ml/test_artifacts.py
git commit -m "feat: persiste artefatos do treinamento ml"
```

---

### Task 8: API pública do módulo ML

**Files:**
- Modify: `src/srag_api/ml/__init__.py`
- Test: `tests/unit/ml/test_training.py`

**Interfaces:**
- Exportar:
  - `build_models`
  - `evaluate_binary_predictions`
  - `select_decision_threshold`
  - `run_admission_training`
  - `save_training_artifacts`

- [ ] **Step 1: Add failing public API test**

```python
import srag_api.ml as ml


def test_training_api_is_exported():
    assert callable(ml.build_models)
    assert callable(ml.evaluate_binary_predictions)
    assert callable(ml.select_decision_threshold)
    assert callable(ml.run_admission_training)
    assert callable(ml.save_training_artifacts)
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests\unit\ml\test_training.py -v
```

- [ ] **Step 3: Export interfaces**

Atualizar `__all__`.

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests\unit\ml\test_training.py -v
```

- [ ] **Step 5: Commit**

```powershell
git add src/srag_api/ml/__init__.py tests/unit/ml/test_training.py
git commit -m "feat: exporta api de treinamento ml"
```

---

### Task 9: Script CLI fino

**Files:**
- Create: `scripts/train_ml_admission.py`
- Test: `tests/integration/test_ml_training_pipeline.py`

**Interfaces:**
- CLI:
  - `--parquet-glob`
  - `--output-dir`
  - `--validation-year`
  - `--test-year`

- [ ] **Step 1: Write first failing integration test**

O teste deve evitar a base real.

Criar pequeno dataframe sintético com registros em:

```text
2023
2024
2025
2026
```

e ambas as classes em cada partição necessária.

Fluxo testado:

```python
dataset = build_admission_dataset(df)
split = temporal_split(dataset.metadata["ANO"])
result = run_admission_training(dataset, split)
paths = save_training_artifacts(...)
```

Assertions:

```text
4 candidates
best_model_name is valid
threshold between 0 and 1
validation metrics exist
test metrics exist
artifact files exist
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests\integration\test_ml_training_pipeline.py -v
```

- [ ] **Step 3: Implement CLI**

A CLI deverá:

```text
load normalized parquet files
build admission dataset
create temporal split
run training
save artifacts
print concise summary
```

A lógica científica deve permanecer nos módulos, não no script.

- [ ] **Step 4: Run integration GREEN**

```powershell
python -m pytest tests\integration\test_ml_training_pipeline.py -v
```

- [ ] **Step 5: Run all ML tests**

```powershell
python -m pytest tests\unit\ml tests\integration\test_ml_dataset_pipeline.py tests\integration\test_ml_training_pipeline.py -v
```

- [ ] **Step 6: Commit**

```powershell
git add scripts/train_ml_admission.py tests/integration/test_ml_training_pipeline.py
git commit -m "feat: adiciona execucao ponta a ponta do treino ml"
```

---

### Task 10: Gitignore e documentação

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Add artifact ignore rule**

Adicionar a regra no `.gitignore` aplicável ao subprojeto:

```gitignore
artifacts/ml-admission/
```

- [ ] **Step 2: Document training usage**

Adicionar seção ao README com:

```powershell
python .\scripts\train_ml_admission.py --parquet-glob "data/parquet/srag/ano=*/srag.parquet"
```

Documentar:

- treino 2019–2024;
- validação 2025;
- teste 2026;
- 4 modelos;
- AUC-PR principal;
- política de limiar;
- diretório dos artefatos;
- 2026 parcial e out-of-time.

- [ ] **Step 3: Verify artifact directory is ignored**

```powershell
git check-ignore artifacts/ml-admission/teste/metrics.json
```

Expected: caminho reconhecido como ignorado.

- [ ] **Step 4: Run ML suite**

```powershell
python -m pytest tests\unit\ml tests\integration\test_ml_dataset_pipeline.py tests\integration\test_ml_training_pipeline.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add README.md .gitignore
git commit -m "docs: documenta treinamento ml de admissao"
```

---

### Task 11: Auditoria metodológica final

**Files:**
- No production changes expected unless audit finds a defect.

- [ ] **Step 1: Audit leakage strings**

```powershell
git grep -n -E "EVOLUCAO|DT_EVOLUCA|UTI|SUPORT_VEN|QTD_DIAS|DIAS_INTERNA|PCR_EVOLUCAO" -- src/srag_api/ml tests/unit/ml tests/integration
```

Expected: ocorrências apenas em:

- `LEAKAGE_FEATURES`;
- testes que provam bloqueio;
- documentação explícita de leakage.

Nenhuma ocorrência dessas colunas deve entrar em feature registry de admissão ou treinamento.

- [ ] **Step 2: Audit temporal boundaries**

```powershell
git grep -n -E "2025|2026|validation|test" -- src/srag_api/ml/training.py src/srag_api/ml/threshold.py
```

Revisar manualmente:

```text
2025 -> seleção de modelo + limiar
2026 -> avaliação final somente
```

- [ ] **Step 3: Run full ML suite**

```powershell
python -m pytest tests\unit\ml tests\integration\test_ml_dataset_pipeline.py tests\integration\test_ml_training_pipeline.py -q
```

- [ ] **Step 4: Run entire repository suite**

```powershell
python -m pytest -q
```

Expected: zero failures. O warning conhecido do FastAPI/Starlette pode permanecer enquanto não for tratado em tarefa separada.

- [ ] **Step 5: Check repository state**

```powershell
git status
git diff --stat
git log --oneline -15
```

Expected: working tree clean after all task commits.

---

### Task 12: Primeira execução na base real

**Files:**
- No code change unless a reproducible defect is found.

**Interfaces:**
- Consumes finalized CLI.
- Produces first experimental artifacts outside Git.

- [ ] **Step 1: Confirm normalized parquet location**

```powershell
Get-ChildItem .\data\parquet\srag -Recurse -Filter *.parquet | Select-Object -First 10 FullName
```

- [ ] **Step 2: Run first real experiment**

```powershell
python .\scripts\train_ml_admission.py --parquet-glob "data/parquet/srag/ano=*/srag.parquet"
```

- [ ] **Step 3: Inspect generated summary**

Confirmar:

```text
train records
validation records
test records
mortality prevalence per partition
AUC-PR of all four models
winner
threshold
threshold policy
2026 final metrics
artifact directory
```

- [ ] **Step 4: Verify artifacts are not tracked**

```powershell
git status
git check-ignore artifacts/ml-admission/*
```

- [ ] **Step 5: Record experiment interpretation separately**

Não alterar o modelo com base em 2026 nesta fase.

Se os resultados de 2026 forem ruins, registrar a observação para a próxima fase metodológica. Não voltar e ajustar esta V1 usando o test set.

---

## Final Verification Checklist

Antes de considerar a fase concluída:

- [ ] Todos os quatro modelos existem e são treináveis.
- [ ] Logistic Regression usa pesos de classe.
- [ ] Random Forest usa pesos de classe.
- [ ] HistGradientBoosting usa pesos de classe.
- [ ] GradientBoosting recebe `sample_weight` derivado apenas de `y_train`.
- [ ] Preprocessador recebe `fit` apenas no treino.
- [ ] AUC-PR é o critério único primário para seleção.
- [ ] Limiar é escolhido somente em validação.
- [ ] Política principal exige `precision >= 0.50`.
- [ ] Fallback F1 é explícito.
- [ ] 2026 não participa de nenhuma decisão.
- [ ] Métricas de 2025 dos quatro modelos são preservadas.
- [ ] Métricas finais de 2026 são preservadas.
- [ ] Melhor pipeline + threshold podem ser carregados por joblib.
- [ ] Metadados de reprodutibilidade são persistidos.
- [ ] Diretório experimental é ignorado pelo Git.
- [ ] Integração sintética passa.
- [ ] Suíte completa do projeto passa.
