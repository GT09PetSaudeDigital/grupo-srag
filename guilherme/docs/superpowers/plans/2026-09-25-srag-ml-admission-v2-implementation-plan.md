# SRAG ML V2 — Predição de Óbito na Admissão Implementation Plan

> **Para agentes:** usar execução task por task com checkbox (`- [ ]`) para
> acompanhamento. Cada task segue TDD: teste vermelho, implementação mínima,
> teste verde, commit.

**Objetivo:** Corrigir os defeitos de dados e de protocolo apontados em
`docs/auditoria-ml-admissao-2019-2025.md` e estabelecer um protocolo de avaliação
que responda à pergunta operacional antes de comparar modelos.

**Arquitetura:** O pacote `srag_api.ml` passa a carregar a origem temporal de
cada feature, tratar ausência de forma explícita e configurável, e expor um
protocolo de avaliação que inclui calibração, volume de alertas, subgrupos e
incerteza. Nenhuma comparação de modelo é executada antes de o objetivo
operacional estar documentado.

**Stack:** Python 3, pandas, scikit-learn, pytest. Nenhuma dependência nova.

**Spec:** `docs/superpowers/specs/2026-09-25-srag-ml-admission-v2-design.md`

## Restrições globais

- O artefato da V1 (`artifacts/ml-admission/20260906-104505/`) permanece
  inalterado e não é reescrito.
- 2026 já foi usado como teste na V1 e não pode ser reutilizado como teste
  intocado. A escolha entre reavaliação e 2027 é registrada antes da
  execução.
- Nenhuma feature cuja disponibilidade não verificável entre em `X`.
- A regra de split temporal e o bloqueio de leakage da V1 continuam válidos.
- Nenhum ajuste de hiperparâmetro antes das tasks 1 a 6.
- A suíte completa do projeto deve continuar passando.

## Mapa de arquivos

### Criar

- `src/srag_api/ml/availability.py` — origem temporal das features e
  verificação de disponibilidade.
- `src/srag_api/ml/missingness.py` — representações de ausência configuráveis
  e detecção de features constantes.
- `src/srag_api/ml/episode.py` — chave de episódio e política de sobreposição.
- `src/srag_api/ml/evaluation.py` — calibração, subgrupos, volume de alertas e
  intervalos de confiança.
- `docs/decisoes/objetivo-operacional-alerta.md` — a pré-condição da seção 3.
- `tests/unit/ml/test_availability.py`
- `tests/unit/ml/test_missingness.py`
- `tests/unit/ml/test_episode.py`
- `tests/unit/ml/test_evaluation.py`
- `tests/unit/ml/test_temporal_features.py`

### Modificar

- `src/srag_api/ml/features.py` — `IDADE_ANOS` no lugar de `NU_IDADE_N`;
  remover features não materializadas; carregar origem temporal.
- `src/srag_api/ml/dataset.py` — aplicar a política de episódio antes do
  split e expor as features faltantes.
- `src/srag_api/ml/training.py` — `NUMERIC_MODEL_FEATURES` coerente com a
  idade em anos; receber a configuração de ausência.
- `src/srag_api/ml/threshold.py` — parametrizar a política a partir do
  objetivo documentado.
- `src/srag_api/ml/__init__.py` — exportar a API nova.
- `scripts/train_ml_admission.py` — expor as opções novas.
- `scripts/audit_ml_admission.py` — reexecutar sobre a base corrigida.
- `README.md`, `docs/relatorio-ml-admissao-v2.md`.

---

### Task 1: Objetivo operacional do alerta

**Por que primeiro:** a métrica de seleção e a política de limiar derivam
disso. Sem ele, as tasks 7 e 8 não têm critério.

**Files:**
- Create: `docs/decisoes/objetivo-operacional-alerta.md`
- Create: `src/srag_api/ml/operational.py`
- Test: `tests/unit/ml/test_operational.py`

**Produz:**
- `ALERT_OBJECTIVE_TEMPLATE` com os campos obrigatórios da seção 3 da spec.
- `AlertObjective` dataclass: `use_case`, `false_negative_cost`,
  `false_positive_cost`, `target_alerts_per_1000`, `author`, `decided_on`.
- `load_alert_objective(path) -> AlertObjective` que falha se o documento
  não preencher todos os campos.

- [ ] **Step 1: escrever testes que exigem documento completo**

```python
def test_missing_use_case_is_rejected(tmp_path):
    path = tmp_path / "objetivo.md"
    path.write_text(ALERT_OBJECTIVE_TEMPLATE.format(use_case=""), encoding="utf-8")

    with pytest.raises(ValueError, match="use_case"):
        load_alert_objective(path)
```

- [ ] **Step 2: run RED**

```powershell
pytest tests/unit/ml/test_operational.py -v
```

- [ ] **Step 3: implementar `operational.py` e escrever o documento real**

O documento deve ser preenchido com a decisão do usuário, não com um exemplo.
Se ele ainda não tiver decidido, a task para aqui e nenhuma task de métricas
ou limiar começa.

- [ ] **Step 4: run GREEN**
- [ ] **Step 5: commit**

```powershell
git add src/srag_api/ml/operational.py tests/unit/ml/test_operational.py docs/decisoes/
git commit -m "docs: define o objetivo operacional do alerta de risco"
```

---

### Task 2: Origem temporal e momento da predição

**Files:**
- Create: `src/srag_api/ml/availability.py`
- Create: `tests/unit/ml/test_availability.py`

**Produz:**
- `PredictionInstant` com `ADMISSION` e `NOTIFICATION`.
- `FeatureAvailability` dataclass: `column`, `available_at`.
- `annotate_availability(columns, instant) -> list[FeatureAvailability]`
- `unavailable_at_instant(columns, instant) -> list[str]`
- `TIMESTAMP_SOURCE: dict[str, str]` mapeando cada coluna para a data que a
  origina.

Regras de disponibilidade:

- `ADMISSION`: disponível o que for conhecido em `DT_INTERNA`;
- `NOTIFICATION`: disponível o que for conhecido em `DT_NOTIFIC`.

- [ ] **Step 1: teste vermelho** — uma coluna de evolução não pode ser
  disponível em `ADMISSION`; uma coluna de sintoma, sim.
- [ ] **Step 2: run RED**
- [ ] **Step 3: implementar**
- [ ] **Step 4: run GREEN**
- [ ] **Step 5: teste deve falhar se alguém declarar `EVOLUCAO` disponível na
  admissão**
- [ ] **Step 6: commit**

```powershell
git commit -m "feat: declara a origem temporal de cada feature"
```

---

### Task 3: Idade em anos no contrato do ML

**Files:**
- Modify: `src/srag_api/ml/features.py`
- Modify: `src/srag_api/ml/training.py` (`NUMERIC_MODEL_FEATURES`)
- Test: `tests/unit/ml/test_features.py`, `tests/unit/ml/test_age_in_ml.py`

**Regras:**

- `DEMOGRAPHIC_FEATURES` passa a conter `IDADE_ANOS` e deixa de conter
  `NU_IDADE_N`;
- `TP_IDADE` nunca entra;
- `validate_feature_registry()` continua exigindo discusses com
  `LEAKAGE_FEATURES`.

- [ ] **Step 1: teste vermelho** — `NU_IDADE_N` ausente do catálogo,
  `IDADE_ANOS` presente, ambas numéricas.
- [ ] **Step 2: run RED**
- [ ] **Step 3: implementar**
- [ ] **Step 4: teste de conversão** — 30 dias, 18 meses, 67 anos, valor
  negativo, valor acima do limite, `TP_IDADE` inválido.
- [ ] **Step 5: run GREEN**
- [ ] **Step 6: suite completa**
- [ ] **Step 7: commit**

```powershell
git commit -m "fix: usa idade em anos no catalogo do ml de admissao"
```

> A V1 usava `NU_IDADE_N` cru. O efeito em AUC-PR e recall é desconhecido e
> deve ser medido na execução, não presumido.

---

### Task 4: Regras de data e features temporais

**Files:**
- Create: `tests/unit/ml/test_temporal_features.py`
- Modify: `src/srag_api/ml/features.py`
- Modify: `src/srag_api/data/clean.py` (se a materialização couber melhor lá)

**Produz:**

- `plausible_year_range(min_year, max_year)` — regra explícita, configurável;
  os anos 2109, 2202 e 2224 da auditoria caem fora.
- `add_temporal_interval_columns(df, instant)` — materializa
  `SINT_ATE_ADM` ou `SINT_ATE_NOTIF` conforme o instante, em dias, com
  inteiro quando integral e ausente quando qualquer das datas faltar.
- As linhas cujo intervalo viola a ordem hipotetizada são contadas e
  reportadas, nunca silenciosamente corrigidas.

- [ ] **Step 1: testes vermelhos** — anos implausíveis, datas não
  reconhecidas, sintoma posterior à internação, intervalo negativo.
- [ ] **Step 2: run RED**
- [ ] **Step 3: implementar**
- [ ] **Step 4: run GREEN**
- [ ] **Step 5: remover do catálogo qualquer feature que o pipeline não
  materialize** — `REGIAO` e `SINT_ATE_NOTIF` estão no catálogo da V1 e nunca
  existiram como coluna.
- [ ] **Step 6: commit**

```powershell
git commit -m "feat: materializa a feature temporal conforme o instante de predicao"
```

---

### Task 5: Identidade do episódio e sobreposição entre anos

**Files:**
- Create: `src/srag_api/ml/episode.py`
- Create: `tests/unit/ml/test_episode.py`
- Modify: `src/srag_api/ml/dataset.py`

**Produz:**

- `EPISODE_KEY_COLUMNS` e `build_episode_key(df)` — a chave candidata da
  auditoria é `NU_NOTIFIC + SG_UF_NOT + CO_MUN_NOT + DT_NOTIFIC`, com a
  política declarada para componentes nulos (8.434 linhas ficaram fora).
- `overlap_report(df)` — contagem de chaves repetidas dentro e entre anos.
- `apply_overlap_policy(df, policy)` — política explícita e testada, aplicada
  **antes** da divisão temporal, que devolve os registros mantidos e as linhas
  afetadas para relatório.

A política **não** exclui registros por padrão. Por padrão ela reporta e
mantém, porque a chave é evidência de sobreposição e não prova de duplicidade
de paciente. Excluir é uma escolha explícita e registrada.

- [ ] **Step 1: testes vermelhos** — repetição dentro do mesmo ano, entre
  anos, componentes nulos, conflito de desfecho.
- [ ] **Step 2: run RED**
- [ ] **Step 3: implementar**
- [ ] **Step 4: run GREEN**
- [ ] **Step 5: teste de integração** — a sobreposição é resolvida antes do
  `temporal_split`, e nenhuma chave aparece em duas partições quando a
  política de exclusão está ativa.
- [ ] **Step 6: commit**

```powershell
git commit -m "feat: trata a identidade do episodio antes da divisao temporal"
```

---

### Task 6: Ausência, ignorado e features constantes

**Files:**
- Create: `src/srag_api/ml/missingness.py`
- Create: `tests/unit/ml/test_missingness.py`
- Modify: `src/srag_api/ml/preprocessing.py`
- Modify: `src/srag_api/ml/training.py`

**Produz:**

- `MissingnessStrategy` com `most_frequent`, `explicit_category` e
  `explicit_category_with_indicators`.
- `build_missingness_strategy(strategy)` → o imputer correspondente.
- `constant_features(X, y) -> list[str]` — detecta features constantes após
  o tratamento, incluindo `FATOR_RISC`.
- `normalise_sivp_yes_no(series)` — `1/2/9` e ausente viram categorias
  `SIM`/`NAO`/`IGNORADO`/`AUSENTE` sem nunca converter vazio em "não".

A escolha da estratégia vem da comparação medida, não de convenção. A task
inclui rodar as três sobre a base e reportar; a adoção é uma decisão
registrada.

- [ ] **Step 1: testes vermelhos** — `9` continua categoria própria, ausente
  não vira `NAO`, `FATOR_RISC` é detectada como constante.
- [ ] **Step 2: run RED**
- [ ] **Step 3: implementar**
- [ ] **Step 4: run GREEN**
- [ ] **Step 5: suite completa**
- [ ] **Step 6: commit**

```powershell
git commit -m "feat: torna explicito o tratamento de ausencia e ignorado"
```

---

### Task 7: Calendário de partição e janelas temporais

**Files:**
- Modify: `src/srag_api/ml/split.py`
- Test: `tests/unit/ml/test_split.py`

**Produz:**

- `partition_calendar(metadata) -> dict` com o intervalo real de início de
  sintomas e de notificação por partição. O arquivo `ano=2025` cobre sintomas
  de 29/12/2024 a 03/01/2026 e notificações até 27/07/2026.
- `temporal_windows(metadata, windows)` — mais de uma janela de validação,
  para a conclusão não depender de um único ponto de corte no tempo.
- `temporal_split` deixa de exigir que o ano de teste exista por padrão; a
  exigência vira opt-in, porque hoje ela impede qualquer experimento antes de
  existir registro do ano.
- `ordering_violations(metadata, split) -> int` — quantos registros de
  validação e de teste são cronologicamente **anteriores** a registros de
  treino, usando a data efetiva e não o ano do arquivo. A partição por ano de
  arquivo não garante ordem no tempo, e essa contagem precisa ser conhecida e
  reportada, não presumida.
- `temporal_split` aceita uma coluna de data (`DT_NOTIFIC` ou
  `DATA_INICIO_SINTOMAS`) como critério primário de ordenação, com `ANO` apenas
  como agrupamento secundário.

- [ ] **Step 1: testes vermelhos** — calendário reflete os extremos reais;
  partição sem ano de teste é permitida quando pedido; janelas múltiplas não
  se sobrepõem; `ordering_violations` detecta inversão quando a janela de
  validação começa antes do fim da de treino.
- [ ] **Step 2: run RED**
- [ ] **Step 3: implementar**
- [ ] **Step 4: run GREEN**
- [ ] **Step 5: commit**

```powershell
git commit -m "feat: explicita o calendario e permite multiplas janelas temporais"
```

---

### Task 8: Protocolo de avaliação

**Files:**
- Create: `src/srag_api/ml/evaluation.py`
- Create: `tests/unit/ml/test_evaluation.py`

**Produz:**

- `calibration_metrics(y_true, probabilities) -> dict` com Brier score e
  curva de confiabilidade por faixa.
- `bootstrap_confidence_interval(y_true, probabilities, metric, n_resamples)`
  — a V1 não reportou nenhum intervalo.
- `alert_volume(y_true, probabilities, threshold) -> dict` com alertas por mil
  admissions, PPV e NPV.
- `subgroup_metrics(metadata_frame, y_true, probabilities, threshold, by)` —
  por faixa etária, UF e ano.
- `evaluate_full(...) -> EvaluationReport` agregando tudo.

- [ ] **Step 1: testes vermelhos** — Brier de probabilidades perfeitas é 0;
  volume de alertas confere com a contagem manual; intervalos contêm o valor
  pontual na maioria das repetições.
- [ ] **Step 2: run RED**
- [ ] **Step 3: implementar**
- [ ] **Step 4: run GREEN**
- [ ] **Step 5: commit**

```powershell
git commit -m "feat: adiciona calibracao, subgrupos e incerteza a avaliacao"
```

---

### Task 9: Limiar derivado do objetivo documentado

**Files:**
- Modify: `src/srag_api/ml/threshold.py`
- Test: `tests/unit/ml/test_threshold.py`

- [ ] **Step 1: testes vermelhos** — a política escolhida responde ao
  `AlertObjective`; `max_recall_precision_ge_0_50` continua disponível como
  uma das opções, mas deixa de ser o padrão implícito.
- [ ] **Step 2: run RED**
- [ ] **Step 3: implementar** — cada política reporta, além do limiar, o
  volume de alertas e o custo esperado sob os pesos da task 1.
- [ ] **Step 4: run GREEN**
- [ ] **Step 5: commit**

```powershell
git commit -m "feat: deriva a politica de limiar do objetivo operacional"
```

---

### Task 10: Reexecução sobre a base nacional

**Files:**
- Modify: `scripts/train_ml_admission.py`, `scripts/audit_ml_admission.py`
- Create: `docs/relatorio-ml-admissao-v2.md`

- [ ] **Step 1: reexecutar a auditoria** sobre os anos cobertos e comparar com
  `artifacts/audit-ml-admission/2019-2025-v1/`. As contagens de sobreposição
  devem cair a zero na fronteira treino–validação.
- [ ] **Step 2: treinar** com a configuração da V2.
- [ ] **Step 3: reportar** as três estratégias de ausência, a comparação com a
  V1 e o efeito real da idade em anos, sem presumir melhora.
- [ ] **Step 4: declarar** se 2026 foi usada como teste de desenvolvimento ou
  reservada, e o recorte temporal efetivamente coberto.
- [ ] **Step 5: escrever o relatório** com os mesmos limites explícitos da V1:
  recall baixo é resultado da política, não do modelo, e os dois devem ser
  julgados separadamente.
- [ ] **Step 6: commit**

```powershell
git commit -m "docs: registra a v2 do ml de admissao sobre a base nacional"
```

---

### Task 11: Auditoria final

- [ ] `pytest -q` passa.
- [ ] `git grep -n -E "NU_IDADE_N|TP_IDADE" -- src/srag_api/ml` só encontra
  o bloqueio e os testes que garantem o bloqueio.
- [ ] `git grep -n "SINT_ATE_NOTIF" -- src/srag_api/ml` não encontra a coluna
  sem materialização.
- [ ] O artefato da V1 está intacto.
- [ ] Nenhuma feature entra em `X` sem origem temporal declarada.
- [ ] O relatório cita a limitação de que 2026 já foi visto.

## Sequência de commits esperada

```text
docs: define o objetivo operacional do alerta de risco
feat: declara a origem temporal de cada feature
fix: usa idade em anos no catalogo do ml de admissao
feat: materializa a feature temporal conforme o instante de predicao
feat: trata a identidade do episodio antes da divisao temporal
feat: torna explicito o tratamento de ausencia e ignorado
feat: explicita o calendario e permite multiplas janelas temporais
feat: adiciona calibracao, subgrupos e incerteza a avaliacao
feat: deriva a politica de limiar do objetivo operacional
docs: registra a v2 do ml de admissao sobre a base nacional
```

## Fora do escopo deste plano

Não implementar:

- estimadores além dos quatro existentes;
- ajuste de hiperparâmetros antes da task 10;
- SHAP ou LIME;
- endpoint de predição na API;
- registro ou versionamento de modelos;
- os Modelos B (laboratório) e C (evolução).

## Risco conhecido

A task 1 depende de uma decisão do usuário que ainda não foi tomada. Se o
objetivo operacional não for definido, as tasks 8 e 9 não têm critério de
aceite, e a task 10 não pode ser interpretada. Essa é a única dependência
externa do plano.
