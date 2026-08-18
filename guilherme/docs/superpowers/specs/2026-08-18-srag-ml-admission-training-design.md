# SRAG ML — Treinamento e Avaliação de Mortalidade na Admissão

**Data:** 2026-08-18  
**Status:** Design aprovado  
**Escopo:** V1 do treinamento e avaliação de modelos de Machine Learning para predição de óbito por SRAG na admissão/notificação.

## 1. Objetivo

Implementar, dentro de `srag_api.ml`, uma camada modular e testável para treinamento, comparação, seleção e avaliação final de modelos de Machine Learning usando exclusivamente o dataset seguro de admissão já construído no projeto.

A pergunta de pesquisa permanece:

> Dado um paciente hospitalizado por SRAG, qual a probabilidade de óbito por SRAG usando apenas informações disponíveis até a notificação/admissão?

Esta fase não altera a definição do alvo, o registro de features, a política anti-leakage ou o split temporal já implementados.

## 2. Princípios metodológicos

A implementação deve preservar as seguintes garantias:

- nenhuma variável pós-admissão pode entrar no modelo;
- `OBITO_OUTRAS_CAUSAS` não é classificado como cura e permanece excluído do treino;
- preprocessamento é ajustado exclusivamente no conjunto de treino;
- balanceamento ou pesos de classe são aplicados apenas ao treino;
- validação de 2025 é usada para comparar modelos e escolher limiar;
- 2026 permanece completamente fora de seleção de modelo, ajuste de limiar, tuning e definição de features;
- o teste de 2026 é uma avaliação out-of-time final;
- ausência de coluna no schema não pode ser convertida artificialmente em resposta negativa;
- todos os experimentos devem ser reproduzíveis.

## 3. Partições temporais

A divisão padrão permanece:

- **Treino:** 2019–2024
- **Validação:** 2025
- **Teste out-of-time:** 2026

O fluxo obrigatório é:

```text
Dados SRAG normalizados
        ↓
build_admission_dataset()
        ↓
temporal_split()
        ↓
2019–2024   2025   2026
   treino   val.   teste final
        ↓
preprocessamento fit somente no treino
        ↓
treino dos modelos
        ↓
avaliação em 2025
        ↓
seleção por AUC-PR
        ↓
seleção de limiar em 2025
        ↓
modelo + limiar congelados
        ↓
avaliação única em 2026
```

## 4. Modelos da V1

A V1 compara quatro modelos clássicos disponíveis no scikit-learn:

1. `LogisticRegression`
2. `RandomForestClassifier`
3. `GradientBoostingClassifier`
4. `HistGradientBoostingClassifier`

Não serão usados PCA, redes neurais, XGBoost, LightGBM ou stacking nesta V1.

### 4.1 Regressão Logística

A Regressão Logística funciona como baseline interpretável.

Configuração inicial:

- `class_weight="balanced"`
- `max_iter` suficientemente alto para convergência
- `random_state` fixo quando aplicável

### 4.2 Random Forest

Configuração inicial:

- pesos de classe para compensar desbalanceamento;
- `random_state` fixo;
- paralelismo permitido;
- hiperparâmetros fixos nesta V1.

### 4.3 Gradient Boosting

Como `GradientBoostingClassifier` não oferece `class_weight` diretamente, deve receber `sample_weight` calculado apenas a partir do conjunto de treino.

Validação e teste nunca recebem pesos usados para ajuste do modelo.

### 4.4 HistGradientBoosting

Configuração inicial:

- `class_weight="balanced"` quando suportado pela versão instalada;
- `random_state` fixo;
- hiperparâmetros fixos nesta V1.

## 5. Política de desbalanceamento

A V1 não usa SMOTE nem qualquer oversampling.

A estratégia inicial é:

- Regressão Logística: `class_weight="balanced"`
- Random Forest: `class_weight="balanced"` ou equivalente suportado
- HistGradientBoosting: `class_weight="balanced"`
- Gradient Boosting: `sample_weight` calculado somente no treino

SMOTE poderá ser estudado posteriormente como experimento separado, nunca misturado à baseline V1.

## 6. PCA

PCA não será usado nesta V1.

Motivos:

- preservação da interpretabilidade;
- redução de complexidade metodológica;
- PCA não beneficia necessariamente modelos baseados em árvores;
- a matriz one-hot pode ser analisada diretamente;
- resultados ficam mais fáceis de relacionar às variáveis clínicas originais.

Uma variante com PCA poderá ser testada futuramente como experimento independente.

## 7. Hiperparâmetros

A V1 usa hiperparâmetros fixos e documentados.

Não haverá `GridSearchCV`, `RandomizedSearchCV`, Optuna ou equivalente nesta fase.

O objetivo inicial é construir uma baseline metodologicamente limpa e comparável.

Tuning será uma etapa posterior e deverá usar somente dados de treino, sem contaminar 2025 ou 2026.

## 8. Métricas

A métrica principal para seleção do melhor modelo é:

- **AUC-PR / Average Precision**

Métricas auxiliares:

- ROC-AUC
- Recall
- Precision
- F1
- matriz de confusão

AUC-PR será usada como métrica principal porque o evento de interesse pode ser desbalanceado e a métrica é mais informativa nesse cenário do que accuracy isolada.

Accuracy não deve ser usada como critério principal de seleção.

## 9. Seleção do melhor modelo

Os quatro modelos são treinados exclusivamente em 2019–2024.

Cada modelo gera probabilidades para 2025.

As métricas de 2025 são calculadas e registradas.

O modelo vencedor é aquele com maior AUC-PR em 2025.

Empates devem ser resolvidos de maneira determinística e documentada. A implementação pode usar a ordem fixa do registro dos modelos como critério secundário, salvo decisão posterior explicitamente documentada.

## 10. Seleção do limiar

Depois de selecionar o modelo vencedor, o limiar de decisão é escolhido usando somente 2025.

Política principal:

> Maximizar o recall da classe óbito entre os limiares que mantenham `precision >= 0.50`.

O valor `0.50` é a restrição inicial aprovada para a V1.

### 10.1 Fallback

Se nenhum limiar atender `precision >= 0.50`, a implementação não deve silenciosamente alterar a regra.

Nesse caso:

- escolher o limiar que maximize F1 em 2025;
- registrar explicitamente:

```text
threshold_policy = "fallback_max_f1"
```

Caso a política principal seja satisfeita, registrar:

```text
threshold_policy = "max_recall_precision_ge_0_50"
```

A escolha do limiar nunca usa dados de 2026.

## 11. Avaliação final em 2026

Após seleção do modelo e do limiar em 2025:

- o pipeline vencedor é considerado congelado;
- o limiar é considerado congelado;
- 2026 é usado uma única vez para avaliação final;
- nenhuma decisão metodológica pode ser alterada com base nos resultados de 2026.

Em 2026 devem ser calculados:

- AUC-PR
- ROC-AUC
- Recall
- Precision
- F1
- matriz de confusão

## 12. Arquitetura de código

A lógica de treinamento deve permanecer dentro do pacote `srag_api.ml`.

Estrutura planejada:

```text
src/srag_api/ml/
├── __init__.py
├── features.py
├── target.py
├── dataset.py
├── split.py
├── preprocessing.py
├── models.py
├── metrics.py
├── threshold.py
├── training.py
└── artifacts.py
```

Responsabilidades:

### `models.py`

- registrar os quatro modelos;
- centralizar hiperparâmetros fixos;
- manter `random_state` reproduzível;
- encapsular diferenças de política de peso entre estimadores.

### `metrics.py`

- calcular métricas a partir de `y_true`, probabilidades e limiar;
- produzir representação estruturada de métricas;
- gerar matriz de confusão.

### `threshold.py`

- selecionar limiar exclusivamente na validação;
- aplicar política `max_recall_precision_ge_0_50`;
- aplicar fallback `fallback_max_f1`;
- retornar limiar, política e métricas associadas.

### `training.py`

- receber dataset e split temporal;
- ajustar preprocessador somente no treino;
- treinar os quatro modelos;
- avaliar todos em 2025;
- selecionar vencedor por AUC-PR;
- escolher limiar em 2025;
- avaliar vencedor em 2026;
- retornar um resultado estruturado da execução.

### `artifacts.py`

- persistir modelo, limiar, métricas e metadados;
- criar diretório de execução;
- salvar arquivos em formatos legíveis e reproduzíveis.

## 13. Script de execução

Será criado um script fino, por exemplo:

```text
scripts/train_ml_admission.py
```

O script não deve conter a lógica científica principal.

Ele deve apenas:

1. localizar/carregar os dados normalizados;
2. montar o dataset seguro;
3. chamar a API de treinamento;
4. persistir os artefatos;
5. imprimir um resumo da execução.

## 14. Artefatos da execução

Cada execução deverá criar uma pasta própria, por exemplo:

```text
artifacts/ml-admission/2026-08-18_1600/
```

Arquivos planejados:

```text
best_model.joblib
metrics.json
metrics.csv
validation_comparison.csv
confusion_matrix_validation.csv
confusion_matrix_test.csv
run_metadata.json
```

Os nomes exatos poderão ser ajustados durante a implementação se houver motivo técnico, desde que as mesmas informações sejam preservadas.

## 15. Conteúdo dos artefatos

### `best_model.joblib`

Deve conter o necessário para reproduzir a predição:

- preprocessamento já ajustado;
- modelo vencedor;
- limiar escolhido em 2025;
- lista de features esperadas;
- metadados mínimos necessários para validação da entrada.

### `metrics.json`

Deve registrar, no mínimo:

- métrica de seleção;
- modelo vencedor;
- limiar;
- política de limiar;
- métricas de validação;
- métricas de teste.

Exemplo estrutural:

```json
{
  "selection_metric": "average_precision",
  "validation_year": 2025,
  "test_year": 2026,
  "best_model": "hist_gradient_boosting",
  "threshold": 0.37,
  "threshold_policy": "max_recall_precision_ge_0_50",
  "validation": {
    "auc_pr": 0.71,
    "roc_auc": 0.83,
    "recall": 0.78,
    "precision": 0.53,
    "f1": 0.63
  },
  "test": {
    "auc_pr": 0.68,
    "roc_auc": 0.81,
    "recall": 0.75,
    "precision": 0.51,
    "f1": 0.61
  }
}
```

Os valores acima são apenas ilustrativos.

### `validation_comparison.csv`

Deve guardar as métricas dos quatro modelos em 2025, permitindo comparação posterior em relatório, artigo ou TCC.

### Matrizes de confusão

Devem ser salvas separadamente para validação e teste.

## 16. Metadados de reprodutibilidade

`run_metadata.json` deve incluir, no mínimo:

- timestamp da execução;
- `random_state`;
- anos de treino;
- ano de validação;
- ano de teste;
- features efetivamente usadas;
- features configuradas mas ausentes;
- quantidade de registros por partição;
- prevalência de óbito por partição;
- modelo vencedor;
- hiperparâmetros do vencedor;
- métrica de seleção;
- política de limiar;
- versão do Python;
- versão do pandas;
- versão do scikit-learn;
- versão do projeto quando disponível.

## 17. Política de versionamento dos artefatos

Artefatos experimentais não devem ser versionados no Git por padrão.

O `.gitignore` deverá ignorar os diretórios de execução de ML.

Código, testes, spec e documentação entram no repositório.

Resultados consolidados só devem ser versionados futuramente quando houver decisão explícita de usá-los em relatório, artigo ou outro produto científico.

## 18. Tratamento de erros

A execução deve falhar com mensagem clara quando:

- não houver dados de treino;
- não houver dados de validação;
- não houver dados de teste;
- uma partição necessária possuir apenas uma classe;
- o alvo não puder ser construído;
- nenhuma feature válida estiver disponível;
- houver tentativa de incluir feature de leakage;
- o preprocessador ou modelo não puder ser ajustado;
- o diretório de artefatos não puder ser criado.

Falhas não devem ser mascaradas por valores default silenciosos.

## 19. Proteções contra leakage

Devem existir testes explícitos garantindo que:

- 2025 nunca participa de `fit` do preprocessador;
- 2026 nunca participa da seleção do modelo;
- 2026 nunca participa da seleção do limiar;
- `LEAKAGE_FEATURES` nunca chega ao treinamento;
- `OBITO_OUTRAS_CAUSAS` permanece fora do dataset elegível;
- coluna ausente não é inventada como resposta negativa;
- pesos de classe ou `sample_weight` são derivados somente do treino;
- o limiar é escolhido somente a partir de 2025.

## 20. Estratégia de testes

A implementação seguirá TDD.

Testes unitários planejados:

```text
tests/unit/ml/test_models.py
tests/unit/ml/test_metrics.py
tests/unit/ml/test_threshold.py
tests/unit/ml/test_training.py
tests/unit/ml/test_artifacts.py
```

Teste de integração planejado:

```text
tests/integration/test_ml_training_pipeline.py
```

O teste de integração usará um dataset sintético pequeno e deve cobrir:

```text
dataset
→ split temporal
→ preprocessamento
→ treino dos 4 modelos
→ avaliação em 2025
→ seleção do vencedor
→ seleção de limiar
→ avaliação em 2026
→ persistência de artefatos
```

Os testes não devem depender da base completa do SIVEP para continuarem rápidos e determinísticos.

## 21. Relação com implementações de outros integrantes

Esta implementação é independente de scripts externos ao módulo `guilherme/src/srag_api/ml/`.

Código de outros integrantes pode ser analisado como referência comparativa, mas não define a metodologia desta V1.

Em particular, esta V1 mantém:

- alvo baseado no desfecho normalizado do projeto;
- exclusão de óbitos por outras causas;
- bloqueio de `UTI`, `SUPORT_VEN` e outras features pós-admissão;
- validação temporal;
- limiar escolhido apenas em validação;
- 2026 como teste final intocado.

## 22. Fora de escopo nesta V1

Não fazem parte desta implementação:

- SMOTE;
- PCA;
- tuning automático de hiperparâmetros;
- XGBoost;
- LightGBM;
- redes neurais;
- stacking;
- calibração probabilística;
- SHAP;
- implantação do modelo na API;
- inferência online;
- monitoramento de drift;
- re-treinamento automático.

Esses itens podem ser tratados em fases posteriores.

## 23. Critérios de aceite

A fase será considerada concluída quando:

1. os quatro modelos forem treináveis via API modular;
2. o preprocessamento for ajustado somente no treino;
3. os quatro modelos forem comparados em 2025;
4. o vencedor for selecionado por AUC-PR;
5. o limiar for escolhido somente em 2025;
6. o teste final de 2026 ocorrer apenas após congelamento do modelo e do limiar;
7. métricas e matrizes de confusão forem persistidas;
8. o melhor pipeline for salvo em `.joblib`;
9. metadados de reprodutibilidade forem persistidos;
10. artefatos experimentais forem ignorados pelo Git;
11. testes unitários e de integração cobrirem as proteções metodológicas;
12. a suíte completa do projeto permanecer verde.

## 24. Decisões aprovadas

Resumo das decisões aprovadas para a V1:

- arquitetura modular em `srag_api.ml`;
- quatro modelos clássicos do scikit-learn;
- sem PCA;
- sem SMOTE;
- pesos de classe apenas no treino;
- hiperparâmetros fixos;
- AUC-PR como métrica principal;
- seleção de modelo em 2025;
- limiar com máximo recall sujeito a `precision >= 0.50`;
- fallback por máximo F1;
- 2026 como teste out-of-time final;
- salvamento de modelo, limiar, métricas e metadados;
- artefatos de execução fora do Git.
