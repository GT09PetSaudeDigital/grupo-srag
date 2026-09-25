# Relatório V1 — Predição de mortalidade por SRAG na admissão

Projeto: GT09PetSaudeDigital/grupo-srag — subprojeto guilherme.  
Execução: 6 de setembro de 2026. Identificador: `20260906-104505`.  
Branch: `feature/ml-admission-training`. Commit presente na elaboração: `4d22e3e`.

## 1. Objetivo e síntese

A V1 avaliou quatro modelos de classificação para predizer óbito por SRAG a partir das variáveis selecionadas para o contexto de admissão. A seleção do modelo e do ponto de corte foi realizada exclusivamente em 2025, seguida da avaliação temporal final nos registros disponíveis de 2026.

A execução concluiu o treinamento com 3.649.309 registros, sem os erros de memória e incompatibilidade sparse/dense anteriormente relatados. O HistGradientBoosting foi selecionado por apresentar a maior Average Precision na validação: 0,2875. No teste temporal, obteve Average Precision de 0,2530 e ROC-AUC de 0,8591.

A principal limitação foi o desempenho no ponto de corte aprovado: o recall no teste foi de 3,44%, com precision de 45,26%. Assim, o resultado demonstra capacidade de ordenação de risco, mas baixa detecção de óbitos sob a política de alertas adotada. A conclusão técnica de execução não deve ser confundida com comprovação de utilidade assistencial.

## 2. Dados e divisão temporal

Foram utilizados os Parquets locais em `data/parquet/srag/ano=*/srag.parquet`, com anos observados de 2019 a 2026. As quantidades abaixo correspondem a registros elegíveis, não a uma contagem comprovada de pacientes únicos.

| Partição | Anos | Registros | Óbitos por SRAG | Proporção de óbitos |
|---|---|---:|---:|---:|
| Treino | 2019–2024 | 3.649.309 | Não registrada nos artefatos consultados | Não calculada |
| Validação | 2025 | 305.362 | 23.023 | 7,54% |
| Teste final temporal | 2026 disponível | 154.156 | 8.747 | 5,67% |
| Total | 2019–2026 disponível | 4.108.827 | — | — |

Os totais de óbitos da validação e do teste foram derivados das matrizes de confusão. O recorte de 2026 não representa o ano completo: a execução ocorreu em setembro. A data exata de atualização da fonte não está registrada nos metadados consultados.

O alvo binário foi definido como `CURA = 0` e `OBITO_SRAG = 1`. `OBITO_OUTRAS_CAUSAS` e demais desfechos não elegíveis foram excluídos. Não houve amostragem de registros, redução dos anos previstos, SMOTE ou PCA.

## 3. Variáveis e prevenção de vazamento

Os metadados registram 26 variáveis utilizadas:

| Grupo | Variáveis |
|---|---|
| Demográficas | `CS_SEXO`, `NU_IDADE_N`, `CS_GESTANT` |
| Sintomas | `FEBRE`, `TOSSE`, `GARGANTA`, `DISPNEIA`, `DESC_RESP`, `SATURACAO`, `DIARREIA`, `VOMITO`, `DOR_ABD`, `FADIGA`, `PERD_OLFT`, `PERD_PALA`, `OUTRO_SIN` |
| Comorbidades e risco | `CARDIOPATI`, `DIABETES`, `PNEUMOPATI`, `RENAL`, `HEPATICA`, `IMUNODEPRE`, `OBESIDADE`, `OUT_MORBI`, `FATOR_RISC` |
| Geográfica | `SG_UF` |

As features previstas `REGIAO` e `SINT_ATE_NOTIF` estavam ausentes. Nesta execução, `NU_IDADE_N` foi a única feature numérica; as demais foram tratadas como categóricas.

A seleção utilizou o catálogo permitido de features. Variáveis explicitamente proibidas, como evolução, desfecho, data de evolução, UTI, suporte ventilatório e duração de internação, ficaram fora dos preditores. Ano e desfecho foram usados para partição e construção do alvo, respectivamente. Essas salvaguardas no código não substituem uma auditoria da disponibilidade real de cada campo no momento da admissão.

## 4. Pré-processamento e modelos

Regressão logística, Random Forest e Gradient Boosting compartilharam um único pré-processador ajustado no treino. As variáveis numéricas receberam imputação pela mediana e `StandardScaler`; as categóricas, imputação pela moda e `OneHotEncoder(handle_unknown="ignore")`. As matrizes transformadas de treino e validação foram reutilizadas pelos três estimadores, preservando representação sparse quando possível.

O HistGradientBoosting recebeu um pré-processador separado, também ajustado exclusivamente no treino. Foi mantido o tratamento numérico; nas categóricas, utilizou-se `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` após imputação pela moda. A saída dense compacta evita densificar a matriz one-hot. Os códigos ordinais são entradas numéricas: não foi habilitado tratamento categórico nativo do HGB. Essa representação introduz uma ordenação de códigos que deve ser considerada na interpretação do experimento.

As matrizes compartilhadas foram liberadas antes da preparação do HGB. Hiperparâmetros e ponderações dos estimadores foram preservados:

| Modelo | Parâmetros explicitamente configurados | Ponderação |
|---|---|---|
| Logistic Regression | `max_iter=2000` | `class_weight="balanced"` |
| Random Forest | `n_estimators=300`, `max_depth=12`, `n_jobs=-1` | `class_weight="balanced_subsample"` |
| Gradient Boosting | `n_estimators=200`, `learning_rate=0.05`, `max_depth=3` | `sample_weight` balanceado, calculado somente com `y_train` |
| HistGradientBoosting | `max_iter=300`, `learning_rate=0.05`, `max_depth=8` | `class_weight="balanced"` |

Todos utilizaram `random_state=42`. Os demais parâmetros permaneceram nos padrões da versão registrada do scikit-learn. A representação ordinal do HGB é uma diferença de pré-processamento em relação ao one-hot anterior; não implica equivalência numérica das predições.

## 5. Seleção e resultados de validação

A métrica principal foi Average Precision, denominada AUC-PR na saída da pipeline; não se trata de integração trapezoidal da curva precision-recall. O vencedor foi escolhido somente por essa métrica em 2025.

| Modelo | Average Precision | ROC-AUC | Precision no corte 0,5 | Recall no corte 0,5 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,2692 | 0,8438 | 22,62% | 73,60% |
| Random Forest | 0,2742 | 0,8450 | 22,17% | 76,02% |
| Gradient Boosting | 0,2819 | 0,8476 | 21,87% | 77,18% |
| HistGradientBoosting | 0,2875 | 0,8517 | 22,55% | 76,71% |

O HGB apresentou o maior valor pontual de Average Precision. Não foram estimados intervalos de confiança nem testada a significância da diferença entre modelos. As métricas no corte 0,5 documentam a comparação original; esse corte não foi utilizado na avaliação final.

## 6. Política de threshold e teste final

A regra predefinida foi maximizar recall entre os thresholds com precision ≥ 0,50 na validação. Se nenhum satisfizesse a restrição, seria adotado o máximo F1. O fallback não foi necessário.

O threshold selecionado foi `0.8489187193708023`, com política `max_recall_precision_ge_0_50`. Apenas depois da seleção do vencedor e do threshold, o pré-processador vencedor transformou o teste, e o modelo foi avaliado em 2026. Nenhum dos outros candidatos foi comparado no teste final nesta execução.

| Métrica do HGB | Validação 2025 | Teste 2026 disponível |
|---|---:|---:|
| Average Precision | 0,287478 | 0,253001 |
| ROC-AUC | 0,851720 | 0,859102 |
| Precision | 50,00% | 45,26% |
| Recall | 3,32% | 3,44% |
| F1 | 0,062314 | 0,063961 |
| Threshold | 0,848919 | 0,848919 |

Matrizes de confusão no threshold escolhido:

| Resultado | Validação | Teste |
|---|---:|---:|
| Cura sem alerta (verdadeiro negativo) | 281.574 | 145.045 |
| Cura com alerta (falso positivo) | 765 | 364 |
| Óbito sem alerta (falso negativo) | 22.258 | 8.446 |
| Óbito com alerta (verdadeiro positivo) | 765 | 301 |

No teste, foram emitidos 665 alertas, dos quais 301 corresponderam a óbitos. Dos 8.747 óbitos, 8.446 não foram identificados. A precisão mínima era uma restrição de seleção em 2025, não uma garantia para dados futuros.

## 7. Interpretação e limitações

A Average Precision de teste (0,2530) ficou acima da prevalência de óbitos (aproximadamente 0,0567), referência aproximada de uma ordenação aleatória. Isso indica capacidade de ordenação de risco, sem demonstrar, por si só, adequação do ponto de corte ou benefício prático.

A política de precisão mínima impôs um compromisso acentuado: no HGB em 2025, passar do corte 0,5 para o corte selecionado elevou precision de 22,55% para 50%, mas reduziu recall de 76,71% para 3,32%. O modelo e a política de decisão devem, portanto, ser avaliados separadamente.

A Average Precision diminuiu de 2025 para 2026, enquanto a ROC-AUC aumentou ligeiramente. A prevalência também mudou. Esses resultados isolados não permitem atribuir a diferença a uma causa específica nem concluir que houve deterioração uniforme do modelo.

Limitações desta V1:

- Baixa sensibilidade no corte final, com aproximadamente 96,56% dos óbitos do teste sem alerta.
- Ausência de avaliação de calibração; o score e o threshold não devem ser interpretados como risco absoluto comprovado de óbito.
- Ausência, nos artefatos examinados, de intervalos de confiança e análises por subgrupos, região ou ano de treino.
- Ausência de auditoria documentada de duplicidade entre pacientes e da disponibilidade temporal real das features.
- Recorte incompleto de 2026 e ausência de registro da data exata de atualização da fonte; completude dos desfechos exige avaliação própria.
- Duas features previstas ausentes e representação ordinal das categóricas no HGB.
- Ausência de medições de pico de RAM e duração por etapa: a conclusão da execução comprova viabilidade nesta execução, mas não quantifica o ganho de desempenho.

## 8. Artefatos e reprodutibilidade

Diretório da execução: `artifacts/ml-admission/20260906-104505`.

- `best_model.joblib`: payload com `pipeline`, `threshold`, `features` e `best_model`. A pipeline reúne o pré-processador ordinal ajustado e o HGB treinado, permitindo `predict_proba` sobre as features originais.
- `metrics.json` e `metrics.csv`: métricas finais de validação e teste.
- `validation_comparison.csv`: comparação dos quatro candidatos em 2025.
- `confusion_matrix_validation.csv` e `confusion_matrix_test.csv`: matrizes de confusão.
- `run_metadata.json`: anos, tamanhos das partições, features, seed e versões.

Ambiente registrado: Python 3.11.9, pandas 2.3.2 e scikit-learn 1.7.1. O identificador de commit acima foi consultado no repositório durante a elaboração; não está incorporado ao `run_metadata.json`. Para reprodução exata, também é necessário preservar a mesma versão dos Parquets, cujos hashes não constam desses metadados.

Comando utilizado, registrado para rastreabilidade e não como indicação de repetir o teste final:

```powershell
$parquets = "data/parquet/srag/ano=*/srag.parquet"
python -u scripts/train_ml_admission.py --parquet-glob $parquets
```

Na verificação de implementação registrada antes da execução nacional, passaram 65 testes da suíte ML específica e 162 da suíte completa. Houve um warning de depreciação relacionado a FastAPI/Starlette. Esses testes verificam contratos de software, não a qualidade preditiva ou a validade clínica do modelo.

## 9. Conclusão e encaminhamento

A V1 estabeleceu uma pipeline funcional para a base nacional disponível e selecionou o HistGradientBoosting por Average Precision na validação temporal. O teste final mostrou capacidade de discriminação, porém sensibilidade muito baixa no ponto de corte definido pela política de precisão mínima. O resultado deve ser apresentado como uma referência experimental inicial, com limitações explícitas, e não como um sistema validado para uso assistencial.

O encerramento da V1 requer preservar os artefatos e documentar os resultados. Uma eventual V2 deve definir previamente o objetivo operacional dos alertas e o protocolo de avaliação, incluindo calibração, qualidade dos dados e subgrupos. Os resultados de 2026 já foram examinados: não devem orientar ajustes mantendo esse mesmo recorte como se fosse um teste intocado. Uma nova avaliação final independente exigirá dados ainda não utilizados nas decisões de desenvolvimento.

## Fontes locais

Resultados: `artifacts/ml-admission/20260906-104505/metrics.json`, `validation_comparison.csv` e `run_metadata.json`. Metodologia: módulos `dataset.py`, `features.py`, `split.py`, `preprocessing.py`, `models.py`, `training.py`, `threshold.py` e `artifacts.py` em `src/srag_api/ml/`. Histórico de testes: saídas registradas na sessão de implementação da otimização. Nenhum treinamento ou nova avaliação de 2026 foi realizado para produzir este relatório.
