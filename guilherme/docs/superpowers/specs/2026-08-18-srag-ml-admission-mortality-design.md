# SRAG ML V1 — Predição de Óbito na Admissão

**Data:** 2026-08-18  
**Projeto:** GT09PetSaudeDigital/grupo-srag  
**Escopo:** `guilherme/`  
**Status:** Design aprovado para revisão da especificação

## 1. Objetivo

Construir um subsistema de Machine Learning para prever risco de óbito por SRAG em pacientes hospitalizados, usando apenas informações disponíveis até a notificação/admissão.

A V1 será nacional, cobrirá todos os casos de SRAG entre 2019 e 2026 e não será restrita a COVID-19.

## 2. Pergunta de pesquisa

> Dado um paciente hospitalizado por SRAG, qual a probabilidade de óbito por SRAG usando somente dados disponíveis até a notificação/admissão?

A V1 não representa prognóstico durante a internação. Variáveis geradas depois da admissão não podem entrar como preditores.

## 3. População

Incluir registros de SRAG hospitalizados no período de 2019 a 2026 que possuam desfecho elegível para construção do alvo.

A análise é nacional.

### 3.1 Desfechos elegíveis

- `CURA` → alvo `0`
- `OBITO_SRAG` → alvo `1`

### 3.2 Registros excluídos do treino

- `OBITO_OUTRAS_CAUSAS`
- `AUSENTE`
- `IGNORADO`
- demais valores de desfecho não classificados como `CURA` ou `OBITO_SRAG`

`OBITO_OUTRAS_CAUSAS` não será tratado como sobrevivente.

## 4. Regra temporal de disponibilidade

Uma variável só pode integrar `X` se puder ser conhecida até o momento da notificação/admissão.

Informações posteriores devem ser excluídas da V1, mesmo que aumentem a performance preditiva.

## 5. Features candidatas

### 5.1 Demográficas

- `CS_SEXO`
- `NU_IDADE_N`
- `CS_GESTANT`

### 5.2 Sintomas e sinais clínicos

- `FEBRE`
- `TOSSE`
- `GARGANTA`
- `DISPNEIA`
- `DESC_RESP`
- `SATURACAO`
- `DIARREIA`
- `VOMITO`
- `DOR_ABD`
- `FADIGA`
- `PERD_OLFT`
- `PERD_PALA`
- `OUTRO_SIN`, sob avaliação de qualidade

### 5.3 Comorbidades e fatores de risco

Incluir as usadas no trabalho de Willian Penteado e ampliar para comorbidades disponíveis na admissão, entre elas:

- `CARDIOPATI`
- `DIABETES`
- `PNEUMOPATI`
- `RENAL`
- `HEPATICA`
- `IMUNODEPRE`
- `OBESIDADE`
- `OUT_MORBI`
- `FATOR_RISC`, sob avaliação de redundância

A lista final dependerá da disponibilidade real por ano e da consistência de schema entre 2019 e 2026.

## 6. Features geográficas

Incluir:

- UF
- região do Brasil derivada da UF, quando útil

Não incluir município na V1, para evitar alta cardinalidade e reduzir risco de o modelo memorizar padrões locais muito específicos.

## 7. Features temporais válidas

Pode ser usada uma variável derivada de datas conhecidas na admissão/notificação, por exemplo:

- `SINT_ATE_NOTIF = DT_NOTIFIC - DT_SIN_PRI`

Datas/derivadas que dependam da evolução posterior são proibidas.

## 8. Variáveis proibidas por leakage na V1

A lista de bloqueio deve incluir, no mínimo:

- `EVOLUCAO`
- `DESFECHO_NORMALIZADO`
- `OBITO_SRAG`
- `DT_EVOLUCA`
- `UTI`
- `SUPORT_VEN`
- `QTD_DIAS`
- `DIAS_INTERNA`
- `PCR_EVOLUCAO`

Também devem ser bloqueadas outras variáveis cujo valor só seja definido após a admissão, mesmo que não apareçam nesta lista inicial.

A implementação deverá ter testes automáticos para garantir que nenhuma variável proibida entre em `ADMISSION_FEATURES`.

## 9. Laboratório e etiologia

A V1 principal será de admissão e não dependerá de resultados laboratoriais posteriores.

A arquitetura deve permitir experimentos futuros separados:

- **Modelo A — Admissão:** somente informações precoces.
- **Modelo B — Admissão + laboratório:** Modelo A + resultados laboratoriais/`ETIOLOGIA_DETALHADA`, quando disponíveis.
- **Modelo C — Prognóstico durante internação:** experimento futuro separado, podendo incluir variáveis como UTI e suporte ventilatório.

Esses modelos representam perguntas científicas diferentes e não devem ser misturados.

## 10. Compatibilidade 2019–2026

O pipeline deve suportar diferenças de schema entre os anos.

Regras:

- coluna ausente em determinado ano não deve ser inventada como resultado negativo;
- ausência de campo, valor ignorado e valor explicitamente negativo devem permanecer conceitualmente distintos;
- features só entram no conjunto final quando houver regra explícita de tratamento;
- variáveis específicas de períodos, como vacinação contra COVID-19, não devem ser tratadas como universalmente disponíveis desde 2019.

## 11. Divisão temporal

A avaliação principal deve ser temporal, não puramente aleatória.

Estratégia:

- anos anteriores → treino;
- período recente anterior a 2026 → validação;
- 2026 → teste final fora do tempo.

A fronteira exata de treino/validação será definida na implementação após inspeção da distribuição real de registros e desfechos por ano.

O conjunto de teste de 2026 não pode participar de imputação, encoding, scaler, seleção de features, balanceamento ou ajuste de hiperparâmetros.

## 12. Pré-processamento sem leakage

Ordem obrigatória:

1. montar população elegível;
2. separar treino/validação/teste temporalmente;
3. ajustar imputação somente no treino;
4. ajustar encoding somente no treino;
5. ajustar scaler somente no treino, quando aplicável;
6. aplicar balanceamento somente no treino;
7. treinar o modelo;
8. avaliar validação e teste sem refazer `fit`.

SMOTE, se utilizado, deve atuar somente sobre a partição de treino.

## 13. Arquitetura do código

Novo pacote:

```text
guilherme/src/srag_api/ml/
├── __init__.py
├── features.py
├── target.py
├── dataset.py
├── split.py
└── preprocessing.py
```

### `features.py`

Responsável por:

- definir `ADMISSION_FEATURES`;
- agrupar features por domínio;
- definir `LEAKAGE_FEATURES`;
- validar interseção vazia entre features permitidas e proibidas.

### `target.py`

Responsável por:

- converter `DESFECHO_NORMALIZADO` em alvo binário;
- `CURA -> 0`;
- `OBITO_SRAG -> 1`;
- marcar demais desfechos como inelegíveis.

### `dataset.py`

Responsável por:

- receber dados já normalizados pelo pipeline SRAG;
- selecionar registros com alvo elegível;
- selecionar features disponíveis;
- devolver `X`, `y` e metadados mínimos;
- garantir que nenhuma feature bloqueada esteja presente.

### `split.py`

Responsável por:

- divisão temporal;
- impedir sobreposição temporal;
- reservar 2026 para teste fora do tempo.

### `preprocessing.py`

Responsável por:

- imputação;
- encoding;
- scaler quando necessário;
- preparação para balanceamento;
- impedir `fit` fora do treino.

## 14. Testes obrigatórios

A suíte deve validar pelo menos:

1. `CURA` vira `0`;
2. `OBITO_SRAG` vira `1`;
3. `OBITO_OUTRAS_CAUSAS` é excluído;
4. ausente/ignorado é excluído;
5. nenhuma `LEAKAGE_FEATURE` aparece em `ADMISSION_FEATURES`;
6. dataset não inclui variável proibida;
7. schema sem uma feature opcional não inventa coluna negativa;
8. divisão temporal não mistura anos;
9. 2026 permanece fora do treino;
10. transformadores são ajustados somente no treino;
11. balanceamento nunca é aplicado em validação/teste.

## 15. Relação com o TCC de Willian Penteado

O TCC será usado como benchmark metodológico e fonte de features candidatas, não como pipeline a ser reproduzido integralmente.

Melhorias deliberadas da V1:

- todos os casos de SRAG, não apenas COVID-19;
- Brasil inteiro, não apenas Paraná;
- 2019–2026;
- expansão das comorbidades;
- bloqueio explícito de leakage;
- split temporal;
- imputação/encoding/scaler ajustados somente no treino;
- balanceamento somente no treino;
- preservação das diferenças de schema entre os anos.

## 16. Critérios de aceite do subsistema V1

A etapa de preparação de dados estará pronta quando:

- o módulo `srag_api.ml` existir e estiver coberto por testes;
- houver uma lista auditável de features permitidas e proibidas;
- o alvo binário for gerado corretamente;
- o dataset de admissão não contiver leakage conhecido;
- a divisão temporal estiver implementada;
- o pré-processamento respeitar isolamento entre treino/validação/teste;
- a suíte completa do projeto continuar passando.

O treinamento e comparação de modelos serão uma etapa posterior, após a validação do dataset V1.
