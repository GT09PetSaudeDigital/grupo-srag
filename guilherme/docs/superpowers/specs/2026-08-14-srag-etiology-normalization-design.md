# SRAG — Design de Normalização de Classificação Final e Etiologia

**Data:** 2026-08-14  
**Projeto:** `GT09PetSaudeDigital/grupo-srag`  
**Escopo:** `guilherme/`  
**Status:** Aprovado pelo usuário

## 1. Objetivo

Corrigir e tornar metodologicamente explícita a normalização etiológica do pipeline SRAG/SIVEP-Gripe, separando:

1. a **classificação final oficial do caso** (`CLASSI_FIN`); e
2. a **etiologia detalhada identificada** por campos laboratoriais.

O desenho deve funcionar de forma consistente sobre dados de 2019 a 2026, sem inventar equivalências entre anos e sem interpretar a ausência de uma coluna como resultado laboratorial negativo.

## 2. Motivação

A implementação atual produz uma única coluna `ETIOLOGIA_NORMALIZADA` combinando `CLASSI_FIN` e flags laboratoriais.

Esse comportamento perde uma distinção epidemiológica importante:

- `CLASSI_FIN` representa a classificação final oficial do caso;
- os campos laboratoriais representam evidências sobre o agente identificado.

Por exemplo, um caso pode estar oficialmente classificado como `SRAG por outro vírus respiratório` e ter VSR identificado em RT-PCR. Essas duas informações devem permanecer separadas.

## 3. Princípio de compatibilidade 2019–2026

O pipeline terá **um único modelo analítico**, com diferenças históricas tratadas como disponibilidade de dados e não como pipelines separados por ano.

### 3.1 Regras gerais

- Preservar sempre as colunas originais.
- Uma coluna inexistente em determinado ano não equivale a resultado negativo.
- Valores ausentes, ignorados e resultados negativos devem permanecer distinguíveis sempre que o dado de origem permitir.
- Não criar uma regra dependente do ano sem evidência documental ou empírica no schema real.
- Validar o schema real de 2026 antes de interpretação científica, porque o portal atualmente disponibiliza um dicionário rotulado 2019–2025.

### 3.2 Matriz de compatibilidade

| Campo/regra | 2019 | 2020–2025 | 2026 | Decisão |
|---|---|---|---|---|
| `TP_IDADE` + `NU_IDADE_N` | usar | usar | usar | regra única |
| `SEM_PRI` | preservar | preservar | preservar | não reconstruir quando informado |
| `UTI` | normalizar | normalizar | normalizar | SIM/NAO/IGNORADO/AUSENTE |
| `EVOLUCAO` | normalizar | normalizar | normalizar | manter óbito SRAG separado de outras causas |
| `CLASSI_FIN` | preservar e normalizar quando aplicável | normalizar | validar no schema real | classificação oficial separada |
| PCR/antígenos SARS-CoV-2 | não presumir existência | usar quando presentes | usar quando presentes | ausência de coluna não é negativo |
| Influenza A/B | usar quando presentes | usar | usar | etiologia detalhada |
| VSR/outros vírus | usar quando presentes | usar | usar | etiologia detalhada |
| Comorbidades | somente campos existentes | idem | idem | schema variável permitido |

## 4. Classificação final normalizada

Criar a coluna:

`CLASSIFICACAO_FINAL_NORMALIZADA`

Mapeamento inicial baseado na ficha oficial vigente:

| `CLASSI_FIN` | Valor normalizado |
|---|---|
| `1` | `INFLUENZA` |
| `2` | `OUTRO_VIRUS_RESPIRATORIO` |
| `3` | `OUTRO_AGENTE_ETIOLOGICO` |
| `4` | `NAO_ESPECIFICADO` |
| `5` | `COVID-19` |
| ausente | `AUSENTE` |
| código inesperado | `OUTRO` |

A normalização não deve inventar `IGNORADO` para `CLASSI_FIN` sem confirmação no dicionário correspondente.

## 5. Etiologia detalhada

Criar a coluna:

`ETIOLOGIA_DETALHADA`

A etiologia detalhada deriva somente de campos laboratoriais disponíveis.

Categorias iniciais:

- `SARS-CoV-2`
- `Influenza A`
- `Influenza B`
- `VSR`
- `Parainfluenza 1`
- `Parainfluenza 2`
- `Parainfluenza 3`
- `Parainfluenza 4`
- `Adenovirus`
- `Metapneumovirus`
- `Bocavirus`
- `Rinovirus`
- `Outro virus respiratorio`
- `NAO_IDENTIFICADA`

Subtipos/linhagens podem ser preservados em colunas adicionais no futuro, sem ampliar a V1 desta alteração.

## 6. Co-detecção

A ficha do SIVEP-Gripe contempla co-detecção.

Portanto, a implementação não deve assumir que todo registro tem exatamente um agente.

Para esta alteração:

- a coluna `ETIOLOGIA_DETALHADA` pode representar uma categoria principal determinística para manter compatibilidade com a API atual;
- a implementação deve preservar os campos laboratoriais originais para permitir estudo posterior de co-detecção;
- não remover nem sobrescrever flags positivas adicionais.

Uma futura evolução poderá expor uma estrutura multirrótulo sem quebrar a camada bruta/analítica.

## 7. Conflitos entre classificação final e laboratório

Quando houver aparente conflito entre `CLASSI_FIN` e resultado laboratorial:

- **não corrigir silenciosamente a classificação final**;
- manter `CLASSIFICACAO_FINAL_NORMALIZADA` derivada de `CLASSI_FIN`;
- manter `ETIOLOGIA_DETALHADA` derivada do laboratório;
- preservar o registro para auditoria de qualidade;
- futuramente contabilizar inconsistências no relatório de qualidade.

Exemplo:

```text
CLASSI_FIN = 4
PCR_SARS2 = 1

CLASSIFICACAO_FINAL_NORMALIZADA = NAO_ESPECIFICADO
ETIOLOGIA_DETALHADA = SARS-CoV-2
```

O pipeline não transforma automaticamente a classificação final em COVID-19.

## 8. Arquitetura de código

### `data/etiology.py`

Responsabilidades:

- normalizar a classificação final;
- detectar etiologia detalhada a partir de campos disponíveis;
- adicionar as duas novas colunas analíticas;
- nunca modificar as colunas fonte.

Interfaces planejadas:

```python
normalize_final_classification(value: object) -> str
normalize_detailed_etiology(row: pd.Series) -> str
add_etiology_columns(df: pd.DataFrame) -> pd.DataFrame
```

### `data/ingest.py`

Substituir a chamada que gera apenas `ETIOLOGIA_NORMALIZADA` pela nova função de enriquecimento.

### `data/repository.py`

Adaptar filtros e agregações sem misturar os dois conceitos.

A API deve manter compatibilidade de comportamento durante a transição, mas o significado de `/etiologia` deve ser explícito antes de qualquer uso científico.

## 9. Compatibilidade da API

Na primeira implementação, evitar criar novos endpoints.

Opções de transição:

- manter `/api/v1/etiologia` agregando por `ETIOLOGIA_DETALHADA`; e
- documentar claramente que o endpoint representa agente detalhado, não classificação final.

A classificação final poderá ser exposta depois por um filtro/endpoint específico somente se houver necessidade real.

## 10. Compatibilidade com Machine Learning

As features devem distinguir:

1. resultado laboratorial negativo;
2. resultado ausente;
3. resultado ignorado;
4. coluna inexistente naquele schema/ano.

Não preencher automaticamente colunas inexistentes historicamente com zero.

Durante modelagem futura:

- preprocessing deve ser ajustado somente no treino;
- validação temporal deve ser preferida;
- diferenças históricas de disponibilidade devem ser tratadas explicitamente;
- `CLASSIFICACAO_FINAL_NORMALIZADA` e `ETIOLOGIA_DETALHADA` não devem ser usadas como preditores se representarem informação disponível somente após o desfecho/encerramento do caso e houver risco de leakage.

## 11. Testes obrigatórios

Adicionar testes para:

1. `CLASSI_FIN = 1` → `INFLUENZA`;
2. `CLASSI_FIN = 2` → `OUTRO_VIRUS_RESPIRATORIO`;
3. `CLASSI_FIN = 3` → `OUTRO_AGENTE_ETIOLOGICO`;
4. `CLASSI_FIN = 4` → `NAO_ESPECIFICADO`;
5. `CLASSI_FIN = 5` → `COVID-19`;
6. `CLASSI_FIN` ausente;
7. código inesperado;
8. Influenza A positiva;
9. Influenza B positiva;
10. VSR positivo;
11. SARS-CoV-2 positivo;
12. demais vírus configurados;
13. campo PCR inexistente;
14. campo PCR ausente;
15. conflito entre `CLASSI_FIN` e PCR;
16. preservação das colunas fonte;
17. fixture compatível com 2019 sem exigir campos exclusivos da era covid.

## 12. Critérios de aceitação

A alteração estará pronta quando:

- todos os testes existentes continuarem passando;
- novos testes da classificação final e etiologia detalhada passarem;
- `CLASSI_FIN` original permanecer intacto;
- PCR/antígenos originais permanecerem intactos;
- as duas novas colunas forem gravadas no Parquet;
- coluna inexistente não for convertida em negativo;
- a API continuar funcionando;
- README documentar a nova semântica;
- os dados reais ainda não tiverem sido reprocessados antes da validação do schema de 2026.

## 13. Fora de escopo

Não faz parte desta alteração:

- treinamento de ML;
- seleção definitiva de features;
- modelagem multirrótulo de co-detecção;
- reconstrução de semana epidemiológica;
- alterações em dados RAW;
- novos dashboards;
- MCP;
- migração de arquitetura da API.

## 14. Sequência de implementação

1. adicionar testes de regressão para `CLASSI_FIN`;
2. implementar normalização da classificação final;
3. adicionar testes da etiologia detalhada;
4. implementar etiologia detalhada;
5. integrar ao `transform_srag_dataframe`;
6. adaptar repository/service/API apenas onde necessário;
7. rodar suíte completa;
8. validar schema real de 2019 e 2026 antes de processar todas as bases;
9. atualizar README;
10. abrir PR separado para revisão.

## 15. Referências oficiais usadas no desenho

- Portal de Dados Abertos do SUS — Banco de dados da SRAG 2019 a 2026.
- Ministério da Saúde — Ficha de Registro Individual do SIVEP-Gripe disponibilizada no conjunto SRAG.
- Dicionário de Dados SIVEP-Gripe identificado no portal como 2019 a 2025.
