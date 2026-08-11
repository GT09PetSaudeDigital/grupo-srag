# SRAG Epidemiological Data API — Design Specification

**Data:** 2026-08-11  
**Projeto:** GT09PetSaudeDigital/grupo-srag  
**Escopo:** `guilherme/`  
**Status:** Aprovado para revisão do usuário

## 1. Objetivo

Evoluir a análise atual de SRAG em `guilherme/analise_srag_pr.py` para uma arquitetura modular, reproduzível e reutilizável, com foco inicial em consultas epidemiológicas agregadas no Brasil, estados e municípios.

A solução deve:

- processar dados SIVEP-Gripe de 2019 a 2026;
- preservar os dados brutos;
- gerar dados analíticos em Parquet;
- consultar os dados com DuckDB;
- oferecer uma API REST pública somente leitura com FastAPI;
- expor métricas epidemiológicas com denominadores explícitos;
- suportar filtros geográficos, temporais, demográficos e etiológicos;
- preparar a arquitetura para futura integração com MCP e Machine Learning;
- manter o escopo de implementação restrito à pasta `guilherme/`.

## 2. Fora de escopo da V1

A V1 não inclui:

- autenticação;
- Redis;
- microserviços;
- registros individuais expostos pela API;
- execução de SQL arbitrário por usuários;
- MCP Server funcional;
- chatbot;
- dashboards;
- treinamento de modelos de ML dentro do pipeline de ingestão;
- infraestrutura de produção avançada.

A arquitetura deve, porém, permitir que MCP e ML sejam adicionados futuramente sem reescrever a camada de dados.

## 3. Arquitetura

A abordagem escolhida é uma arquitetura modular em camadas.

```text
SIVEP-Gripe
     ↓
CSV 2019–2026
     ↓
Data Ingestion
     ↓
Validation
     ↓
Normalization
     ↓
Quality Reports
     ↓
Parquet
     ↓
DuckDB
     ↓
SragRepository
     ↓
Services
 ┌───┴────┐
 ↓        ↓
FastAPI   MCP Server (futuro)
          ↓
        AI Agent
```

Machine Learning será um consumidor independente da mesma base preparada:

```text
Parquet / DuckDB
      ├── FastAPI
      ├── MCP
      └── ML
```

## 4. Estrutura proposta

```text
guilherme/
├── src/
│   └── srag_api/
│       ├── api/
│       │   ├── app.py
│       │   └── routes/
│       │       ├── epidemiologia.py
│       │       ├── locais.py
│       │       ├── etiologia.py
│       │       └── comparacoes.py
│       ├── data/
│       │   ├── ingest.py
│       │   ├── clean.py
│       │   ├── schema.py
│       │   └── repository.py
│       ├── services/
│       │   ├── epidemiology.py
│       │   ├── etiology.py
│       │   └── comparison.py
│       └── config.py
├── data/
│   ├── raw/
│   ├── parquet/
│   └── quality/
├── scripts/
│   ├── ingest_year.py
│   └── ingest_all.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   └── superpowers/
│       └── specs/
├── analise_srag_pr.py
├── pyproject.toml
└── README.md
```

O arquivo `analise_srag_pr.py` atual não deve ser removido na primeira etapa. Ele servirá como referência histórica até que as responsabilidades sejam migradas para módulos separados.

## 5. Dados

### 5.1 Período

O pipeline suportará dados de:

- 2019
- 2020
- 2021
- 2022
- 2023
- 2024
- 2025
- 2026

### 5.2 Armazenamento RAW

Arquivos originais devem ser preservados sem alteração:

```text
data/raw/
├── 2019/
├── 2020/
├── 2021/
├── 2022/
├── 2023/
├── 2024/
├── 2025/
└── 2026/
```

### 5.3 Parquet

Os dados processados serão gravados de forma particionada por ano:

```text
data/parquet/srag/
├── ano=2019/
├── ano=2020/
├── ano=2021/
├── ano=2022/
├── ano=2023/
├── ano=2024/
├── ano=2025/
└── ano=2026/
```

DuckDB será utilizado para consultar esses arquivos diretamente.

## 6. Pipeline

O pipeline será dividido em cinco etapas.

### 6.1 Raw

Preserva o CSV original.

### 6.2 Validação

Antes da transformação, o pipeline deve validar:

- existência do arquivo;
- leitura e encoding;
- presença das colunas essenciais;
- compatibilidade mínima do schema;
- quantidade de registros;
- possibilidade de interpretar campos principais.

Problemas de qualidade não devem ser ocultados silenciosamente.

### 6.3 Normalização

A normalização deve incluir:

- nomes de colunas;
- datas;
- sexo;
- raça/cor quando aplicável;
- município e UF;
- evolução;
- UTI;
- comorbidades;
- etiologia;
- idade.

### 6.4 Normalização da idade

`NU_IDADE_N` não deve ser tratado isoladamente como idade em anos.

O pipeline deve considerar `TP_IDADE` e `NU_IDADE_N` e produzir:

- `IDADE_ANOS`;
- `FAIXA_ETARIA`.

Faixas iniciais:

- `<1`
- `1-4`
- `5-11`
- `12-17`
- `18-29`
- `30-44`
- `45-59`
- `60-74`
- `75+`

A definição das faixas deve ficar centralizada em configuração.

### 6.5 Valores ausentes e ignorados

Sempre que possível, o pipeline deve distinguir:

- SIM;
- NÃO;
- IGNORADO;
- AUSENTE.

Valores ignorados não devem ser automaticamente confundidos com ausência de informação.

## 7. Enriquecimento

Variáveis derivadas planejadas:

- `IDADE_ANOS`;
- `FAIXA_ETARIA`;
- `ANO`;
- `MES`;
- `SEMANA_EPIDEMIOLOGICA`;
- `UF`;
- `CODIGO_MUNICIPIO`;
- `MUNICIPIO`;
- `ETIOLOGIA_NORMALIZADA`;
- `DESFECHO_NORMALIZADO`;
- `FOI_UTI`;
- `OBITO_SRAG`;
- `TOTAL_COMORBIDADES`.

A V1 deve evitar variáveis derivadas sem uso claro.

## 8. Etiologia

A API deve incluir identificação etiológica já na V1.

Categorias amigáveis previstas:

- COVID-19;
- Influenza A;
- Influenza B;
- VSR;
- Outros vírus respiratórios;
- Outro agente;
- Não identificado;
- Ignorado.

Os códigos originais devem permanecer disponíveis na camada analítica para auditoria e reprodutibilidade.

## 9. Qualidade dos dados

Cada ano processado deverá gerar um relatório de qualidade:

```text
data/quality/
├── quality_2019.json
├── quality_2020.json
├── quality_2021.json
├── quality_2022.json
├── quality_2023.json
├── quality_2024.json
├── quality_2025.json
└── quality_2026.json
```

Indicadores mínimos:

- registros recebidos;
- registros processados;
- duplicados;
- datas inválidas;
- idade ausente;
- sexo ausente;
- município ausente;
- evolução ausente;
- UTI ausente;
- etiologia não identificada.

O relatório deve permitir avaliar completude e consistência da base.

## 10. Ingestão incremental

O processamento será incremental por ano.

Exemplos de interface:

```bash
python scripts/ingest_year.py --year 2025
python scripts/ingest_year.py --year 2026 --force
python scripts/ingest_all.py
```

Se um ano já estiver processado e não houver solicitação de reprocessamento, ele deverá ser ignorado.

Essa estratégia é especialmente importante para o banco vivo do ano corrente.

## 11. Repository

`SragRepository` será a única camada que conhecerá detalhes de DuckDB e SQL.

Operações previstas:

- `get_total_cases(filters)`
- `get_deaths(filters)`
- `get_icu_cases(filters)`
- `get_age_distribution(filters)`
- `get_comorbidity_distribution(filters)`
- `get_etiology_distribution(filters)`
- `get_time_series(filters)`
- `get_ranking(filters)`

Nenhum endpoint deverá construir SQL diretamente.

Nenhuma futura ferramenta MCP deverá executar SQL arbitrário.

## 12. Services

Os services serão responsáveis por regras epidemiológicas e composição de respostas.

Serviços previstos:

- `EpidemiologyService`;
- `EtiologyService`;
- `ComparisonService`.

Responsabilidades:

- aplicar filtros;
- validar combinações;
- calcular indicadores;
- controlar denominadores;
- ordenar resultados;
- comparar localidades;
- padronizar respostas;
- tratar dados ignorados.

## 13. Geografia

A API suportará:

- Brasil;
- Estado;
- Município.

Campos internos mínimos:

- `UF`;
- `CODIGO_MUNICIPIO`;
- `MUNICIPIO`.

Municípios devem ser identificados internamente por código sempre que possível.

Consultas por município devem exigir UF ou código de município para evitar ambiguidade.

## 14. Filtros comuns

Filtros previstos:

- `ano_inicio`;
- `ano_fim`;
- `uf`;
- `municipio`;
- `codigo_municipio`;
- `sexo`;
- `faixa_etaria`;
- `etiologia`.

Os mesmos nomes e semântica devem ser utilizados entre endpoints sempre que aplicável.

## 15. API REST

A API será:

- pública;
- somente leitura;
- sem autenticação na V1;
- documentada automaticamente via Swagger/OpenAPI.

Endpoints iniciais:

```text
GET /health

GET /api/v1/casos
GET /api/v1/obitos
GET /api/v1/uti
GET /api/v1/comorbidades
GET /api/v1/faixas-etarias
GET /api/v1/etiologia
GET /api/v1/serie-temporal
GET /api/v1/ranking
GET /api/v1/comparar
```

Endpoints específicos por cidade, agente ou ano não devem ser criados quando filtros genéricos resolverem o mesmo problema.

## 16. Métricas epidemiológicas

A API deve evitar usar a palavra "taxa" sem definição clara do denominador.

### 16.1 Letalidade por SRAG

```text
óbitos por SRAG
-------------------------------
casos com evolução conhecida
```

Nome da métrica:

`letalidade_srag`

### 16.2 Proporção de UTI

```text
casos com internação em UTI
--------------------------------
casos com informação de UTI conhecida
```

Nome da métrica:

`proporcao_uti`

### 16.3 Transparência

Respostas de indicadores devem incluir, quando aplicável:

- valor;
- unidade;
- numerador;
- denominador;
- quantidade ignorada.

Exemplo:

```json
{
  "metrica": "letalidade_srag",
  "valor": 7.08,
  "unidade": "%",
  "numerador": 850,
  "denominador": 12005,
  "ignorados": 535
}
```

## 17. Séries temporais

Endpoint:

```text
GET /api/v1/serie-temporal
```

Frequências previstas:

- semana epidemiológica;
- mês.

O pipeline deve preservar semana epidemiológica de forma consistente.

## 18. Ranking

O endpoint de ranking deve permitir comparar:

- estados;
- municípios.

Métricas iniciais possíveis:

- casos;
- óbitos;
- letalidade;
- UTI;
- proporção de UTI.

## 19. Comparações

Endpoint:

```text
GET /api/v1/comparar
```

Deve permitir comparar dois ou mais locais usando a mesma métrica e filtros.

Exemplo conceitual:

```text
locais=4106902,4113700,4104808
metrica=letalidade_srag
ano=2025
etiologia=influenza
```

## 20. Tratamento de erros

A API deve usar erros estruturados e previsíveis.

Casos principais:

- `400`: filtro inválido;
- `404`: localidade não encontrada;
- `422`: combinação de parâmetros inválida;
- `500`: erro interno;
- `503`: dados necessários ainda não processados.

Mensagens genéricas sem contexto devem ser evitadas.

## 21. Cache

Redis não fará parte da V1.

DuckDB + Parquet será a estratégia inicial.

Cache adicional só será introduzido após evidência de gargalo.

## 22. Logs

A V1 usará logging estruturado simples.

Exemplos:

```text
INFO ingest_year ano=2025 registros=253421
INFO parquet_written ano=2025
INFO api_query endpoint=/casos uf=PR ano=2025
WARNING missing_field campo=EVOLUCAO quantidade=1234
ERROR ingestion_failed ano=2026
```

Prometheus e Grafana não fazem parte da V1.

## 23. Health check

Endpoint:

```text
GET /health
```

Resposta mínima:

```json
{
  "status": "ok",
  "data_available": {
    "from": 2019,
    "to": 2026
  }
}
```

## 24. Testes

Estrutura:

```text
tests/
├── unit/
│   ├── test_clean.py
│   ├── test_age_normalization.py
│   ├── test_etiology.py
│   └── test_services.py
├── integration/
│   ├── test_repository.py
│   └── test_api.py
└── fixtures/
    └── sample_srag.csv
```

### 24.1 Testes unitários

Devem cobrir pelo menos:

- idade em dias;
- idade em meses;
- idade em anos;
- faixas etárias;
- valores ignorados;
- etiologia;
- letalidade;
- proporção de UTI;
- filtros.

### 24.2 Testes de integração

Fluxo mínimo:

```text
fixture CSV
   ↓
ingestão
   ↓
Parquet
   ↓
DuckDB
   ↓
Repository
   ↓
Service
   ↓
FastAPI
```

## 25. Machine Learning

ML não será executado como parte da ingestão.

Estrutura futura:

```text
src/srag_api/ml/
├── features.py
├── training.py
├── evaluation.py
└── explainability.py
```

A evolução do ML deve corrigir problemas metodológicos comuns, incluindo:

- preprocessing ajustado somente no treino;
- prevenção de data leakage;
- validação temporal;
- tratamento de desbalanceamento;
- métricas adequadas;
- importância de features;
- explicabilidade;
- comparação entre períodos.

## 26. MCP

O MCP Server será implementado após a API.

Ele não acessará DuckDB diretamente.

Arquitetura:

```text
FastAPI ──────┐
              ↓
       Services
              ↑
MCP Server ───┘
```

Tools futuras:

- `consultar_casos`;
- `consultar_letalidade`;
- `consultar_uti`;
- `consultar_etiologias`;
- `consultar_serie_temporal`;
- `comparar_locais`.

Isso permite que um agente de IA consulte os dados sem gerar SQL arbitrário.

## 27. README

O README de `guilherme/` deverá evoluir para apresentar o projeto como uma plataforma epidemiológica de dados.

Estrutura sugerida:

```text
# SRAG Epidemiological Data API

## Objetivo
## Arquitetura
## Fonte dos dados
## Pipeline
## Métricas epidemiológicas
## API
## Exemplos
## Qualidade dos dados
## Machine Learning
## Roadmap
## Referências
```

## 28. Roadmap

### Fase 1 — Data Pipeline

CSV → validação → normalização → qualidade → Parquet.

### Fase 2 — DuckDB + Repository

Consultas epidemiológicas reutilizáveis.

### Fase 3 — FastAPI

Swagger/OpenAPI e endpoints agregados.

### Fase 4 — ML revisado

Predição, validação temporal e explicabilidade.

### Fase 5 — MCP Server

Ferramentas para agentes de IA consultarem a camada de serviços.

## 29. Estratégia de entrega

A implementação deve ser dividida em incrementos pequenos e revisáveis.

Sugestão de entregas:

1. pipeline e fundação da arquitetura;
2. repository e DuckDB;
3. API REST;
4. revisão de ML;
5. MCP Server.

Cada incremento deve manter testes e documentação compatíveis com o estado do projeto.

## 30. Restrições

- Alterações deste trabalho devem permanecer no repositório `GT09PetSaudeDigital/grupo-srag`.
- O escopo principal é a pasta `guilherme/`.
- Não modificar nenhum arquivo ou repositório relacionado a `infosetecinco/agrocifra`.
- Commits e pushes serão executados pelo usuário.
- A implementação não deve alterar dados RAW.
- A API não expõe registros individuais na V1.
- A API não aceita SQL arbitrário.
- MCP não consulta DuckDB diretamente.
- ML não deve compartilhar fitted preprocessing entre treino e teste.

## 31. Critérios de sucesso da V1

A V1 será considerada bem-sucedida quando:

1. dados de ao menos um ano puderem ser ingeridos de CSV para Parquet;
2. o pipeline gerar relatório de qualidade;
3. DuckDB consultar os arquivos Parquet;
4. a API iniciar localmente via FastAPI;
5. `/health` responder corretamente;
6. endpoints epidemiológicos básicos retornarem dados agregados;
7. filtros Brasil → UF → Município funcionarem;
8. etiologia estiver normalizada;
9. letalidade e proporção de UTI tiverem denominadores explícitos;
10. testes unitários e de integração principais passarem;
11. o desenho permitir adicionar MCP sem duplicar regras de negócio.

## 32. Decisões aprovadas

- arquitetura modular em camadas;
- Brasil → Estado → Município;
- Parquet + DuckDB;
- dados 2019–2026;
- ingestão incremental;
- V1 com dados agregados;
- arquitetura preparada para registros individuais anonimizados no futuro;
- etiologia incluída na V1;
- API pública somente leitura;
- Swagger/OpenAPI;
- MCP posterior;
- ML desacoplado da ingestão;
- usuário fará os commits e pushes.
