# SRAG DuckDB Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development while implementing each behavior.

**Goal:** Adicionar uma camada somente-leitura com DuckDB sobre os Parquets produzidos na Fase 1, com filtros reutilizáveis e consultas epidemiológicas básicas.

**Architecture:** DuckDB abrirá os Parquets particionados como uma relation e criará uma view em memória chamada `srag`. `SragRepository` será o único componente que conhece SQL nesta fase. Valores de filtros entram como parâmetros; nomes de coluna, métricas e agrupamentos ficam em listas fechadas no código.

**Tech Stack:** Python, DuckDB, Parquet, pandas, pytest.

## Tasks

1. Adicionar DuckDB às dependências do projeto.
2. Materializar `MES` a partir de `DT_SIN_PRI` e preservar `SEM_PRI` como `SEMANA_EPIDEMIOLOGICA`.
3. Criar `SragFilters` e `SragRepository`.
4. Implementar contagens de casos, óbitos e UTI.
5. Implementar denominadores conhecidos e contagens ignoradas.
6. Implementar distribuições por faixa etária e etiologia.
7. Implementar séries mensais e por semana epidemiológica.
8. Implementar ranking de UF/município para casos, óbitos e UTI.
9. Cobrir filtros Brasil → UF → Município em testes de integração.
10. Rodar a suíte completa antes de seguir para Services/FastAPI.

## Constraints

- Não modificar `guilherme/analise_srag_pr.py`.
- Não modificar dados RAW.
- Nenhum SQL arbitrário recebido de usuário.
- Município por nome exige UF ou código do município.
- Filtros usam parâmetros SQL.
- Letalidade e proporção de UTI serão calculadas posteriormente na camada Services.
