# Resultado — efeito da idade em anos

**Data:** 2026-09-26
**Origem:** task 3 do plano da V2
**Comando:** `python scripts/compare_age_feature.py --train-years 2022 2023 2024`
**Artefato:** `artifacts/age-comparison/age_comparison.json`

## Pergunta

A auditoria afirmou, na seção 1, que o uso de `NU_IDADE_N` sem unidade é um
problema demonstrável, mas que "o problema é demonstrável, porém seu efeito
quantitativo na AUC-PR ou no recall ainda não foi medido". Esta execução
mede.

## Desenho

A/B com um único fator variável. Idênticos nos dois braços: modelo
HistGradientBoosting, janela de treino 2022–2024, semente 42,
pré-processamento ordinal ajustado só no treino, e validação em 2025 com
305.362 registros. Muda apenas a coluna de idade: `NU_IDADE_N` contra
`IDADE_ANOS`.

979.264 registros de treino, 26 features em cada braço.

## Resultado

| Métrica | `NU_IDADE_N` | `IDADE_ANOS` | Diferença |
|---|---:|---:|---:|
| AUC-PR | 0,3109 (0,3050–0,3157) | 0,3103 (0,3046–0,3153) | **−0,0006** |
| ROC-AUC | 0,8574 | 0,8584 | +0,0010 |
| Brier | 0,1346 | 0,1346 | −0,000001 |
| Alertas por mil (corte 0,8489) | 17,1 | 17,0 | −0,1 |

Cobertura da fila em 2025, nos dois braços, é indistinguível:

| Fração da fila | Cobertura (cru) | Cobertura (anos) |
|---|---:|---:|
| 1% | 6,65% | 6,57% |
| 5% | 24,19% | 24,10% |
| 10% | 41,53% | 41,51% |
| 20% | 67,07% | 67,02% |

**Os intervalos de AUC-PR se sobrepõem quase por completo.** A diferença
observada é da ordem do ruído do próprio bootstrap.

## Conclusão

A troca **não melhora a desempenho.** Ela deve ser mantida por higiene de
dados e interpretabilidade, não por ganho de métrica, e essa distinção
precisa constar de qualquer relatório.

O motivo provável é que a informação não se perde por completo: em
`NU_IDADE_N`, um valor pequeno e uma unidade de dias carregam a mesma
direção ordinal que a idade, e o HistGradientBoosting trabalha por
divisões de limiar, não por escala. A unidade ambígua atrapalha mais a
leitura humana do resultado do que a escala do modelo.

Isso não generaliza para outros modelos. Regressão logística e Random
Forest com escalonamento ou distância receberiam a mesma coluna de forma
diferente, e a comparação abaixo é específica do HistGradientBoosting.

## Observação lateral que merece registro

Esta execução usou 2022–2024 e obteve AUC-PR de 0,3109 em 2025. A linha
de base da V1, treinada em 2019–2024 com os quatro modelos, obteve 0,2875
na mesma partição de validação. **A janela menor teve desempenho melhor.**

A proporção de óbitos por ano varia muito: 28,98% em 2020, 28,86% em 2021,
18,99% em 2022, 9,93% em 2023, 8,62% em 2024, 7,54% em 2025. Treinar com
2019–2021 mistura a era da pandemia na qual dois em cada três
internados morriam, e a distribuição de rótulos muda por um fator de quatro
em cinco anos.

Isso é evidência de que a **janela de treino é uma variável de projeto**,
não um detalhe. Não foi testado de forma controlada nesta execução — a
janela mudou junto com o modelo, de quatro candidatos para um — então não
é conclusão, é hipótese que a V2 deve testar com atenção.

O limiar da V1 (0,8489) não transfere entre configurações: ele produz 17,1
alertas por mil neste modelo contra 5,0 na linha de base da V1. Limiar é
propriedade de um modelo, e deve ser revisto sempre que o modelo muda.

## Limitações

- Um modelo, uma janela, uma semente. Não é o protocolo definitivo da V2.
- A janela de treino não é a da V1, então os valores absolutos não são
  comparáveis com `artifacts/ml-admission/20260906-104505/`.
- 200 reamostras de bootstrap, suficiente para leitura de ordem de grandeza
  e insuficiente para afirmar significância.
- A execução autenticada visual e a recalibração por subgrupo não foram
  feitas aqui.
