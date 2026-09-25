# SRAG ML V2 — Predição de Óbito na Admissão, com protocolo corrigido

**Data:** 2026-09-25
**Projeto:** GT09PetSaudeDigital/grupo-srag
**Escopo:** `guilherme/`
**Origem:** `docs/auditoria-ml-admissao-2019-2025.md` e `docs/relatorio-ml-admissao-v1.md`
**Status:** Design proposto para aprovação

## 1. Objetivo

Reexecutar a predição de óbito por SRAG na admissão corrigindo os defeitos
de dados e de protocolo encontrados na auditoria, e estabelecer um protocolo de
avaliação que permita afirmar o que o modelo serve e o que ele não serve.

A V1 demonstrou capacidade de ordenação de risco (ROC-AUC 0,8591 no teste
temporal) e, ao mesmo tempo, uma política de alertas clinicamente inútil
(recall de 3,44%). A V2 não deve tentar apenas aumentar AUC. Deve resolver a
pergunta que a V1 deixou em aberto: **qual é a utilidade operacional
pretendida, e o modelo a entrega?**

## 2. Por que a V1 não é comparável

A V1 usou `NU_IDADE_N` como idade numérica, sem a unidade. Em 2025, 27,49%
dos registros elegíveis tinham idade registrada em dias ou meses. O número
bruto `6` podia significar 6 dias, 6 meses ou 6 anos, e o modelo recebia
sempre a mesma entrada.

Além disso:

- 49,53% das linhas comparáveis de 2019–2025 têm notificação **posterior à
  internação** (56,50% em 2025). O modelo chamado de "de admissão" usava
  variáveis cujo instante de preenchimento não era o da admissão.
- 15 chaves de notificação aparecem tanto no treino quanto na validação,
  atravessando a fronteira temporal usada para escolher o modelo.
- Comorbidades têm 66% a 70% de ausência em 2025, e `FATOR_RISC` possui
  uma única categoria preenchida (`1.0`), virando constante depois da
  imputação por moda.
- O calendário de partição nunca foi explicitado: o arquivo `ano=2025`
  contém sintomas de 29/12/2024 a 03/01/2026 e notificações até 27/07/2026.

O resultado da V1 é um número válido para a configuração executada, mas
não para a pergunta de pesquisa declarada.

## 3. Pré-condição: objetivo operacional do alerta

**Nenhuma comparação de modelos começa antes desta seção.**

A V1 aplicou, sem justificativa explícita, a política "maximizar recall
sujeito a precision ≥ 0,50". O resultado — 665 alertas para 8.447 óbitos
não detectados — só é julgado adequado ou absurdo conforme a finalidade.

A V2 exige um documento curto, versionado, com:

- a decisão que o alerta deve apoiar (triagem em leito, liberação de UTI,
  alocação de recurso de alto custo, ou pesquisa de estratificação de risco);
- o custo de um falso positivo e o custo de um falso negativo, na mesma unidade;
- a prevalência-alvo e o volume de alertas aceitável em Operações por dia;
- quem age sobre o alerta e em que prazo.

Sem isso, qualquer escolha de limiar é arbitrária e a métrica de seleção
não responde a nenhuma pergunta útil.

## 4. Momento da predição e disponibilidade das features

### 4.1 Escolha explícita

A V2 declara um instante de predição entre:

- **T0 = admissão/internação** (`DT_INTERNA`), ou
- **T1 = notificação** (`DT_NOTIFIC`).

A escolha é uma decisão de pesquisa, não de implementação. As duas geram
perguntas diferentes e não devem ser misturadas.

### 4.2 Restrição de disponibilidade

Nenhuma variável pode entrar em `X` se puder ser conocida **depois** do
instante declarado. Isso passa a ser verificável por teste, não apenas por
convenção: para cada feature candidata, o pipeline deriva a origem
temporal e o catálogo passa a carregá-la.

### 4.3 Regras para datas

A auditoria encontrou anos 2109, 2202 e 2224 reconhecidos pelo parser, 14
datas de internação não reconhecidas e 2.511 linhas de 2025 com sintomas
posteriores à internação. A V2 deve:

- rejeitar anos implausíveis por regra explícita e documentada, não por
  tentativa-e-erro;
- tratar data não reconhecida como ausente, nunca como erro fatal;
- não derivar intervalos a partir de datas cuja ordem viola a hipótese
  (por exemplo, sintoma após a internação em um modelo de admissão), e
  registrar essas linhas como perda explícita em vez de silenciá-las.

### 4.4 Features temporais

`SINT_ATE_NOTIF` está no catálogo desde a V1 e nunca existiu: a coluna não é
materializada. O intervalo `DT_NOTIFIC - DT_SIN_PRI` é calculável em 100%
dos elegíveis auditados, mas depende do instante escolhido em 4.1.

A V2 decide entre:

- derivar `SINT_ATE_ADM = DT_INTERNA - DT_SIN_PRI`, coerente com T0; ou
- derivar `SINT_ATE_NOTIF`, coerente com T1; ou
- manter o grupo temporal vazio, com o motivo registrado.

Não é aceitável manter no catálogo uma feature que o pipeline não produz.

## 5. Idade em unidade consistente

O contrato da V2 passa a usar `IDADE_ANOS`, já calculada pela ingestão a
partir de `TP_IDADE` e `NU_IDADE_N`, e rejeita valores ausentes, negativos
ou acima do limite declarado.

Regras:

- `NU_IDADE_N` e `TP_IDADE` saem do catálogo do ML;
- a conversão é testada com dias, meses, anos e valores inválidos;
- a regra de descarte de valores improváveis é documentada como decisão do
  projeto, não como verdade sobre os dados (a auditoria encontrou 27 valores
  ausentes na idade normalizada em todo o conjunto auditado);
- o artefato da V1 permanece inalterado e não é reescrito.

Efeito sobre as métricas: **não é previamente conhecido**. A auditoria
registrou que o problema é demonstrável, mas o impacto em AUC-PR ou recall
ainda não foi medido. A V2 deve reportar a comparação, não a presumir.

## 6. Identidade do episódio e separação temporal

A auditoria encontrou 15 grupos repetidos (30 registros, 15 ocorrências
excedentes) entre arquivos, todos atravessando a fronteira treino–validação:
2022×2025 (1), 2023×2025 (1) e 2024×2025 (13). Não houve conflito de
desfecho entre CURA e OBITO_SRAG.

A V2 deve:

- adotar uma chave de episódio explícita, versionada e testada, com o que
  se faz quando parte da chave é nula (8.434 linhas ficaram fora da
  comparação na auditoria);
- aplicar a política **antes** da divisão temporal;
- **não** excluir registros automaticamente com base nessa chave: é
  evidência de sobreposição, não prova de duplicidade de paciente;
- medir quantos registros a política remove e reportar o efeito nas
  partições, em vez de descartar a diferença.

A chave não é identificador validado de paciente e pode deixar passar
repetições com campos alterados. Isso deve permanecer explícito nos
artefatos.

## 7. Ausência, ignorado e features constantes

### 7.1 Ausência como informação

Comorbidades com 66% a 70% de ausência em 2025 não podem ser tratadas como
"não" por omissão. A V2 compara, de forma explícita e medida:

1. a imputação atual por moda;
2. uma representação categórica que preserve `AUSENTE` e `IGNORADO` como
   categorias próprias;
3. a opção (2) acrescida de indicadores binários de ausência.

Nenhuma variante é adotada por convenção. A escolha vem da comparação de
métricas **e** da leitura do caso, e a representação adotada fica no
artefato.

### 7.2 Códigos por variável

- `CS_GESTANT` contém o código `0` em todos os anos, fora das categorias da
  referência consultada. Deve ser revisado, não removido automaticamente.
- "Ignorado" é tratado por variável, conforme a codificação do SIVEP, e
  não por uma regra global.

### 7.3 Features constantes

`FATOR_RISC` tem uma única categoria preenchida em todos os anos auditados.
A V2 deve detectar features constantes após o tratamento de ausência e
reportá-las, em vez de escondê-las no artefato. A decisão de manter ou
remover precisa ser registrada.

## 8. Calendário de partição e janelas temporais

A V1 usou a coluna `ANO`, atribuída pelo ano do arquivo. A auditoria mostrou
que esse ano não coincide com o ano civil das datas.

O agravante é que a partição por ano de arquivo **não garante ordem
cronológica**. O arquivo `ano=2025` contém início de sintomas de 29/12/2024,
anteriores a parte do `ano=2024`, e notificações até 27/07/2026. Isso significa
que:

- linhas de validação podem ser cronologicamente **anteriores** a linhas de
  treino;
- a afirmação "validação temporal" da V1 é mais fraca do que a documentação
  sugere, embora os anos de arquivo não se sobreponham;
- o mesmo vale para o teste de 2026, cujos registros mais antigos podem
  preceder parte da validação de 2025.

A V2 deve:

- usar a data efetiva do registro, não o ano do arquivo, para ordenar as
  partições, com a data declarada em `ANO` apenas como agrupamento secundário;
- documentar o calendário efetivo de cada partição, com o intervalo real de
  início de sintomas e de notificação observados;
- medir e reportar quantos pares treino/validação estão invertidos na ordem
  do tempo, para que a magnitude do problema seja conhecida;
- oferecer mais de uma janela temporal de validação, para que a conclusão
  não dependa de um único ponto de corte no tempo;
- declarar explicitamente que o arquivo de 2026 está parcial e qual o
  período efetivamente coberto.

## 9. Protocolo de avaliação

Além de ROC-AUC e Average Precision, a V2 reporta:

- **Calibração.** Brier score e curva de confiabilidade por faixa. A V1 não avaliou
  calibração; o score não pode ser lido como risco absoluto sem isso.
- **Volume de alertas.** Alertas por mil admissions, por ano e por subgrupo.
  Sem esse número, precision e recall não descrevem carga de trabalho.
- **Subgrupos.** Métricas por faixa etária, UF e ano de treino. A
  distância de distribuição entre anos foi de 0,5033 entre 2019 e 2020, o
  que impede tratar a base como homogênea.
- **Incerteza.** Intervalos por bootstrap. A V1 não reportou nenhum, e a
  diferença pontual entre modelos (0,2692 a 0,2875 em AUC-PR) não foi
  testada.

### 9.1 Sobre o conjunto de teste

O ano de 2026 já foi usado como teste na V1 e seus resultados já foram
examinados. **Não pode ser reutilizado como teste intocado de uma V2
orientada pelos resultados da V1.**

Opções, a decidir na spec de execução:

- manter 2026 como teste e declarar que a V2 é uma reavaliação, não uma
  validação independente;
- ou reservar 2027 para o teste final independente, aceitando que a
  avaliação de 2026 é de desenvolvimento.

A segunda é metodologicamente mais forte e exige que a V2 não decida
nada olhando 2026.

## 10. Fora do escopo da V2

Não implementar nesta etapa:

- redes convolucionais, XGBoost ou qualquer estimador além dos quatro já
  existentes;
- ajuste de hiperparâmetros antes de corrigir os itens das seções 4 a 8;
- interpretabilidade por SHAP ou LIME;
- endpoint de predição na API;
- registro ou versionamento de modelos em produção;
- os Modelos B (laboratório) e C (evolução durante a internação), que são
  perguntas diferentes.

## 11. Critérios de aceite

A V2 só é considerada pronta quando:

- o instante de predição está declarado e a disponibilidade das features é
  verificada por teste;
- a idade entra em anos, com testes de dias, meses, anos e inválidos;
- a política de episódio está documentada, aplicada antes da divisão e seu
  efeito nas partições foi medido;
- a representação de ausência foi comparada e a escolha está justificada;
- features constantes foram detectadas e reportadas;
- o calendário de partição está documentado, incluindo o recorte parcial;
- a avaliação inclui calibração, volume de alertas, subgrupos e intervalos;
- o objetivo operacional do alerta está documentado e a política de limiar
  deriva dele;
- a suíte completa do projeto continua passando.

## 12. Ordem sugerida

A ordem não é estética. Definir o momento da predição (seção 4) muda o que
é legítimo usar como feature, e portanto antecede a correção da idade e do
tratamento de ausência.

1. Objetivo operacional do alerta (seção 3).
2. Momento da predição e regras de data (seção 4).
3. Idade em anos (seção 5).
4. Identidade do episódio (seção 6).
5. Ausência e constantes (seção 7).
6. Calendário e janelas (seção 8).
7. Protocolo de avaliação (seção 9).
8. Escolha do limiar orientada ao objetivo (seção 3).
9. Comparação de modelos e relatório.

## Fontes locais

- `docs/auditoria-ml-admissao-2019-2025.md` — achados e ações propostas.
- `docs/relatorio-ml-admissao-v1.md` — execução de 4.108.827 registros,
  identificação `20260906-104505`.
- `artifacts/audit-ml-admission/2019-2025-v1/` — contagens por ano,
  preenchimento, datas, chaves repetidas e mudanças anuais.
- `docs/superpowers/specs/2026-08-18-srag-ml-admission-mortality-design.md` —
  catálogo de features e bloqueio de leakage da V1, que permanecem válidos.
