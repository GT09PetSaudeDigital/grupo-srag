# Auditoria de dados para a V2 — SRAG na admissão

Data: 06/09/2026. Branch: `feature/ml-admission-training`.

## Escopo e método

Foram auditados os sete arquivos `data/parquet/srag/ano=2019/srag.parquet` até `ano=2025/srag.parquet`: **4.445.703 registros**, dos quais **3.954.671** com desfecho CURA ou OBITO_SRAG. A soma dos elegíveis de 2019–2024 é 3.649.309; a de 2025 é 305.362, coincidentes com a V1.

Não foi aberto o arquivo `ano=2026`, nem realizada nova predição, seleção ou treinamento. A expressão “2019–2025” neste relatório se refere às partições dos arquivos, e não garante que todas as datas internas estejam nesses anos; essa distinção é um dos achados da auditoria.

O script `scripts/audit_ml_admission.py` faz leitura de colunas selecionadas em lotes de 65.536 registros. Contagens e distribuições usam todos os registros elegíveis, sem amostragem. Desfechos são contados antes do filtro. A busca de chaves repetidas usa DuckDB com limite configurado de 512 MB, duas threads e até 2 GB de armazenamento temporário. Esses limites não representam medição de RAM total do processo. A execução final durou **80,5 segundos**.

Os Parquets foram somente lidos. Os CSVs de saída contêm estatísticas agregadas, sem números de notificação ou linhas individuais. As fontes locais, tamanhos e datas de modificação constam de `artifacts/audit-ml-admission/2019-2025-v1/manifest.json`.

## Síntese dos achados

| Prioridade | Achado | Consequência para a V2 |
|---|---|---|
| Alta | O ML usa idade bruta sem unidade, embora exista idade normalizada | Corrigir a representação da idade antes de comparar modelos |
| Alta | 15 chaves candidatas de notificação aparecem tanto no treino quanto na validação | Revisar episódios e garantir separação por registro/episódio antes da divisão temporal |
| Alta | Muitas notificações são posteriores à internação | Definir o instante da predição e auditar disponibilidade das features |
| Alta | `FATOR_RISC` vira constante após a imputação atual | Rever codificação e tratamento de ausência |
| Alta | Comorbidades com 66%–70% de ausência em 2025 | Comparar tratamento explícito de ausência, sem assumir que vazio significa “não” |
| Média | Forte mudança de perfil entre anos | Avaliar em múltiplas janelas temporais e por subgrupos |
| Média | Códigos e datas suspeitos; features previstas não derivadas | Definir regras documentadas antes de novos experimentos |

## 1. Idade: problema confirmado na representação do ML

O catálogo de ML inclui `NU_IDADE_N`, mas não `TP_IDADE` nem `IDADE_ANOS`. O pré-processamento trata `NU_IDADE_N` como número contínuo, apenas imputando e padronizando. Entretanto, a ingestão já calcula `IDADE_ANOS`, convertendo dias e meses para anos. Portanto, na V1, um valor bruto 6 pode representar 6 dias, 6 meses ou 6 anos, sem que o modelo receba a unidade.

O significado das unidades está documentado no [dicionário SIVEP-Gripe, página 4](https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/pdfs/dicionario_de_dados_srag_hosp_17_02_2022.pdf): `TP_IDADE` distingue dia, mês e ano. A referência consultada é de 2022; diferenças entre versões anuais exigem conferência específica.

| Ano do arquivo | Elegíveis | Idade em dias ou meses | Percentual |
|---|---:|---:|---:|
| 2019 | 44.678 | 13.408 | 30,01% |
| 2020 | 1.088.676 | 20.660 | 1,90% |
| 2021 | 1.536.691 | 45.386 | 2,95% |
| 2022 | 487.734 | 55.074 | 11,29% |
| 2023 | 251.095 | 61.803 | 24,61% |
| 2024 | 240.435 | 62.063 | 25,81% |
| 2025 | 305.362 | 83.956 | 27,49% |

Em todos os registros elegíveis, `IDADE_ANOS` coincide com o recálculo das regras locais, inclusive a transformação de valores inválidos em ausentes. Não houve unidade ausente ou fora de 1/2/3. Foram encontrados 27 valores ausentes na idade normalizada; a regra local rejeita idades negativas e acima de 120 anos. Esse limite é uma regra do projeto e deve ser documentado, não confundido com prova de que todas essas observações são erros.

**Ação proposta:** usar idade em anos no contrato da V2, com testes de dias/meses/anos e controle de valores inválidos. Manter o artefato V1 inalterado. O problema é demonstrável, mas seu efeito quantitativo na AUC-PR ou no recall ainda não foi medido.

## 2. Ausência, imputação e categorias

Entre os 305.362 registros elegíveis de 2025:

| Feature | Ausentes | Percentual ausente |
|---|---:|---:|
| `CARDIOPATI` | 202.507 | 66,32% |
| `DIABETES` | 206.071 | 67,48% |
| `PNEUMOPATI` | 209.317 | 68,55% |
| `IMUNODEPRE` | 210.301 | 68,87% |
| `RENAL` | 211.337 | 69,21% |
| `HEPATICA` | 212.156 | 69,48% |
| `OBESIDADE` | 212.276 | 69,52% |
| `FATOR_RISC` | 178.293 | 58,39% |

### `FATOR_RISC` perde toda a variação

Em todos os anos auditados, a única categoria preenchida de `FATOR_RISC` é `1.0`; o restante é ausente. Com `SimpleImputer(strategy="most_frequent")`, todos os vazios recebem 1. O resultado é uma feature constante para os estimadores. Essa conclusão decorre diretamente das contagens e do código, sem reabrir o modelo salvo.

O [dicionário consultado, página 9](https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/pdfs/dicionario_de_dados_srag_hosp_17_02_2022.pdf) prevê respostas sim, não e ignorado para esse campo. A auditoria não determina por que o extrato local contém apenas 1 ou vazio. É necessário conferir o preenchimento e a origem antes de recodificar.

### Categorias e lacunas históricas

- `CS_GESTANT` contém código `0` em todos os anos: 234 casos em 2025. O código não aparece entre as categorias da referência consultada; deve ser revisado, não removido automaticamente.
- `DOR_ABD`, `FADIGA`, `PERD_OLFT` e `PERD_PALA` estavam ausentes em mais de 99% dos elegíveis de 2019. Não são colunas ausentes do schema: são colunas quase vazias naquele ano.
- `REGIAO` e `SINT_ATE_NOTIF` não existem nos schemas auditados. O CSV de preenchimento representa essa ausência estrutural como 100% ausente; `schema.csv` permite distingui-la de valores nulos em colunas existentes.
- `SG_UF` tem 27 categorias preenchidas em 2025 e 104 ausências. Os valores por ano estão em `categorias.csv`.

**Ação proposta:** comparar a imputação atual com representação explícita de ausência e, quando adequado, indicadores de ausência. Tratar códigos “ignorado” por variável. Não transformar vazio em “não” sem fundamentação no processo de coleta. Os marcadores em `preenchimento.csv` são candidatos de ignorado, não uma validação completa de todos os dicionários anuais.

## 3. Datas e disponibilidade no momento da predição

As datas brutas estão em strings ISO com milissegundos. Após usar o parser correspondente, todas as datas de sintomas e notificação dos elegíveis foram reconhecidas, sem ausência. Assim, o intervalo `DT_NOTIFIC - DT_SIN_PRI` é **calculável em 100% dos elegíveis** auditados. Não houve intervalo negativo. Há 71.400 intervalos acima de 60 dias; 60 é um marcador exploratório, não um limite clínico nem regra de exclusão.

Em 2025, 3.281 intervalos excedem 60 dias (1,07%). A feature prevista `SINT_ATE_NOTIF` está ausente porque não foi materializada no fluxo consultado, e não por inexistência das duas datas de origem.

Entretanto, **calculável não significa disponível na admissão**:

- Em 2019–2025, 1.853.173 das 3.741.672 linhas com notificação e internação comparáveis têm notificação posterior à internação (49,53%).
- Em 2025, são 164.773 de 291.642 linhas comparáveis (56,50%).
- Há 14 datas de internação não reconhecidas pelo parser e extremos reconhecidos, porém suspeitos: por exemplo, anos 2109, 2202 e 2224. Valores parseáveis também precisam de validação de plausibilidade.
- Sintomas posteriores à internação aparecem em 2.511 linhas de 2025. Isso exige investigação do contexto; não prova automaticamente erro de digitação.

**Ação proposta:** definir se a predição ocorre na admissão ou na notificação. Para admissão, não incluir automaticamente o intervalo até uma notificação futura. Um intervalo até a internação pode ser candidato, mas exige regras para datas inconsistentes, contexto do episódio e informação disponível naquele instante. O dicionário define os eventos de sintomas e preenchimento da ficha; não comprova quando cada sintoma/comorbidade foi efetivamente registrado.

### O ano do arquivo não equivale sempre ao ano civil das datas

No arquivo `ano=2025`, o início dos sintomas vai de 29/12/2024 a 03/01/2026; há notificações até 27/07/2026. Os limites de início dos sintomas entre arquivos sugerem organização epidemiológica anual, mas essa interpretação deve ser confirmada na documentação de extração.

O código de ingestão atribui `ANO` a partir do ano informado para o arquivo, e a divisão ML usa essa coluna. Não houve divergência entre a coluna `ANO` e o nome da partição. Isso **não** prova separação por data civil nem reprodução de uma base tal como disponível no encerramento de cada ano. Na V2, explicitar o calendário de partição e a disponibilidade dos registros/desfechos em cada janela de validação.

## 4. Possíveis duplicidades e sobreposição treino–validação

A chave candidata usada foi `NU_NOTIFIC + SG_UF_NOT + CO_MUN_NOT + DT_NOTIFIC`, restrita a CURA/OBITO_SRAG. Linhas com componentes nulos ficaram fora da comparação: **8.434** no conjunto auditado.

Não houve repetição dessa chave dentro de cada arquivo anual. Entre arquivos, houve **15 grupos repetidos, com 30 registros e 15 ocorrências excedentes**, sem conflito de desfecho entre CURA e OBITO_SRAG.

Uma consulta agregada complementar localizou os pares:

| Par de anos | Chaves repetidas |
|---|---:|
| 2022 e 2025 | 1 |
| 2023 e 2025 | 1 |
| 2024 e 2025 | 13 |

Todos os grupos encontrados atravessam a divisão treino–validação da V1. Isso é um sinal de sobreposição de notificações a investigar, embora pequeno em quantidade. Não foi estimado impacto nas métricas. A chave não é um identificador validado de paciente e pode deixar passar repetições com campos alterados. Não houve tentativa de reidentificação nem exportação das chaves individuais.

Os relatórios prévios `data/quality/quality_2019.json` até `quality_2025.json` informam zero linhas integralmente duplicadas em cada entrada. Essa checagem anterior é distinta: não exclui notificações repetidas com diferenças de campos ou presentes em arquivos diferentes.

**Ação proposta:** investigar a identidade do episódio e implementar uma política documentada de deduplicação/separação antes da divisão temporal. Não excluir registros automaticamente apenas com base nesta chave.

## 5. Desfechos e mudanças temporais

| Ano | Total no Parquet | Elegíveis | Óbitos SRAG | Óbitos/elegíveis |
|---|---:|---:|---:|---:|
| 2019 | 48.941 | 44.678 | 5.423 | 12,14% |
| 2020 | 1.206.920 | 1.088.676 | 315.445 | 28,98% |
| 2021 | 1.745.672 | 1.536.691 | 443.524 | 28,86% |
| 2022 | 560.577 | 487.734 | 92.635 | 18,99% |
| 2023 | 279.453 | 251.095 | 24.944 | 9,93% |
| 2024 | 267.986 | 240.435 | 20.730 | 8,62% |
| 2025 | 336.154 | 305.362 | 23.023 | 7,54% |

Em 2025, foram excluídos da população binária 15.515 desfechos ausentes, 6.188 ignorados e 9.089 óbitos por outras causas: 30.792 registros. Não foi avaliado se a exclusão por desfecho ausente é seletiva em relação ao perfil dos casos; isso permanece como limitação.

A participação de registros com idade em dias/meses muda de 2,95% em 2021 para 27,49% em 2025. A proporção de óbitos também muda substancialmente. A distância de variação total das faixas de idade normalizada é 0,5033 entre 2019 e 2020 e 0,3148 entre 2021 e 2022. Essa medida compara distribuições: 0 indica igualdade e 1 indica suportes disjuntos; não é um teste de significância. As categorias incluem ausência, e a idade bruta é comparada por seus valores discretos no CSV.

**Ação proposta:** validar em múltiplas janelas temporais anteriores a 2026 e reportar métricas por idade, ano e UF. Não atribuir automaticamente as mudanças a uma causa única nem prometer que a correção da idade resolverá a perda de desempenho.

## Entregáveis e verificação

Pasta: `artifacts/audit-ml-admission/2019-2025-v1/`.

| Arquivo | Conteúdo |
|---|---|
| `manifest.json` | Escopo, arquivos, tamanhos, datas de modificação, versões e duração |
| `desfechos.csv` | Desfechos de todos os registros por ano |
| `schema.csv` | Presença e tipos das colunas selecionadas |
| `preenchimento.csv` | Ausências, marcadores candidatos de ignorado e cardinalidade entre elegíveis |
| `categorias.csv` | Frequências de features e faixas de idade entre elegíveis |
| `idade.csv` | Unidades, valores suspeitos e consistência da conversão |
| `datas.csv`, `datas_extremos.csv` | Intervalos, contagens e extremos das datas |
| `chaves_repetidas.csv` | Contagens de repetição por ano e entre anos |
| `mudancas_anuais.csv` | Distâncias de distribuição e categorias novas entre anos consecutivos |

Há três testes específicos: datas dia/mês e ISO; formato real com milissegundos; auditoria sintética em lotes unitários, incluindo duplicata entre lotes e arquivo 2026 deliberadamente inválido para garantir que não seja aberto. As contagens de cada feature são reconciliadas com o número de elegíveis, e as linhas lidas com o metadata do Parquet.

Limitações: não é uma auditoria de prontuários; não valida identidade de paciente, completude de todos os desfechos, calibração ou performance por subgrupo. Ausências são perfiladas entre elegíveis, não comparadas entre incluídos e excluídos. Os arquivos não receberam hashes completos; tamanho/data de modificação são evidência de procedência limitada. Nenhum dado bruto foi corrigido.

## Encaminhamento para a V2

Antes de otimizar hiperparâmetros, priorizar: (1) idade em unidade consistente; (2) revisão das 15 chaves sobrepostas e do calendário de partição; (3) definição do instante da predição; (4) política explícita para ausência/ignorado. Só então comparar modelos e pontos de corte com foco em recall, reportando precision e volume de alertas, e planejar calibração e avaliação por subgrupos.

Esta auditoria acrescenta limitações à leitura da V1; não altera seus resultados históricos. Ganhos preditivos ainda precisam ser demonstrados por experimentos. O arquivo de teste 2026 permanece fora desta auditoria e não deve ser reutilizado como teste supostamente intocado de uma V2 orientada pelos resultados já observados.
