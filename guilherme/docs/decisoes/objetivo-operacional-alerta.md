# Objetivo operacional do alerta de risco por SRAG

**Data da decisão:** 2026-09-25
**Status:** Caso de uso definido. Parâmetros quantitativos ainda em aberto.
**Relacionado:** `docs/superpowers/specs/2026-09-25-srag-ml-admission-v2-design.md` (seção 3)

## 1. Caso de uso

O alerta de risco por óbito de SRAG apoia **triagem e priorização de leito**.

Não é liberação de UTI, não é alocação de recurso de alto custo e não é
estratificação populacional para pesquisa. É uma ferramenta de fila: define
por qual paciente a equipe atende primeiro dentro de um leito.

## 2. Consequência imediata para a V1

Uma fila de triagem é limitada por **sensibilidade**, não por precisão. Uma
troca de turno perdida custa mais que um leito examinado à frente.

A política adotada na V1 — "maximizar recall sujeito a `precision >= 0,50`" —
produziu **recall de 3,44%** no teste temporal, com 665 alertas para 8.447
óbitos não detectados. Esse resultado é **inaceitável** para este caso de uso.

Consequências que a V2 precisa absorver:

1. A política de limiar da V1 está descartada. A restrição `precision >= 0,50`
   não vem do caso de uso e não pode ser o padrão implícito.
2. O alerta deixa de ser medido por recall e precision. Precisa ser medido
   por **volume por turno** e por **cobertura**: que fração dos óbitos
   entra nos primeiros lugares da fila.
3. A ordenação de risco que o modelo demonstra (ROC-AUC 0,8591 no teste de
   2026) passa a ser o produto principal. A pergunta deixa de ser "o modelo
   acerta?" e passa a ser "os óbitos estão no topo da fila?".
4. O número de alertas deixa de ser um detalhe de relato e vira uma
   restrição de projeto.

## 3. Parâmetros quantitativos ainda em aberto

A decisão de caso de uso está tomada. Os números abaixo **não** foram
definidos, e a task 9 do plano (política de limiar) não pode ser implementada
sem eles. Estão registrados aqui como pendentes, não preenchidos com
estimativas inventadas.

### 3.1 Volume aceitável de alertas por turno

Precisa responder: quantos pacientes por turno a equipe de triagem consegue
avaliar de fato? Um teto de 20 alertas por turno e outro de 200 não levam à
mesma política.

Determina: o teto de alertas, e portanto o recall viável.

### 3.2 Custo relativo de um falso positivo

Um paciente examinado e liberado sem necessidade, o custo é o tempo da
enfermagem? O bloqueio de um leito? A investigação de um quadro que não era
SRAG?

Determina: se a restrição é em precision, em tempo de equipe, ou nas duas.

### 3.3 Quem age e em quanto tempo

Quem lê a fila: médico de cama, enfermeiro, só a escala? E em quanto tempo
após a admissão a fila precisa estar disponível?

Determina: se o instante de predição é T0 (admissão) ou T1 (notificação).
Sem essa resposta, a seção 4 da spec da V2 — o momento da predição — não pode
ser fechada, e é ela que antecede as demais correções.

### 3.4 Posição alvo na fila

A cobertura desejada: os 50 mais graves da fila? os 100? Isso é o que
transforma ROC-AUC em métrica acionável.

Determina: a métrica de seleção do modelo.

## 4. O que já pode ser implementado sem esses números

As tasks 2 a 8 do plano da V2 não dependem dos parâmetros acima:

| Task | Depende dos parâmetros? |
|---|---|
| 2. Origem temporal das features | não |
| 3. Idade em anos | não |
| 4. Regras de data e feature temporal | não |
| 5. Identidade do episódio | não |
| 6. Ausência, ignorado e constantes | não |
| 7. Calendário e janelas temporais | não |
| 8. Protocolo de avaliação | não — a task 8 **é** o que vai medir o volume |
| 9. Política de limiar | **sim** |
| 10. Reexecução sobre a base nacional | **sim**, para a interpretação |

A task 8 entrega as métricas de volume de alertas e cobertura por posição na
fila. Ela não precisa dos parâmetros para existir; os parâmetros são o que
permite **julgar** o resultado dela.

## 5. Teste de aceite desta decisão

A task 1 só está completa quando:

- [x] o caso de uso está definido e registrado;
- [ ] o volume aceitável por turno está definido;
- [ ] o custo de um falso positivo está definido;
- [ ] quem age e em quanto tempo está definido;
- [ ] a posição alvo na fila está definida;
- [ ] `operational.py` valida o documento e `tests/unit/ml/test_operational.py`
  falha quando um campo obrigatório está vazio.

## 6. Nota sobre a V1

A V1 continua válida como resultado histórico. Ela não se torna inválida por
esta decisão: mede com honesto a política que aplicou, e o próprio relatório
diz que recall baixo é resultado da política, não do modelo, e que os dois
devem ser julgados separadamente.

O artefato `artifacts/ml-admission/20260906-104505/` não é reescrito.
