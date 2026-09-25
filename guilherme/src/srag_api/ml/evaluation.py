"""Protocolo de avaliacao do modelo de mortalidade por SRAG.

A V1 reportou AUC-PR, ROC-AUC, recall e precision. A auditoria
correspondente apontou quatro lacunas: nao houve avaliacao de calibracao,
nao houve intervalo de confianca, nao houve metricas por subgrupo e o
volume de alertas nunca foi reportado.

Este modulo cobre as quatro. A mudanca de caso de uso para triagem de
leito torna a cobertura da fila mais relevante que recall e precision
isolados: na fila, a pergunta e quantas pessoas que vao morrer estao no
topo, e nao quantos alarmes o modelo emite.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

MISSING_GROUP = "AUSENTE"

DEFAULT_QUEUE_FRACTIONS = (0.05, 0.10, 0.20, 0.50)
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95


# --------------------------------------------------------------------------
# Validacao de entrada
# --------------------------------------------------------------------------


def _validate_common(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Confere tamanho e dominio das probabilidades.

    Nao exige as duas classes: Brier, volume e cobertura sao definidos
    sobre uma fatia com uma classe so, e e comum em subgrupos.
    """
    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true e probabilities devem possuir o mesmo tamanho: "
            f"{len(y_true)} e {len(probabilities)}."
        )

    values = np.asarray(probabilities, dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("As probabilidades contem valores nao finitos.")

    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        raise ValueError(
            "As probabilidades devem estar no intervalo [0, 1]. "
            f"Observado: min={values.min()}, max={values.max()}."
        )

    return values


def _require_both_classes(y_true: pd.Series, context: str) -> None:
    classes = set(pd.Series(y_true).dropna().unique().tolist())
    if classes != {0, 1}:
        raise ValueError(
            f"{context} exige as duas classes 0 e 1. "
            f"Ausentes: {sorted({0, 1} - classes)}. Presentes: {sorted(classes)}."
        )


def _validate_ranked_inputs(
    y_true: pd.Series,
    probabilities: np.ndarray,
    context: str = "A particao",
) -> np.ndarray:
    """Valida o par e exige as duas classes, como as metricas de ordenacao.

    AUC nao tem leitura com uma classe so: o scikit-learn emite um aviso e
    devolve ``nan``, o que contaminaria um intervalo de confianca.
    """
    values = _validate_common(y_true, probabilities)
    _require_both_classes(y_true, context)
    return values


def _as_binary(y_true: pd.Series) -> np.ndarray:
    return pd.Series(y_true).to_numpy(dtype=int)


# --------------------------------------------------------------------------
# Calibracao
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBin:
    """Faixa de probabilidade prevista e o que nela realmente aconteceu."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_frequency: float


@dataclass(frozen=True)
class CalibrationMetrics:
    brier_score: float
    bins: tuple[CalibrationBin, ...]


def brier_score(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Erro quadratico medio da probabilidade prevista.

    Zero quando a probabilidade prevista e exatamente a observada. Sem
    isto, um score nao pode ser lido como risco absoluto.
    """
    values = _validate_common(y_true, probabilities)
    return float(brier_score_loss(_as_binary(y_true), values))


def calibration_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Curva de confiabilidade por faixa de probabilidade prevista.

    Faixas vazias nao entram no resultado: uma faixa sem observacao nao
    diz nada sobre calibracao e inflaria a leitura do grafico.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins deve ser pelo menos 1. Recebido: {n_bins}.")

    values = _validate_common(y_true, probabilities)
    observed = _as_binary(y_true)

    indices = np.minimum((values * n_bins).astype(int), n_bins - 1)

    bins: list[CalibrationBin] = []
    for position in range(n_bins):
        mask = indices == position
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            CalibrationBin(
                lower=position / n_bins,
                upper=(position + 1) / n_bins,
                count=count,
                mean_predicted=float(values[mask].mean()),
                observed_frequency=float(observed[mask].mean()),
            )
        )

    return CalibrationMetrics(
        brier_score=float(brier_score_loss(observed, values)),
        bins=tuple(bins),
    )


# --------------------------------------------------------------------------
# Volume de alertas
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertVolume:
    """Carga de trabalho que um limiar produz, alem da acuracia."""

    threshold: float
    total: int
    positives: int
    alerts: int
    alerts_per_1000: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    sensitivity: float
    specificity: float
    ppv: float
    npv: float


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def alert_volume(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> AlertVolume:
    """Traduz um limiar em volume de alertas e nas quatro celulas.

    ``alerts_per_1000`` e a medida de carga de trabalho: quantos pacientes
    a equipe precisa avaliar a cada mil admissoes. Precision e recall, sem
    esse numero, nao descrevem nenhuma carga.
    """
    values = _validate_common(y_true, probabilities)
    observed = _as_binary(y_true)

    predicted = (values >= threshold).astype(int)

    true_positives = int(((predicted == 1) & (observed == 1)).sum())
    false_positives = int(((predicted == 1) & (observed == 0)).sum())
    false_negatives = int(((predicted == 0) & (observed == 1)).sum())
    true_negatives = int(((predicted == 0) & (observed == 0)).sum())
    alerts = true_positives + false_positives
    total = len(observed)

    return AlertVolume(
        threshold=float(threshold),
        total=total,
        positives=int(observed.sum()),
        alerts=alerts,
        alerts_per_1000=1000.0 * alerts / total if total else float("nan"),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        sensitivity=_safe_ratio(true_positives, true_positives + false_negatives),
        specificity=_safe_ratio(true_negatives, true_negatives + false_positives),
        ppv=_safe_ratio(true_positives, alerts),
        npv=_safe_ratio(true_negatives, true_negatives + false_negatives),
    )


# --------------------------------------------------------------------------
# Cobertura da fila
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QueuePoint:
    """Cobertura de obitos numa fracao da fila ordenada por risco."""

    fraction: float
    queue_size: int
    alerts: int
    covered_positives: int
    total_positives: int
    coverage: float
    precision: float


def queue_coverage(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    fractions: Sequence[float] = DEFAULT_QUEUE_FRACTIONS,
) -> tuple[QueuePoint, ...]:
    """Responde: se a equipe so avalia o topo da fila, quantos obitos ela ve?

    Ordena por probabilidade decrescente e mede, para cada fracao da fila,
    quantos dos obitos totais ficam dentro dela. E a metrica que traduz
    ordenacao de risco em operacao: na triagem de leito, 5% da fila com
    30% dos obitos e um resultado util, e 5% com 3% nao e.
    """
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                "Cada fracao em 'fractions' deve estar no intervalo (0, 1]. "
                f"Recebido: {fraction}."
            )

    values = _validate_common(y_true, probabilities)
    observed = _as_binary(y_true)
    total_positives = int(observed.sum())

    order = np.argsort(-values, kind="stable")
    ranked = observed[order]
    cumulative_positives = np.cumsum(ranked)

    points: list[QueuePoint] = []
    for fraction in fractions:
        queue_size = max(1, int(round(fraction * len(observed))))
        covered = int(cumulative_positives[queue_size - 1])
        points.append(
            QueuePoint(
                fraction=float(fraction),
                queue_size=queue_size,
                alerts=queue_size,
                covered_positives=covered,
                total_positives=total_positives,
                coverage=_safe_ratio(covered, total_positives),
                precision=_safe_ratio(covered, queue_size),
            )
        )

    return tuple(points)


# --------------------------------------------------------------------------
# Incerteza
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceInterval:
    point_estimate: float
    lower: float
    upper: float
    level: float
    valid_resamples: int
    total_resamples: int


def average_precision(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """AUC-PR, a metrica de selecao do modelo."""
    return float(average_precision_score(_as_binary(y_true), probabilities))


def roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """ROC-AUC, reported alongside AUC-PR for orientation only."""
    return float(roc_auc_score(_as_binary(y_true), probabilities))


METRICS: dict[str, Callable[[pd.Series, np.ndarray], float]] = {
    "auc_pr": average_precision,
    "roc_auc": roc_auc,
    "brier": brier_score,
}


def bootstrap_confidence_interval(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    metric: str,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = 42,
) -> ConfidenceInterval:
    """Intervalo de percentis por bootstrap.

    A V1 reportou apenas o valor pontual. Sem intervalo, a diferenca entre
    0,2692 e 0,2875 de AUC-PR entre modelos nao pode ser distinguida de
    ruido.

    Reamostras que ficam com uma classe so sao descartadas: ``roc_auc`` nao
    existe nelas. Quantas foram descartadas entra no resultado, porque um
    numero de reamostras validas que cai muito e, em si, um achado.
    """
    if metric not in METRICS:
        raise ValueError(
            f"Metrica desconhecida: {metric}. Use uma de: {sorted(METRICS)}."
        )
    if n_resamples < 1:
        raise ValueError(f"n_resamples deve ser pelo menos 1. Recebido: {n_resamples}.")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level deve estar entre 0 e 1. Recebido: {level}.")

    values = _validate_ranked_inputs(y_true, probabilities)
    observed = _as_binary(y_true)
    function = METRICS[metric]

    generator = np.random.default_rng(random_state)
    size = len(observed)
    estimates: list[float] = []

    for _ in range(n_resamples):
        picked = generator.integers(0, size, size=size)
        resampled = observed[picked]
        # A reamostra pode ficar com uma classe so, e entao AUC nao tem
        # leitura. O scikit-learn devolve nan com um aviso em vez de erro,
        # e um nan dentro do quantil contamina o intervalo inteiro.
        if set(resampled.tolist()) != {0, 1}:
            continue
        estimates.append(float(function(pd.Series(resampled), values[picked])))

    point = function(pd.Series(observed), values)

    if not estimates:
        return ConfidenceInterval(
            point_estimate=point,
            lower=float("nan"),
            upper=float("nan"),
            level=level,
            valid_resamples=0,
            total_resamples=n_resamples,
        )

    tail = (1.0 - level) / 2.0
    return ConfidenceInterval(
        point_estimate=point,
        lower=float(np.quantile(estimates, tail)),
        upper=float(np.quantile(estimates, 1.0 - tail)),
        level=level,
        valid_resamples=len(estimates),
        total_resamples=n_resamples,
    )


# --------------------------------------------------------------------------
# Subgrupos
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgroupMetrics:
    """Metricas de uma fatia da populacao."""

    column: str
    key: str
    total: int
    positives: int
    observed_rate: float
    auc_pr: float | None
    roc_auc: float | None
    brier_score: float | None
    alerts: int
    alerts_per_1000: float
    coverage_at_threshold: float


def _group_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return MISSING_GROUP
    try:
        if pd.isna(value):
            return MISSING_GROUP
    except (TypeError, ValueError):
        pass
    return str(value)


def subgroup_metrics(
    metadata: pd.DataFrame,
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
    by: str,
) -> tuple[SubgroupMetrics, ...]:
    """Repete as metricas por faixa de uma coluna de metadados.

    A distancia de distribuicao entre 2019 e 2020 foi de 0,5033 na
    auditoria. Uma unica metrica nacional esconde essa heterogeneidade.
    Faixa com uma classe so recebe ``None`` em AUC, porque nao ha o que
    ordenar, mas continua com Brier e volume.
    """
    values = _validate_common(y_true, probabilities)
    observed = _as_binary(y_true)

    if by not in metadata.columns:
        raise ValueError(
            f"Coluna de metadados ausente: {by}. Disponiveis: "
            f"{sorted(map(str, metadata.columns))}."
        )

    if len(metadata) != len(observed):
        raise ValueError(
            "metadata e y_true devem ter o mesmo tamanho: "
            f"{len(metadata)} e {len(observed)}."
        )

    keys = metadata[by].reset_index(drop=True).map(_group_key)

    results: list[SubgroupMetrics] = []
    for key in sorted(keys.unique().tolist()):
        mask = (keys == key).to_numpy()
        subgroup_y = pd.Series(observed[mask])
        subgroup_p = values[mask]

        positives = int(subgroup_y.sum())
        total = int(mask.sum())
        classes = set(subgroup_y.unique().tolist())

        volume = alert_volume(subgroup_y, subgroup_p, threshold=threshold)

        results.append(
            SubgroupMetrics(
                column=by,
                key=key,
                total=total,
                positives=positives,
                observed_rate=_safe_ratio(positives, total),
                auc_pr=(
                    average_precision(subgroup_y, subgroup_p)
                    if classes == {0, 1}
                    else None
                ),
                roc_auc=(
                    roc_auc(subgroup_y, subgroup_p) if classes == {0, 1} else None
                ),
                brier_score=brier_score(subgroup_y, subgroup_p),
                alerts=volume.alerts,
                alerts_per_1000=volume.alerts_per_1000,
                coverage_at_threshold=volume.sensitivity,
            )
        )

    return tuple(results)


# --------------------------------------------------------------------------
# Relatorio agregado
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationReport:
    partition: str
    total: int
    positives: int
    prevalence: float
    auc_pr: float
    roc_auc: float
    calibration: CalibrationMetrics
    volume: AlertVolume
    queue: tuple[QueuePoint, ...]
    auc_pr_interval: ConfidenceInterval | None
    roc_auc_interval: ConfidenceInterval | None
    subgroups: tuple[SubgroupMetrics, ...]


def evaluate_partition(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
    partition: str = "particao",
    n_bins: int = 10,
    queue_fractions: Sequence[float] = DEFAULT_QUEUE_FRACTIONS,
    metadata: pd.DataFrame | None = None,
    by: Sequence[str] = (),
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = 42,
) -> EvaluationReport:
    """Reune as medidas que a auditoria apontou como ausentes na V1."""
    values = _validate_ranked_inputs(y_true, probabilities)
    observed = _as_binary(y_true)

    subgroups: tuple[SubgroupMetrics, ...] = ()
    if by:
        if metadata is None:
            raise ValueError("metadata e obrigatorio quando ha colunas em 'by'.")
        collected: list[SubgroupMetrics] = []
        for column in by:
            collected.extend(
                subgroup_metrics(
                    metadata,
                    pd.Series(observed),
                    values,
                    threshold=threshold,
                    by=column,
                )
            )
        subgroups = tuple(collected)

    return EvaluationReport(
        partition=partition,
        total=len(observed),
        positives=int(observed.sum()),
        prevalence=float(observed.mean()),
        auc_pr=average_precision(pd.Series(observed), values),
        roc_auc=roc_auc(pd.Series(observed), values),
        calibration=calibration_metrics(
            pd.Series(observed),
            values,
            n_bins=n_bins,
        ),
        volume=alert_volume(
            pd.Series(observed),
            values,
            threshold=threshold,
        ),
        queue=queue_coverage(
            pd.Series(observed),
            values,
            fractions=queue_fractions,
        ),
        auc_pr_interval=bootstrap_confidence_interval(
            pd.Series(observed),
            values,
            metric="auc_pr",
            n_resamples=n_resamples,
            level=level,
            random_state=random_state,
        ),
        roc_auc_interval=bootstrap_confidence_interval(
            pd.Series(observed),
            values,
            metric="roc_auc",
            n_resamples=n_resamples,
            level=level,
            random_state=random_state + 1,
        ),
        subgroups=subgroups,
    )
