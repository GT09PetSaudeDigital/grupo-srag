"""Estabelece a linha de base da V1 com o protocolo de avaliacao novo.

Reaproveita o modelo ja salvo pela V1 e reavalia as mesmas particoes, sem
retrainar. A auditoria concluiu que o efeito das correcoes de dado sobre
AUC-PR e recall nunca foi medido. Esta linha de base e o ponto de
comparacao: sem ele, cada correcao da V2 e uma mudanca sem antes e sem
depois.

Uso:

```powershell
python scripts/baseline_v1.py --threshold 0.8489187193708023
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq
from numpy import flatnonzero

from srag_api.ml import evaluate_partition
from srag_api.ml.dataset import build_admission_dataset

VALIDATION_YEAR = 2025
TEST_YEAR = 2026
TRAIN_YEARS = (2019, 2020, 2021, 2022, 2023, 2024)

QUEUE_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)
SUBGROUP_COLUMNS = ("SG_UF",)
N_RESAMPLES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reavalia o modelo da V1 com o protocolo de avaliacao novo.",
    )
    parser.add_argument(
        "--parquet-glob",
        default="data/parquet/srag/ano=*/srag.parquet",
        help="Padrao glob dos Parquets normalizados.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="best_model.joblib produzido pela V1.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Limiar escolhido pela V1 em 2025.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baseline-v1"),
        help="Diretorio do relatorio em JSON.",
    )
    return parser.parse_args()


def load_partitions(pattern: str, features: list[str]) -> pd.DataFrame:
    """Le os Parquets com as colunas de que o modelo precisa."""
    from glob import glob

    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum Parquet encontrado para: {pattern}")

    desired = list(
        dict.fromkeys(
            (
                *features,
                "DESFECHO_NORMALIZADO",
                "ANO",
                "SG_UF",
                "FAIXA_ETARIA",
            )
        )
    )

    frames = []
    for path in files:
        available = set(pq.read_schema(path).names)
        selected = [column for column in desired if column in available]
        frames.append(pd.read_parquet(path, columns=selected))

    return pd.concat(frames, ignore_index=True)


def partition_of(year: int) -> str:
    if year in TRAIN_YEARS:
        return "treino"
    if year == VALIDATION_YEAR:
        return "validacao"
    if year == TEST_YEAR:
        return "teste"
    return f"ano_{year}"


def report_to_dict(report) -> dict:
    """Achata o relatorio para JSON, sem perder nenhum campo."""
    return {
        "partition": report.partition,
        "total": report.total,
        "positives": report.positives,
        "prevalence": report.prevalence,
        "auc_pr": report.auc_pr,
        "roc_auc": report.roc_auc,
        "auc_pr_interval": {
            "point_estimate": report.auc_pr_interval.point_estimate,
            "lower": report.auc_pr_interval.lower,
            "upper": report.auc_pr_interval.upper,
            "level": report.auc_pr_interval.level,
            "valid_resamples": report.auc_pr_interval.valid_resamples,
            "total_resamples": report.auc_pr_interval.total_resamples,
        },
        "roc_auc_interval": {
            "point_estimate": report.roc_auc_interval.point_estimate,
            "lower": report.roc_auc_interval.lower,
            "upper": report.roc_auc_interval.upper,
            "level": report.roc_auc_interval.level,
            "valid_resamples": report.roc_auc_interval.valid_resamples,
            "total_resamples": report.roc_auc_interval.total_resamples,
        },
        "brier_score": report.calibration.brier_score,
        "calibration_bins": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_predicted": b.mean_predicted,
                "observed_frequency": b.observed_frequency,
            }
            for b in report.calibration.bins
        ],
        "volume": {
            "threshold": report.volume.threshold,
            "alerts": report.volume.alerts,
            "alerts_per_1000": report.volume.alerts_per_1000,
            "true_positives": report.volume.true_positives,
            "false_positives": report.volume.false_positives,
            "false_negatives": report.volume.false_negatives,
            "true_negatives": report.volume.true_negatives,
            "sensitivity": report.volume.sensitivity,
            "specificity": report.volume.specificity,
            "ppv": report.volume.ppv,
            "npv": report.volume.npv,
        },
        "queue": [
            {
                "fraction": p.fraction,
                "queue_size": p.queue_size,
                "covered_positives": p.covered_positives,
                "total_positives": p.total_positives,
                "coverage": p.coverage,
                "precision": p.precision,
            }
            for p in report.queue
        ],
        "subgroups": [
            {
                "column": s.column,
                "key": s.key,
                "total": s.total,
                "positives": s.positives,
                "observed_rate": s.observed_rate,
                "auc_pr": s.auc_pr,
                "roc_auc": s.roc_auc,
                "brier_score": s.brier_score,
                "alerts": s.alerts,
                "alerts_per_1000": s.alerts_per_1000,
                "coverage_at_threshold": s.coverage_at_threshold,
            }
            for s in report.subgroups
        ],
    }


def main() -> int:
    args = parse_args()

    artifact = joblib.load(args.model)
    features = list(artifact["features"])
    pipeline = artifact["pipeline"]

    print(f"[DADOS] lendo {args.parquet_glob}")
    raw = load_partitions(args.parquet_glob, features)
    dataset = build_admission_dataset(raw)
    print(f"[DADOS] elegiveis: {len(dataset.X):,}")

    # build_admission_dataset filtra as linhas inelegiveis, entao os labels
    # do indice nao sao mais as posicoes originais. As particoes sao
    # resolvidas por mascara posicional sobre o proprio dataset.
    years = dataset.metadata["ANO"]
    masks = {
        "treino": years.isin(TRAIN_YEARS).to_numpy(),
        "validacao": (years == VALIDATION_YEAR).to_numpy(),
        "teste": (years == TEST_YEAR).to_numpy(),
    }

    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(
                f"A particao {name} ficou vazia. Verifique o glob e os anos."
            )

    partitions = {
        name: flatnonzero(mask) for name, mask in masks.items()
    }

    by_column: dict[str, list] = {}

    for name, positions in partitions.items():
        print(f"[EVAL] {name}: {len(positions):,} registros")
        X = dataset.X.iloc[positions]
        y = dataset.y.iloc[positions]
        metadata = dataset.metadata.iloc[positions]

        probabilities = pipeline.predict_proba(X)[:, 1]

        report = evaluate_partition(
            y,
            probabilities,
            threshold=args.threshold,
            partition=name,
            metadata=metadata,
            by=SUBGROUP_COLUMNS,
            queue_fractions=QUEUE_FRACTIONS,
            n_resamples=N_RESAMPLES,
            random_state=42,
        )

        print(
            f"  AUC-PR={report.auc_pr:.4f} "
            f"ROC-AUC={report.roc_auc:.4f} "
            f"Brier={report.calibration.brier_score:.4f} "
            f"alertas/mil={report.volume.alerts_per_1000:.1f} "
            f"sensibilidade={report.volume.sensitivity:.4f}"
        )
        for point in report.queue:
            print(
                f"  fila {point.fraction:.0%}: "
                f"{point.covered_positives} obitos "
                f"({point.coverage:.2%} de {point.total_positives}), "
                f"precision {point.precision:.2%}"
            )

        payload = report_to_dict(report)
        by_column[name] = payload

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "baseline_v1.json"
    destination.write_text(
        json.dumps(by_column, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
