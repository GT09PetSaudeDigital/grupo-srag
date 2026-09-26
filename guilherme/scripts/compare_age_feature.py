"""Compara, no mesmo modelo, a idade em anos contra a idade crua.

A auditoria afirmou que o problema da idade e demonstravel, mas que seu
efeito sobre AUC-PR e recall nunca foi medido. Este script existe para
medir, mantendo tudo constante exceto a coluna de idade: mesmo modelo,
mesma janela de treino, mesma semente, mesma pre-processamento e mesma
particao de validacao.

O desenho e um A/B com um unico fator variavel. Deliberadamente nao usa a
base nacional completa: o objetivo e isolar a idade, e treinar quatro
modelos sobre 3,6 milhoes de linhas nao cabe na memoria disponivel. A
janela menor muda o numero absoluto em relacao a linha de base da V1, mas
nao invalida a comparacao entre os dois bracos.

Uso:

```powershell
python scripts/compare_age_feature.py --train-years 2022 2023 2024
```
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from srag_api.ml.evaluation import evaluate_partition
from srag_api.ml.preprocessing import build_hist_gradient_boosting_preprocessor
from srag_api.ml.target import eligible_outcome_mask

RAW_AGE = "NU_IDADE_N"
AGE_IN_YEARS = "IDADE_ANOS"
VALIDATION_YEAR = 2025
RANDOM_STATE = 42

QUEUE_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)
N_RESAMPLES = 200

# Espelho de ADMISSION_FEATURES com a idade trocada. Fixado aqui de
# proposito: se a lista mudar, as duas colunas comparadas mudam juntas e a
# comparacao deixa de valer.
BASE_FEATURES: tuple[str, ...] = (
    "CS_SEXO",
    "CS_GESTANT",
    "FEBRE",
    "TOSSE",
    "GARGANTA",
    "DISPNEIA",
    "DESC_RESP",
    "SATURACAO",
    "DIARREIA",
    "VOMITO",
    "DOR_ABD",
    "FADIGA",
    "PERD_OLFT",
    "PERD_PALA",
    "OUTRO_SIN",
    "CARDIOPATI",
    "DIABETES",
    "PNEUMOPATI",
    "RENAL",
    "HEPATICA",
    "IMUNODEPRE",
    "OBESIDADE",
    "OUT_MORBI",
    "FATOR_RISC",
    "SG_UF",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B entre idade em anos e idade crua, com tudo o mais fixo.",
    )
    parser.add_argument(
        "--parquet-glob",
        default="data/parquet/srag/ano=*/srag.parquet",
    )
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        help="Anos de treino. Menos anos, menos memoria.",
    )
    parser.add_argument(
        "--validation-year",
        type=int,
        default=VALIDATION_YEAR,
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Limita as linhas de treino, 0 usa todas. Amostra estratificada.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8489187193708023,
        help="Limiar da V1, para comparar volume em condicao equivalente.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/age-comparison"),
    )
    return parser.parse_args()


def load_window(pattern: str, years: list[int]) -> pd.DataFrame:
    """Le apenas os anos pedidos, com as colunas de que os dois bracos precisam."""
    wanted = set(years)

    desired = list(
        dict.fromkeys(
            (
                *BASE_FEATURES,
                RAW_AGE,
                AGE_IN_YEARS,
                "DESFECHO_NORMALIZADO",
                "ANO",
                "SG_UF",
                "FAIXA_ETARIA",
            )
        )
    )

    frames = []
    for path in sorted(glob(pattern)):
        parts = Path(path).parts
        ano = next(
            (int(p.split("=")[1]) for p in parts if p.startswith("ano=")),
            None,
        )
        if ano not in wanted:
            continue
        available = set(pq.read_schema(path).names)
        selected = [c for c in desired if c in available]
        print(f"[DADOS] {ano}: {len(selected)} colunas de {path}")
        frames.append(pd.read_parquet(path, columns=selected))

    if not frames:
        raise ValueError(f"Nenhum Parquet dos anos {sorted(wanted)} em {pattern}.")

    return pd.concat(frames, ignore_index=True)


def build_arm(
    raw: pd.DataFrame,
    age_column: str,
    years: list[int],
    validation_year: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Monta X, y e as particoes para uma coluna de idade."""
    missing = [c for c in BASE_FEATURES if c not in raw.columns and c != age_column]
    if missing:
        raise ValueError(f"Colunas ausentes na origem: {missing}")
    if age_column not in raw.columns:
        raise ValueError(f"A coluna de idade {age_column} nao existe na origem.")

    eligible = raw.loc[eligible_outcome_mask(raw)].reset_index(drop=True)

    # A idade entra no lugar de uma coluna demografica qualquer: e a
    # unica diferenca entre os dois bracos.
    features = [age_column, *BASE_FEATURES]

    X = eligible[features].copy()
    y = eligible["DESFECHO_NORMALIZADO"].map({"CURA": 0, "OBITO_SRAG": 1}).astype(int)
    years_series = eligible["ANO"]

    is_train = years_series.isin(years).to_numpy()
    is_validation = (years_series == validation_year).to_numpy()

    metadata = eligible[["SG_UF", "FAIXA_ETARIA"]].reset_index(drop=True)

    return (
        X.loc[is_train].reset_index(drop=True),
        y.loc[is_train].reset_index(drop=True),
        X.loc[is_validation].reset_index(drop=True),
        y.loc[is_validation].reset_index(drop=True),
    ), is_train, is_validation, metadata


def subsample(X: pd.DataFrame, y: pd.Series, limit: int, seed: int):
    """Amostra estratificada, para caber na memoria."""
    if limit <= 0 or len(y) <= limit:
        return X, y
    positives = y == 1
    rate = float(positives.mean())
    positive_quota = max(1, int(limit * rate))
    negative_quota = limit - positive_quota

    generator = np.random.default_rng(seed)
    positive_positions = np.flatnonzero(positives.to_numpy())
    negative_positions = np.flatnonzero((~positives).to_numpy())

    picked = np.concatenate(
        (
            generator.choice(positive_positions, positive_quota, replace=False),
            generator.choice(negative_positions, negative_quota, replace=False),
        )
    )
    generator.shuffle(picked)
    return X.iloc[picked].reset_index(drop=True), y.iloc[picked].reset_index(drop=True)


def fit_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    metadata: pd.DataFrame,
    threshold: float,
    label: str,
    age_column: str,
) -> dict:
    numeric = [c for c in X_train.columns if c in (AGE_IN_YEARS, RAW_AGE)]
    categorical = [c for c in X_train.columns if c not in numeric]

    preprocessor = build_hist_gradient_boosting_preprocessor(
        numeric_features=numeric,
        categorical_features=categorical,
    )
    print(f"[TREINO {label}] ajustando pre-processador...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_validation_t = preprocessor.transform(X_validation)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    print(f"[TREINO {label}] ajustando HGB em {X_train_t.shape}...")
    model.fit(X_train_t, y_train)
    print(f"[TREINO {label}] pronto")

    probabilities = model.predict_proba(X_validation_t)[:, 1]
    del X_train_t, X_validation_t

    report = evaluate_partition(
        y_validation,
        probabilities,
        threshold=threshold,
        partition=f"validacao_{label}",
        metadata=metadata,
        by=("FAIXA_ETARIA",),
        queue_fractions=QUEUE_FRACTIONS,
        n_resamples=N_RESAMPLES,
        random_state=RANDOM_STATE,
    )

    print(
        f"[{label}] AUC-PR={report.auc_pr:.4f} "
        f"({report.auc_pr_interval.lower:.4f}-{report.auc_pr_interval.upper:.4f}) "
        f"ROC-AUC={report.roc_auc:.4f} "
        f"Brier={report.calibration.brier_score:.4f} "
        f"alertas/mil={report.volume.alerts_per_1000:.1f}"
    )
    for point in report.queue:
        print(
            f"  fila {point.fraction:.0%}: cobertura {point.coverage:.2%}, "
            f"precision {point.precision:.2%}"
        )

    return {
        "label": label,
        "idade": age_column,
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "auc_pr": report.auc_pr,
        "auc_pr_lower": report.auc_pr_interval.lower,
        "auc_pr_upper": report.auc_pr_interval.upper,
        "roc_auc": report.roc_auc,
        "brier_score": report.calibration.brier_score,
        "alerts_per_1000": report.volume.alerts_per_1000,
        "sensitivity_at_threshold": report.volume.sensitivity,
        "ppv_at_threshold": report.volume.ppv,
        "queue": [
            {
                "fraction": p.fraction,
                "coverage": p.coverage,
                "precision": p.precision,
            }
            for p in report.queue
        ],
    }


def main() -> int:
    args = parse_args()

    raw = load_window(args.parquet_glob, args.train_years + [args.validation_year])
    print(f"[DADOS] total elegivel antes do filtro: {len(raw):,}")

    results = {}

    for age_column in (RAW_AGE, AGE_IN_YEARS):
        (
            X_train,
            y_train,
            X_validation,
            y_validation,
        ), is_train, is_validation, metadata = build_arm(
            raw,
            age_column,
            args.train_years,
            args.validation_year,
        )

        X_train, y_train = subsample(
            X_train,
            y_train,
            args.max_train_rows,
            RANDOM_STATE,
        )
        print(
            f"[DADOS {age_column}] treino={len(y_train):,} "
            f"validacao={len(y_validation):,}"
        )

        results[age_column] = fit_and_evaluate(
            X_train,
            y_train,
            X_validation,
            y_validation,
            metadata.loc[is_validation].reset_index(drop=True),
            args.threshold,
            age_column,
            age_column,
        )

    raw_result = results[RAW_AGE]
    years_result = results[AGE_IN_YEARS]

    delta = {
        "auc_pr_delta": years_result["auc_pr"] - raw_result["auc_pr"],
        "roc_auc_delta": years_result["roc_auc"] - raw_result["roc_auc"],
        "brier_delta": years_result["brier_score"] - raw_result["brier_score"],
        "intervals_overlap": not (
            years_result["auc_pr_lower"] > raw_result["auc_pr_upper"]
            or raw_result["auc_pr_lower"] > years_result["auc_pr_upper"]
        ),
    }

    print("\n=== diferenca (anos menos cru) ===")
    print(f"  AUC-PR: {delta['auc_pr_delta']:+.4f}")
    print(f"  ROC-AUC: {delta['roc_auc_delta']:+.4f}")
    print(f"  Brier:   {delta['brier_delta']:+.4f}")
    print(f"  intervalos de AUC-PR se sobrepoem: {delta['intervals_overlap']}")

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "age_comparison.json"
    destination.write_text(
        json.dumps(
            {
                "train_years": args.train_years,
                "validation_year": args.validation_year,
                "max_train_rows": args.max_train_rows,
                "threshold": args.threshold,
                "model": "HistGradientBoostingClassifier",
                "arms": results,
                "delta": delta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
