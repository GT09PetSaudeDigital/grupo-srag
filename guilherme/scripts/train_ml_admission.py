from __future__ import annotations

import argparse
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd

from srag_api.ml import (
    ADMISSION_FEATURES,
    build_admission_dataset,
    run_admission_training,
    save_training_artifacts,
    temporal_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina os modelos de mortalidade por SRAG na admissao."
    )
    parser.add_argument(
        "--parquet-glob",
        required=True,
        help="Padrao glob dos Parquets normalizados de SRAG.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretorio de artefatos. Padrao: artifacts/ml-admission/<timestamp>.",
    )
    parser.add_argument(
        "--validation-year",
        type=int,
        default=2025,
        help="Ano usado para validacao e selecao de limiar.",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2026,
        help="Ano reservado para avaliacao final out-of-time.",
    )
    return parser.parse_args()


def load_normalized_parquets(pattern: str) -> pd.DataFrame:
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo Parquet encontrado para o padrao: {pattern}"
        )

    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()

    df = load_normalized_parquets(args.parquet_glob)
    dataset = build_admission_dataset(df)

    if "ANO" not in dataset.metadata.columns:
        raise ValueError("O dataset normalizado precisa conter a coluna ANO.")

    split = temporal_split(
        dataset.metadata["ANO"],
        validation_year=args.validation_year,
        test_year=args.test_year,
    )

    result = run_admission_training(dataset, split)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("artifacts") / "ml-admission" / timestamp
    )

    observed_train_years = sorted(
        {
            int(year)
            for year in dataset.metadata["ANO"].iloc[split.train_idx].dropna().tolist()
        }
    )

    metadata = {
        "timestamp": timestamp,
        "random_state": 42,
        "train_years": observed_train_years,
        "validation_year": args.validation_year,
        "test_year": args.test_year,
        "features_used": list(dataset.X.columns),
        "features_missing": sorted(
            set(ADMISSION_FEATURES) - set(dataset.X.columns)
        ),
    }

    paths = save_training_artifacts(
        result,
        output_dir=output_dir,
        metadata=metadata,
    )

    print(f"Treino: {result.train_size} registros")
    print(f"Validacao ({args.validation_year}): {result.validation_size} registros")
    print(f"Teste ({args.test_year}): {result.test_size} registros")
    for name, candidate in result.candidates.items():
        print(
            f"{name}: AUC-PR validacao="
            f"{candidate.validation_metrics.auc_pr:.4f}"
        )
    print(f"Melhor modelo: {result.best_model_name}")
    print(f"Limiar: {result.threshold:.6f}")
    print(f"Politica do limiar: {result.threshold_policy}")
    print(f"AUC-PR teste: {result.test_metrics.auc_pr:.4f}")
    print(f"Artefatos: {paths.run_metadata.parent}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
