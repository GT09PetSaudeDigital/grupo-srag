from __future__ import annotations

import argparse
from pathlib import Path

from srag_api.config import SUPPORTED_YEARS
from srag_api.data.ingest import ingest_year

def discover_year_file(raw_root: Path, year: int) -> Path | None:
    year_dir = raw_root / str(year)
    if not year_dir.exists():
        return None

    csv_files = sorted(year_dir.glob("*.csv"))
    if not csv_files:
        return None

    if len(csv_files) > 1:
        raise ValueError(
            f"Mais de um CSV encontrado para {year}: "
            + ", ".join(path.name for path in csv_files)
        )

    return csv_files[0]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processa todos os anos disponiveis do SIVEP-Gripe."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    return parser

def main() -> None:
    args = build_parser().parse_args()

    for year in SUPPORTED_YEARS:
        input_path = discover_year_file(args.raw_root, year)

        if input_path is None:
            print(f"[SKIP] {year}: nenhum CSV encontrado")
            continue

        try:
            output = ingest_year(
                input_path=input_path,
                parquet_root=Path("data/parquet"),
                quality_root=Path("data/quality"),
                year=year,
                force=args.force,
            )
        except FileExistsError as exc:
            print(f"[SKIP] {exc}")
            continue

        print(f"[OK] {year}: {output}")

if __name__ == "__main__":
    main()
