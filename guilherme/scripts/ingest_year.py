from __future__ import annotations

import argparse
from pathlib import Path

from srag_api.data.ingest import ingest_year

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processa um ano do SIVEP-Gripe para Parquet."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser

def main() -> None:
    args = build_parser().parse_args()
    output = ingest_year(
        input_path=args.input,
        parquet_root=Path("data/parquet"),
        quality_root=Path("data/quality"),
        year=args.year,
        force=args.force,
    )
    print(f"Parquet gerado: {output}")

if __name__ == "__main__":
    main()
