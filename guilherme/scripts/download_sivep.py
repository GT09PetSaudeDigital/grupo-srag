"""Baixa os bancos publicos do SIVEP-Gripe do Ministerio da Saude.

Exemplos:

```powershell
# Lista as URLs descobertas, sem baixar nada
python scripts/download_sivep.py --years 2025 --list

# Baixa um ano especifico
python scripts/download_sivep.py --years 2025

# Baixa todos os anos suportados
python scripts/download_sivep.py --years all

# Quando a descoberta automatica falhar, informe a URL
python scripts/download_sivep.py --years 2025 --url https://...
```

Os arquivos sao gravados em ``data/raw/{ano}/`` com o nome original do
portal e tambem copiados para ``INFLUD{yy}.csv``, que e o nome esperado pelo
``scripts/ingest_all.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from srag_api.config import SUPPORTED_YEARS
from srag_api.data.download import (
    DownloadError,
    copy_to_stable_name,
    discover_source,
    download_file,
    format_megabytes,
    write_source_manifest,
)

DEFAULT_RAW_ROOT = Path("data/raw")


def resolve_years(raw_years: list[str]) -> list[int]:
    """Expande ``all`` e valida os anos informados."""
    if "all" in raw_years:
        return list(SUPPORTED_YEARS)

    years: list[int] = []
    for raw_year in raw_years:
        try:
            year = int(raw_year)
        except ValueError as exc:
            raise ValueError(f"Ano invalido: {raw_year}") from exc

        if year not in SUPPORTED_YEARS:
            supported = ", ".join(str(item) for item in SUPPORTED_YEARS)
            raise ValueError(f"Ano fora de 2019-2026: {year}. Suportados: {supported}")

        if year not in years:
            years.append(year)

    if not years:
        raise ValueError("Informe ao menos um ano ou use --years all.")

    return sorted(years)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baixa os bancos do SIVEP-Gripe para data/raw.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["all"],
        help="Anos a baixar, ou 'all' para 2019-2026.",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--format",
        default="csv",
        help="Formato publicado no portal. Use 'parquet' para arquivos menores.",
    )
    parser.add_argument(
        "--url",
        help=(
            "URL direta do arquivo. Necessaria apenas quando a descoberta "
            "automatica falhar."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="Mostra as URLs descobertas e sai sem baixar.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Baixa novamente mesmo com o arquivo ja presente.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        years = resolve_years(args.years)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.url and len(years) > 1:
        raise SystemExit(
            "--url serve a um unico ano. Informe um ano por vez."
        )

    for year in years:
        try:
            url, filename, origin = discover_source(
                year,
                args.format,
                explicit_url=args.url,
                timeout=args.timeout,
            )
        except DownloadError as exc:
            print(f"[ERRO] {year}: {exc}")
            continue

        print(f"[{year}] {filename} (origem: {origin})")
        print(f"       {url}")

        if args.list_only:
            continue

        year_dir = args.raw_root / str(year)
        stored = year_dir / filename

        try:
            written, downloaded = download_file(
                url,
                stored,
                timeout=args.timeout,
                force=args.force,
            )
        except DownloadError as exc:
            print(f"[ERRO] {year}: {exc}")
            continue

        if not downloaded:
            print(f"[SKIP] {year}: {filename} ja existe ({format_megabytes(written)})")
        else:
            write_source_manifest(stored, url, origin, written)
            print(f"[OK] {year}: {format_megabytes(written)}")

        if stored.suffix.lower() == ".csv":
            stable = copy_to_stable_name(
                stored,
                year_dir / f"INFLUD{str(year)[-2:]}.csv",
                force=args.force,
            )
            print(f"       pronto para ingest: {stable}")

    if args.list_only:
        print("\nUse sem --list para baixar.")


if __name__ == "__main__":
    main()
