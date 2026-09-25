"""Auditoria agregada, somente leitura, dos Parquets de 2019 a 2025.

Nao treina modelos e nao abre 2026. Nenhum identificador individual e exportado.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from srag_api.ml.features import ADMISSION_FEATURES


YEARS = tuple(range(2019, 2026))
DATES = ("DT_SIN_PRI", "DT_NOTIFIC", "DT_INTERNA")
EXTRA = ("ANO", "DESFECHO_NORMALIZADO", "TP_IDADE", "IDADE_ANOS", *DATES)
MISSING = "<AUSENTE>"


def parse_dates(values: pd.Series) -> pd.Series:
    """Aceita formatos explicitos, sem inferencia ambigua mes/dia."""
    text = values.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        mask = result.isna() & text.notna()
        if not mask.any():
            break
        parsed = pd.to_datetime(text.loc[mask], format=fmt, errors="coerce", utc=True)
        result.loc[mask] = parsed.dt.tz_convert(None)
    return result


def canonical(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip().str.upper()
    return result.replace("", pd.NA).fillna(MISSING)


def date_counts(frame: pd.DataFrame) -> dict[str, int]:
    parsed = {c: parse_dates(frame[c]) for c in DATES}
    counts = {}
    for c in DATES:
        present = canonical(frame[c]).ne(MISSING)
        counts[c + "_ausente"] = int((~present).sum())
        counts[c + "_invalida"] = int((present & parsed[c].isna()).sum())
    delta = (parsed["DT_NOTIFIC"] - parsed["DT_SIN_PRI"]).dt.days
    counts["intervalo_disponivel"] = int(delta.notna().sum())
    counts["intervalo_negativo"] = int(delta.lt(0).sum())
    counts["intervalo_maior_60_dias"] = int(delta.gt(60).sum())
    counts["intervalo_0_a_60_dias"] = int(delta.between(0, 60).sum())
    admission = (parsed["DT_NOTIFIC"] - parsed["DT_INTERNA"]).dt.days
    counts["notificacao_internacao_comparaveis"] = int(admission.notna().sum())
    counts["notificacao_apos_internacao"] = int(admission.gt(0).sum())
    counts["sintoma_apos_internacao"] = int(
        (parsed["DT_SIN_PRI"] > parsed["DT_INTERNA"]).sum()
    )
    return counts


def audit(root: Path, output: Path, batch_size: int = 65536) -> None:
    started = time.perf_counter()
    paths = [root / f"ano={year}" / "srag.parquet" for year in YEARS]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    population, schema_rows, age_rows, temporal_rows, provenance = [], [], [], [], []
    categories = defaultdict(Counter)
    eligible_by_year = {}
    date_extrema = []
    for year, path in zip(YEARS, paths):
        print(f"[AUDIT] {year}: leitura em lotes...", flush=True)
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        provenance.append({"year": year, "path": str(path.resolve()),
                           "size_bytes": path.stat().st_size,
                           "mtime_ns": path.stat().st_mtime_ns,
                           "rows": parquet.metadata.num_rows})
        required = set(EXTRA)
        if not required <= names:
            raise ValueError(f"{year}: colunas de auditoria ausentes {required - names}")
        for column in (*ADMISSION_FEATURES, *EXTRA):
            schema_rows.append({"ano": year, "coluna": column,
                                "presente": column in names,
                                "tipo": str(parquet.schema_arrow.field(column).type) if column in names else "ausente"})
        columns = sorted((set(ADMISSION_FEATURES) | set(EXTRA)) & names)
        outcomes, age, temporal = Counter(), Counter(), Counter()
        total = eligible = year_mismatch = 0
        minima, maxima = {}, {}
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            df = batch.to_pandas()
            total += len(df)
            outcomes.update(canonical(df["DESFECHO_NORMALIZADO"]).value_counts().to_dict())
            year_mismatch += int(pd.to_numeric(df["ANO"], errors="coerce").ne(year).sum())
            df = df.loc[df["DESFECHO_NORMALIZADO"].isin(["CURA", "OBITO_SRAG"])].copy()
            eligible += len(df)
            for c in ADMISSION_FEATURES:
                values = canonical(df[c]) if c in df else pd.Series(MISSING, index=df.index)
                categories[year, c].update(values.value_counts().to_dict())
            raw = pd.to_numeric(df["NU_IDADE_N"], errors="coerce")
            unit = pd.to_numeric(df["TP_IDADE"], errors="coerce")
            normalized = pd.to_numeric(df["IDADE_ANOS"], errors="coerce")
            expected = raw.where(unit.eq(3), np.nan)
            expected = expected.mask(unit.eq(1), raw / 365.25).mask(unit.eq(2), raw / 12)
            expected = expected.where(raw.ge(0) & expected.le(120))
            age.update({"registros": len(df), "unidade_dias": int(unit.eq(1).sum()),
                        "unidade_meses": int(unit.eq(2).sum()), "unidade_anos": int(unit.eq(3).sum()),
                        "unidade_invalida_ou_ausente": int((~unit.isin([1, 2, 3])).sum()),
                        "idade_bruta_ausente": int(raw.isna().sum()),
                        "idade_bruta_negativa": int(raw.lt(0).sum()),
                        "idade_bruta_maior_120": int(raw.gt(120).sum()),
                        "idade_anos_ausente": int(normalized.isna().sum()),
                        "idade_anos_fora_0_120": int((normalized.lt(0) | normalized.gt(120)).sum()),
                        "normalizacao_divergente": int((~np.isclose(normalized, expected, equal_nan=True)).sum()),
                        "menores_1_ano": int(normalized.lt(1).sum())})
            bins = pd.cut(normalized, [-np.inf, 0, 1, 5, 18, 40, 60, 80, 121, np.inf],
                          right=False).astype("string").fillna(MISSING)
            categories[year, "IDADE_ANOS_FAIXA"].update(bins.value_counts().to_dict())
            temporal.update(date_counts(df))
            for c in DATES:
                parsed = parse_dates(df[c]).dropna()
                if len(parsed):
                    minima[c] = min(minima.get(c, parsed.min()), parsed.min())
                    maxima[c] = max(maxima.get(c, parsed.max()), parsed.max())
            onset = parse_dates(df["DT_SIN_PRI"])
            temporal["ano_sintomas_diferente_arquivo"] += int((onset.notna() & onset.dt.year.ne(year)).sum())
        assert total == parquet.metadata.num_rows
        assert eligible == outcomes["CURA"] + outcomes["OBITO_SRAG"]
        eligible_by_year[year] = eligible
        for outcome, count in sorted(outcomes.items()):
            population.append({"ano": year, "desfecho": outcome, "registros": count,
                               "total_ano": total, "elegiveis": eligible,
                               "ano_coluna_divergente": year_mismatch})
        age_rows.append({"ano": year, **age})
        temporal_rows.append({"ano": year, "registros": eligible, **temporal})
        for c in DATES:
            date_extrema.append({"ano": year, "coluna": c,
                                 "minimo": str(minima.get(c)), "maximo": str(maxima.get(c))})

    category_rows, missing_rows, shifts = [], [], []
    for (year, column), counts in sorted(categories.items()):
        n = eligible_by_year[year]
        assert sum(counts.values()) == n
        for value, count in sorted(counts.items()):
            category_rows.append({"ano": year, "coluna": column, "valor": value,
                                  "registros": count, "fracao": count / n})
        # Marcadores candidatos; 9 nao e ignorado em idade. Confirmar dicionario.
        markers = {"9", "9.0", "IGNORADO", "I"} if column not in {"NU_IDADE_N", "SINT_ATE_NOTIF", "IDADE_ANOS_FAIXA", "SG_UF", "REGIAO"} else set()
        missing_rows.append({"ano": year, "coluna": column, "registros": n,
                             "ausentes": counts[MISSING], "fracao_ausente": counts[MISSING] / n,
                             "marcadores_ignorado": sum(counts[x] for x in markers),
                             "categorias_preenchidas": sum(v > 0 for k, v in counts.items() if k != MISSING)})
        if year > 2019:
            previous = categories[year - 1, column]
            keys = counts.keys() | previous.keys()
            tvd = sum(abs(counts[k] / n - previous[k] / eligible_by_year[year - 1]) for k in keys) / 2
            shifts.append({"ano_anterior": year - 1, "ano": year, "coluna": column,
                           "distancia_variacao_total": tvd,
                           "categorias_novas": json.dumps(sorted(k for k in counts if counts[k] > 0 and previous[k] == 0), ensure_ascii=False)})

    print("[AUDIT] repeticoes de chave candidata de notificacao...", flush=True)
    duplicates = []
    with tempfile.TemporaryDirectory(prefix="srag-audit-") as temp:
        with duckdb.connect() as con:
            con.execute("SET memory_limit='512MB'")
            con.execute("SET threads=2")
            con.execute("SET max_temp_directory_size='2GB'")
            con.execute("SET temp_directory=?", [temp])
            # Explicit file list: never expand a glob that could include 2026.
            con.read_parquet([str(p) for p in paths], union_by_name=True, hive_partitioning=False).create_view("source")
            con.execute("""CREATE VIEW keys AS SELECT ANO, NU_NOTIFIC, SG_UF_NOT, CO_MUN_NOT,
                DT_NOTIFIC, DESFECHO_NORMALIZADO FROM source
                WHERE DESFECHO_NORMALIZADO IN ('CURA', 'OBITO_SRAG')""")
            key = "NU_NOTIFIC, SG_UF_NOT, CO_MUN_NOT, DT_NOTIFIC"
            complete = "NU_NOTIFIC IS NOT NULL AND SG_UF_NOT IS NOT NULL AND CO_MUN_NOT IS NOT NULL AND DT_NOTIFIC IS NOT NULL"
            for scope, where in [(str(y), f"ANO={y}") for y in YEARS] + [("2019-2025", "TRUE")]:
                incomplete = con.execute(f"SELECT count(*) FROM keys WHERE ({where}) AND NOT ({complete})").fetchone()[0]
                row = con.execute(f"""SELECT count(*), coalesce(sum(n-1),0),
                    coalesce(sum(n),0), count(*) FILTER (WHERE outcomes > 1),
                    count(*) FILTER (WHERE year_count > 1)
                    FROM (SELECT count(*) n, count(DISTINCT DESFECHO_NORMALIZADO) outcomes,
                        count(DISTINCT ANO) year_count FROM keys WHERE ({where}) AND ({complete})
                        GROUP BY {key} HAVING count(*) > 1)""").fetchone()
                duplicates.append(dict(zip(["escopo", "chave_incompleta", "grupos_repetidos", "repeticoes_excedentes", "registros_nos_grupos", "grupos_desfecho_conflitante", "grupos_entre_anos"], [scope, incomplete, *row])))

    tables = {"desfechos": population, "schema": schema_rows, "idade": age_rows,
              "datas": temporal_rows, "datas_extremos": date_extrema,
              "categorias": category_rows, "preenchimento": missing_rows,
              "mudancas_anuais": shifts, "chaves_repetidas": duplicates}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "years": YEARS,
                "population": "preenchimento/idade/datas/chaves: CURA ou OBITO_SRAG; desfechos: todos",
                "batch_size": batch_size, "duration_seconds": time.perf_counter() - started,
                "files": provenance, "patient_identifiers_exported": False,
                "duplicate_key": ["NU_NOTIFIC", "SG_UF_NOT", "CO_MUN_NOT", "DT_NOTIFIC"],
                "versions": {"pandas": pd.__version__, "duckdb": duckdb.__version__}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {output}: {sum(x['rows'] for x in provenance)} registros, {manifest['duration_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet-root", type=Path, default=Path("data/parquet/srag"))
    args = parser.parse_args()
    audit(args.parquet_root, args.output_dir)
