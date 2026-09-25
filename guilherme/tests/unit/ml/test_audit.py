import json

import pandas as pd

from scripts.audit_ml_admission import audit, parse_dates


def test_parse_dates_rejects_invalid_and_preserves_day_month():
    parsed = parse_dates(pd.Series(["03/04/2025", "2025-04-03", "31/02/2025", None]))
    assert parsed.iloc[0] == pd.Timestamp("2025-04-03")
    assert parsed.iloc[1] == pd.Timestamp("2025-04-03")
    assert parsed.iloc[2:].isna().all()


def test_parse_dates_accepts_parquet_iso_milliseconds():
    parsed = parse_dates(pd.Series(["2025-07-19T00:00:00.000Z"]))
    assert parsed.iloc[0] == pd.Timestamp("2025-07-19")


def test_audit_counts_batches_and_excludes_2026(tmp_path):
    root = tmp_path / "source"
    for year in range(2019, 2026):
        folder = root / f"ano={year}"
        folder.mkdir(parents=True)
        pd.DataFrame({
            "ANO": [year] * 3,
            "DESFECHO_NORMALIZADO": ["CURA", "OBITO_SRAG", "IGNORADO"],
            "NU_IDADE_N": [6, 30, 20], "TP_IDADE": [2, 3, 3],
            "IDADE_ANOS": [0.5, 30, 20], "CS_SEXO": ["F", None, "M"],
            "DT_SIN_PRI": [f"01/01/{year}"] * 3,
            "DT_NOTIFIC": [f"03/01/{year}"] * 3,
            "DT_INTERNA": [f"02/01/{year}"] * 3,
            "NU_NOTIFIC": [123, 123, 999], "SG_UF_NOT": ["MT"] * 3,
            "CO_MUN_NOT": [1] * 3,
        }).to_parquet(folder / "srag.parquet", index=False)
    excluded = root / "ano=2026"
    excluded.mkdir()
    (excluded / "srag.parquet").write_bytes(b"not a parquet: must not be opened")
    output = tmp_path / "audit"
    audit(root, output, batch_size=1)
    age = pd.read_csv(output / "idade.csv")
    assert age["registros"].tolist() == [2] * 7
    assert age["unidade_meses"].tolist() == [1] * 7
    assert age["normalizacao_divergente"].sum() == 0
    dates = pd.read_csv(output / "datas.csv")
    assert dates["intervalo_0_a_60_dias"].tolist() == [2] * 7
    assert dates["notificacao_apos_internacao"].tolist() == [2] * 7
    duplicates = pd.read_csv(output / "chaves_repetidas.csv")
    assert duplicates.iloc[0]["repeticoes_excedentes"] == 1
    assert duplicates.iloc[-1]["grupos_repetidos"] == 7
    missing = pd.read_csv(output / "preenchimento.csv")
    assert missing.loc[missing.coluna.eq("CS_SEXO"), "ausentes"].tolist() == [1] * 7
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["years"] == list(range(2019, 2026))
