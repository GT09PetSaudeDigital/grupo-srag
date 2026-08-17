import json
import pandas as pd

from srag_api.data.quality import build_quality_report, write_quality_report

def test_quality_report_counts_missing_and_duplicates(tmp_path):
    raw = pd.DataFrame({
        "TP_IDADE": [3, 3, 3],
        "NU_IDADE_N": [70, 70, None],
        "SG_UF": ["PR", "PR", None],
        "ID_MUNICIP": ["CURITIBA", "CURITIBA", None],
        "EVOLUCAO": [2, 2, 9],
        "UTI": [1, 1, None],
    })
    processed = raw.drop_duplicates().copy()
    processed["IDADE_ANOS"] = [70.0, None]
    processed["ETIOLOGIA_DETALHADA"] = ["SARS-CoV-2", "NAO_IDENTIFICADA"]

    report = build_quality_report(raw, processed, 2025)
    assert report["duplicados"] == 1
    assert report["idade_ausente"] == 1
    assert report["municipio_ausente"] == 1
    assert report["etiologia_nao_identificada"] == 1

    output = tmp_path / "quality_2025.json"
    write_quality_report(report, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["ano"] == 2025
