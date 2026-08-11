import pandas as pd

from srag_api.data.clean import add_core_normalized_columns, normalize_outcome, normalize_yes_no

def test_normalize_yes_no_codes():
    assert normalize_yes_no(1) == "SIM"
    assert normalize_yes_no(2) == "NAO"
    assert normalize_yes_no(9) == "IGNORADO"
    assert normalize_yes_no(None) == "AUSENTE"

def test_normalize_outcome_codes():
    assert normalize_outcome(1) == "CURA"
    assert normalize_outcome(2) == "OBITO_SRAG"
    assert normalize_outcome(3) == "OBITO_OUTRAS_CAUSAS"
    assert normalize_outcome(9) == "IGNORADO"
    assert normalize_outcome(None) == "AUSENTE"
    assert normalize_outcome(7) == "OUTRO"

def test_add_core_normalized_columns_preserves_source_fields():
    df = pd.DataFrame({"EVOLUCAO": [1, 2, 9], "UTI": [2, 1, 9]})
    result = add_core_normalized_columns(df)
    assert result["DESFECHO_NORMALIZADO"].tolist() == ["CURA", "OBITO_SRAG", "IGNORADO"]
    assert result["FOI_UTI"].tolist() == ["NAO", "SIM", "IGNORADO"]
    assert result["OBITO_SRAG"].tolist() == [False, True, False]
