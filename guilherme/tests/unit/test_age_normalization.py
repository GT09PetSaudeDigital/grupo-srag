import math
import pandas as pd

from srag_api.data.clean import add_normalized_age_columns, classify_age_band, normalize_age

def test_normalize_age_from_days():
    assert math.isclose(normalize_age(1, 20), 20 / 365.25, rel_tol=1e-6)

def test_normalize_age_from_months():
    assert math.isclose(normalize_age(2, 18), 1.5, rel_tol=1e-6)

def test_normalize_age_from_years():
    assert normalize_age(3, 67) == 67.0

def test_normalize_age_rejects_negative_value():
    assert normalize_age(3, -1) is None

def test_normalize_age_rejects_unknown_unit():
    assert normalize_age(9, 30) is None

def test_normalize_age_rejects_more_than_120_years():
    assert normalize_age(3, 121) is None

def test_classify_age_band():
    assert classify_age_band(0.5) == "<1"
    assert classify_age_band(4.0) == "1-4"
    assert classify_age_band(17.0) == "12-17"
    assert classify_age_band(60.0) == "60-74"
    assert classify_age_band(80.0) == "75+"

def test_add_normalized_age_columns():
    df = pd.DataFrame({"TP_IDADE": [1, 2, 3], "NU_IDADE_N": [30, 18, 75]})
    result = add_normalized_age_columns(df)
    assert result["FAIXA_ETARIA"].tolist() == ["<1", "1-4", "75+"]
