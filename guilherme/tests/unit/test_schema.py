import pandas as pd
import pytest

from srag_api.config import AGE_BANDS, SUPPORTED_YEARS
from srag_api.data.schema import (
    ESSENTIAL_COLUMNS,
    normalize_column_names,
    validate_required_columns,
)

def test_supported_years_are_2019_through_2026():
    assert SUPPORTED_YEARS == tuple(range(2019, 2027))

def test_age_bands_have_expected_labels():
    labels = [label for _, _, label in AGE_BANDS]
    assert labels == ["<1", "1-4", "5-11", "12-17", "18-29", "30-44", "45-59", "60-74", "75+"]

def test_normalize_column_names_strips_and_uppercases():
    df = pd.DataFrame(columns=[" tp_idade ", "Nu_Idade_N", " evolucao"])
    result = normalize_column_names(df)
    assert list(result.columns) == ["TP_IDADE", "NU_IDADE_N", "EVOLUCAO"]

def test_validate_required_columns_accepts_minimum_schema():
    df = pd.DataFrame(columns=sorted(ESSENTIAL_COLUMNS))
    validate_required_columns(df)

def test_validate_required_columns_reports_missing_fields():
    df = pd.DataFrame(columns=["TP_IDADE", "NU_IDADE_N"])
    with pytest.raises(ValueError, match="Colunas essenciais ausentes"):
        validate_required_columns(df)
