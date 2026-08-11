import pandas as pd

ESSENTIAL_COLUMNS = frozenset(
    {
        "TP_IDADE",
        "NU_IDADE_N",
        "SG_UF",
        "ID_MUNICIP",
        "EVOLUCAO",
        "UTI",
    }
)

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(column).strip().upper() for column in result.columns]
    return result

def validate_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(ESSENTIAL_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError("Colunas essenciais ausentes: " + ", ".join(missing))
