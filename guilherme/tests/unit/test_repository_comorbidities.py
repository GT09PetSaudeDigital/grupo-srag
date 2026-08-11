from contextlib import AbstractContextManager

from srag_api.data.repository import SragFilters, SragRepository


class FakeConnection(AbstractContextManager):
    def __init__(self):
        self.description = [("column_name",)]
        self._rows = []
        self.calls = []

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        params = list(params or [])
        normalized = " ".join(sql.split())
        self.calls.append((normalized, params))

        if normalized.startswith("DESCRIBE"):
            self._rows = [
                ("ANO",),
                ("UF",),
                ("CARDIOPATI",),
                ("DIABETES",),
                ("OBESIDADE",),
            ]
        elif '"CARDIOPATI" = 1' in normalized:
            self._rows = [(3,)]
        elif '"DIABETES" = 1' in normalized:
            self._rows = [(2,)]
        elif '"OBESIDADE" = 1' in normalized:
            self._rows = [(0,)]
        else:
            raise AssertionError(f"SQL inesperado: {normalized}")
        return self

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


def test_get_available_columns_reads_duckdb_schema(monkeypatch):
    connection = FakeConnection()
    repository = SragRepository("ignored")
    monkeypatch.setattr(repository, "_connect", lambda: connection)

    columns = repository.get_available_columns()

    assert {"CARDIOPATI", "DIABETES", "OBESIDADE"}.issubset(columns)


def test_comorbidity_distribution_counts_only_yes_and_existing_columns(monkeypatch):
    connection = FakeConnection()
    repository = SragRepository("ignored")
    monkeypatch.setattr(repository, "_connect", lambda: connection)

    result = repository.get_comorbidity_distribution(
        SragFilters(uf="PR"),
        {
            "CARDIOPATI": "CARDIOPATIA",
            "DIABETES": "DIABETES",
            "OBESIDADE": "OBESIDADE",
            "INEXISTENTE": "NAO_DEVE_APARECER",
        },
    )

    assert result == [
        {"comorbidade": "CARDIOPATIA", "casos": 3},
        {"comorbidade": "DIABETES", "casos": 2},
    ]

    count_calls = [call for call in connection.calls if call[0].startswith("SELECT COUNT")]
    assert all(call[1] == ["PR"] for call in count_calls)
    assert not any("INEXISTENTE" in call[0] for call in count_calls)
