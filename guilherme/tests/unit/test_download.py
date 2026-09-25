import pytest

from srag_api.data import download


def _source(year: int, name: str, file_format: str | None = None) -> download.SourceFile:
    resolved = file_format or name.rsplit(".", 1)[-1]
    return download.SourceFile(
        year=year,
        name=name,
        file_format=resolved,
        url=download.build_s3_file_url(year, name),
    )


DATASET_PAGE = """
<html><body>
  <a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2019/INFLUD19-23-03-2026.csv">2019 csv</a>
  <a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2019/INFLUD19-23-03-2026.parquet">2019 parquet</a>
  <a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2025/INFLUD25-14-09-2026.csv">2025 csv</a>
  <a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2025/INFLUD25-14-09-2026.parquet">2025 parquet</a>
  <a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2026/INFLUD26-14-09-2026.csv">2026 csv</a>
</body></html>
"""


def test_year_short_label_uses_two_digits():
    assert download.year_short_label(2019) == "19"
    assert download.year_short_label(2026) == "26"


def test_filename_for_stamp_keeps_portal_pattern():
    assert download.filename_for_stamp(2025, "14-09-2026", "csv") == (
        "INFLUD25-14-09-2026.csv"
    )


def test_validate_format_rejects_unknown_format():
    with pytest.raises(ValueError, match="Formato nao suportado"):
        download.validate_format("xlsx")


def test_validate_format_accepts_uppercase_and_dot():
    assert download.validate_format(".PARQUET") == "parquet"


def test_stamp_to_date_reads_portal_pattern():
    assert download.stamp_to_date("INFLUD25-14-09-2026.csv").isoformat() == "2026-09-14"


def test_stamp_to_date_returns_none_without_pattern():
    assert download.stamp_to_date("INFLUD25.csv") is None


def test_parse_dataset_page_extracts_year_format_and_url():
    sources = download.parse_dataset_page(DATASET_PAGE)

    years = sorted({source.year for source in sources})
    formats = sorted({source.file_format for source in sources})

    assert years == [2019, 2025, 2026]
    assert formats == ["csv", "parquet"]


def test_parse_dataset_page_keeps_full_url():
    sources = download.parse_dataset_page(DATASET_PAGE)

    chosen = download.select_year_source(sources, 2025, "parquet")

    assert chosen.url == (
        "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/"
        "SRAG/2025/INFLUD25-14-09-2026.parquet"
    )


def test_parse_dataset_page_ignores_duplicated_links():
    page = DATASET_PAGE + DATASET_PAGE

    sources = download.parse_dataset_page(page)

    assert len(sources) == 5


def test_parse_dataset_page_fails_when_portal_has_no_files():
    with pytest.raises(download.DownloadError, match="nenhum arquivo"):
        download.parse_dataset_page("<html><body>sem arquivos</body></html>")


def test_select_year_source_picks_most_recent_bank():
    sources = [
        _source(2025, "INFLUD25-23-03-2026.csv"),
        _source(2025, "INFLUD25-14-09-2026.csv"),
    ]

    chosen = download.select_year_source(sources, 2025, "csv")

    assert chosen.name == "INFLUD25-14-09-2026.csv"


def test_select_year_source_reports_published_formats():
    sources = [_source(2025, "INFLUD25-14-09-2026.xml")]

    with pytest.raises(download.ResourceNotFoundError, match="xml"):
        download.select_year_source(sources, 2025, "csv")


def test_select_year_source_reports_year_without_files():
    with pytest.raises(download.ResourceNotFoundError, match="nenhum arquivo do ano"):
        download.select_year_source([_source(2025, "INFLUD25-14-09-2026.csv")], 2019, "csv")


def test_select_year_source_rejects_undated_file():
    sources = [_source(2025, "INFLUD25.csv")]

    with pytest.raises(download.ResourceNotFoundError, match="data de"):
        download.select_year_source(sources, 2025, "csv")


def test_select_year_source_fails_when_date_is_ambiguous():
    sources = [
        _source(2025, "INFLUD25-14-09-2026.csv"),
        _source(2025, "INFLUD25-14-09-2026-outro.csv"),
    ]

    with pytest.raises(ValueError, match="mesma data"):
        download.select_year_source(sources, 2025, "csv")


def test_build_dataset_page_url_points_to_portal():
    assert download.build_dataset_page_url() == (
        "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026"
    )


def test_discover_source_prefers_explicit_url():
    url, filename, origin = download.discover_source(
        2025,
        "csv",
        explicit_url="https://exemplo/INFLUD25-14-09-2026.csv",
    )

    assert (origin, filename) == ("manual", "INFLUD25-14-09-2026.csv")
    assert url.endswith("INFLUD25-14-09-2026.csv")


def test_discover_source_reads_dataset_page(monkeypatch):
    monkeypatch.setattr(
        download,
        "fetch_dataset_sources",
        lambda **_: download.parse_dataset_page(DATASET_PAGE),
    )

    url, filename, origin = download.discover_source(2026, "csv")

    assert (origin, filename) == ("dataset_page", "INFLUD26-14-09-2026.csv")
    assert url.endswith("SRAG/2026/INFLUD26-14-09-2026.csv")


def test_discover_source_mentions_url_flag_when_page_has_no_year(monkeypatch):
    monkeypatch.setattr(
        download,
        "fetch_dataset_sources",
        lambda **_: download.parse_dataset_page(DATASET_PAGE),
    )

    with pytest.raises(download.ResourceNotFoundError, match="2018"):
        download.discover_source(2018, "csv")


def test_discover_source_propagates_network_failure(monkeypatch):
    def _fail(*_, **__):
        raise download.DownloadError("sem rede")

    monkeypatch.setattr(download, "_http_get", _fail)

    with pytest.raises(download.DownloadError, match="sem rede"):
        download.discover_source(2025, "csv")


def test_download_file_writes_atomically(tmp_path, monkeypatch):
    class _Response:
        """Fluxo que respeita o tamanho de chunk pedido, como um stream real."""

        def __init__(self, payload: bytes):
            self._buffer = payload

        def read(self, size):
            chunk, self._buffer = self._buffer[:size], self._buffer[size:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(download, "urlopen", lambda *_, **__: _Response(b"a;b;c\nd"))

    destination = tmp_path / "INFLUD25.csv"
    written, downloaded = download.download_file(
        "https://exemplo/INFLUD25.csv",
        destination,
        chunk_size=5,
    )

    assert (written, downloaded) == (7, True)
    assert destination.read_bytes() == b"a;b;c\nd"
    assert not (tmp_path / "INFLUD25.csv.part").exists()


def test_download_file_skips_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "INFLUD25.csv"
    destination.write_text("ja existe", encoding="utf-8")

    def _boom(*_, **__):
        raise AssertionError("nao deveria baixar")

    monkeypatch.setattr(download, "urlopen", _boom)

    written, downloaded = download.download_file(
        "https://exemplo/INFLUD25.csv",
        destination,
    )

    assert (written, downloaded) == (9, False)
    assert destination.read_text(encoding="utf-8") == "ja existe"


def test_download_file_removes_partial_file_on_failure(tmp_path, monkeypatch):
    def _boom(*_, **__):
        raise download.DownloadError("conexao caiu")

    monkeypatch.setattr(download, "urlopen", _boom)

    destination = tmp_path / "INFLUD25.csv"

    with pytest.raises(download.DownloadError, match="conexao caiu"):
        download.download_file("https://exemplo/INFLUD25.csv", destination)

    assert not destination.exists()
    assert not (tmp_path / "INFLUD25.csv.part").exists()


def test_write_source_manifest_records_provenance(tmp_path):
    destination = tmp_path / "INFLUD25.csv"
    destination.write_text("conteudo", encoding="utf-8")

    manifest = download.write_source_manifest(
        destination,
        "https://exemplo/INFLUD25.csv",
        "dataset_page",
        9,
    )

    assert manifest.name == "INFLUD25.fonte.json"
    payload = manifest.read_text(encoding="utf-8")
    assert '"origem": "dataset_page"' in payload
    assert '"bytes": 9' in payload


def test_copy_to_stable_name_normalizes_portal_filename(tmp_path):
    source = tmp_path / "INFLUD25-14-09-2026.csv"
    source.write_text("conteudo", encoding="utf-8")
    destination = tmp_path / "INFLUD25.csv"

    result = download.copy_to_stable_name(source, destination, force=False)

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "conteudo"
    assert source.exists()
