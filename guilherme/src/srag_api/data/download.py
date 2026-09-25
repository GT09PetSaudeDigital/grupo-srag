"""Descoberta e download dos bancos publicos do SIVEP-Gripe.

Os arquivos publicados pelo Ministerio da Saude tem nome com a data de
extracao do "banco vivo", por exemplo ``INFLUD25-14-09-2026.csv``. Essa data
muda a cada atualizacao semanal, portanto a URL nunca pode ser fixada no
codigo: ela precisa ser descoberta a cada execucao.

A descoberta tenta duas fontes, nesta ordem:

1. URL informada explicitamente pelo usuario;
2. pagina do conjunto de dados no Portal de Dados Abertos da Saude.

A API CKAN do portal nao e publica (responde 404) e o bucket S3 nao permite
listagem anonima (responde 403), mas os objetos e a pagina do conjunto sao
publicos. Por isso a pagina do dataset e a fonte usada.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

SIVEP_PACKAGE_ID = "srag-2019-a-2026"
DATASET_PAGE_URL = "https://dadosabertos.saude.gov.br/dataset/{package_id}"
S3_BUCKET_URL = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br"
S3_PREFIX = "SRAG/{year}/"

SUPPORTED_FORMATS: tuple[str, ...] = ("csv", "json", "parquet", "xml")

STAMP_PATTERN = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
S3_URL_PATTERN = re.compile(
    r"https://s3\.sa-east-1\.amazonaws\.com/ckan\.saude\.gov\.br"
    r"/SRAG/(?P<year>\d{4})/(?P<name>[^\"'\s<>]+?\.(?:csv|json|parquet|xml))"
)

DEFAULT_TIMEOUT = 60
DEFAULT_CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """Falha generica de download ou descoberta de recurso."""


class ResourceNotFoundError(DownloadError):
    """Nenhum recurso do ano/Formato foi encontrado no portal."""


def year_short_label(year: int) -> str:
    """Converte o ano no rotulo de dois digitos usado nos nomes dos arquivos."""
    label = str(year)[-2:]
    if not label.isdigit():
        raise ValueError(f"Ano invalido para rotulo SIVEP: {year}")
    return label


def filename_for_stamp(year: int, stamp: str, file_format: str) -> str:
    """Monta o nome padrao ``INFLUD{yy}-{stamp}.{ext}``."""
    normalize_format(file_format)
    return f"INFLUD{year_short_label(year)}-{stamp}.{normalize_format(file_format)}"


def normalize_format(file_format: str) -> str:
    """Normaliza o formato aceito para minusculas sem ponto."""
    normalized = str(file_format).strip().lower().lstrip(".")
    if not normalized:
        raise ValueError("Formato de arquivo vazio.")
    return normalized


def validate_format(file_format: str) -> str:
    """Normaliza o formato e garante que ele e um dos publicados pelo portal."""
    normalized = normalize_format(file_format)
    if normalized not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Formato nao suportado: {file_format}. Use um de: {supported}"
        )
    return normalized


def stamp_to_date(stamp: str) -> date | None:
    """Converte ``dd-mm-aaaa`` em ``date``; devolve ``None`` se nao casar."""
    match = STAMP_PATTERN.search(stamp)
    if not match:
        return None
    day, month, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


@dataclass(frozen=True)
class SourceFile:
    """Um arquivo publicado para um ano especifico do SIVEP-Gripe."""

    year: int
    name: str
    file_format: str
    url: str

    @property
    def stamp(self) -> date:
        """Data de extracao embutida no nome do arquivo."""
        parsed = stamp_to_date(self.name)
        if parsed is None:
            raise DownloadError(
                f"Nome de arquivo sem data de extracao: {self.name}"
            )
        return parsed


def parse_dataset_page(payload: str) -> list[SourceFile]:
    """Extrai os arquivos S3 citados na pagina do conjunto de dados.

    A pagina do Portal de Dados Abertos publica os enderecos completos dos
    arquivos do bucket, um por ano e formato, sem depender de API.
    """
    sources: list[SourceFile] = []
    seen: set[tuple[int, str]] = set()

    for match in S3_URL_PATTERN.finditer(html.unescape(payload)):
        name = match.group("name")
        year = int(match.group("year"))
        file_format = name.rsplit(".", 1)[-1].lower()
        key = (year, name)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceFile(
                year=year,
                name=name,
                file_format=file_format,
                url=match.group(0),
            )
        )

    if not sources:
        raise DownloadError(
            "A pagina do conjunto de dados nao revelou nenhum arquivo do SIVEP."
        )

    return sources


def select_year_source(
    sources: list[SourceFile],
    year: int,
    file_format: str,
) -> SourceFile:
    """Escolhe o arquivo mais recente do ano e formato pedidos.

    Levanta :class:ResourceNotFoundError quando nada casa e
    `ValueError` quando dois arquivos disputam a mesma data de extracao.
    """
    target = validate_format(file_format)

    candidates = [
        source
        for source in sources
        if source.year == year and source.file_format == target
    ]

    if not candidates:
        available = sorted(
            {
                source.file_format
                for source in sources
                if source.year == year
            }
        )
        detail = ", ".join(available) if available else "nenhum arquivo do ano"
        raise ResourceNotFoundError(
            f"Arquivo {year} em formato {target.upper()} nao encontrado. "
            f"Formatos publicados para o ano: {detail}"
        )

    dated = [source for source in candidates if stamp_to_date(source.name)]
    if not dated:
        raise ResourceNotFoundError(
            f"Nenhum arquivo de {year} em {target.upper()} tem data de "
            "extracao reconhecivel."
        )

    ordered = sorted(dated, key=lambda source: (source.stamp, source.name), reverse=True)

    if len(ordered) > 1 and ordered[0].stamp == ordered[1].stamp:
        raise ValueError(
            f"Mais de um arquivo com a mesma data em {year}: "
            f"{ordered[0].name}, {ordered[1].name}. Informe a URL com --url."
        )

    return ordered[0]


def build_s3_file_url(year: int, filename: str) -> str:
    """Monta a URL publica do arquivo no bucket do portal."""
    return f"{S3_BUCKET_URL}/{S3_PREFIX.format(year=year)}{quote(filename)}"


def build_dataset_page_url(package_id: str = SIVEP_PACKAGE_ID) -> str:
    """Monta a URL da pagina do conjunto de dados no portal."""
    return DATASET_PAGE_URL.format(package_id=package_id)


def _http_get(url: str, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": "grupo-srag/0.3"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        raise DownloadError(f"Falha HTTP {exc.code} ao acessar {url}") from exc
    except URLError as exc:
        raise DownloadError(f"Falha de rede ao acessar {url}: {exc.reason}") from exc


def fetch_dataset_sources(
    package_id: str = SIVEP_PACKAGE_ID,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[SourceFile]:
    """Le a pagina do conjunto de dados e devolve os arquivos publicados."""
    url = build_dataset_page_url(package_id)
    return parse_dataset_page(_http_get(url, timeout).decode("utf-8", errors="replace"))


def discover_source(
    year: int,
    file_format: str = "csv",
    *,
    explicit_url: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, str, str]:
    """Descobre a URL do arquivo do ano.

    Devolve `(url, filename, origin)`, onde `origin` identifica a fonte
    usada: `manual` ou `dataset_page`.
    """
    target = validate_format(file_format)

    if explicit_url:
        filename = explicit_url.rsplit("/", 1)[-1].split("?", 1)[0]
        if not filename:
            raise DownloadError(f"URL nao permite identificar o arquivo: {explicit_url}")
        return (explicit_url, filename, "manual")

    sources = fetch_dataset_sources(timeout=timeout)
    source = select_year_source(sources, year, target)
    return (source.url, source.name, "dataset_page")

def format_megabytes(byte_count: int) -> str:
    """Formata bytes em megabytes com uma casa decimal."""
    return f"{byte_count / (1024 * 1024):.1f} MB"


def write_source_manifest(
    destination: Path,
    url: str,
    origin: str,
    byte_count: int,
) -> Path:
    """Registra a origem do arquivo baixado ao lado do CSV."""
    manifest_path = destination.with_suffix(".fonte.json")
    payload = {
        "arquivo": destination.name,
        "url": url,
        "origem": discovery_origin(origin),
        "bytes": byte_count,
        "baixado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def discovery_origin(origin: str) -> str:
    """Normaliza a origem para o manifesto."""
    if origin not in {"manual", "dataset_page"}:
        raise ValueError(f"Origem desconhecida: {origin}")
    return origin


def download_file(
    url: str,
    destination: Path,
    timeout: int = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    force: bool = False,
) -> tuple[int, bool]:
    """Baixa ``url`` em ``destination`` de forma atomica.

    Devolve ``(bytes_escritos, baixou)``. Quando o arquivo ja existe e
    ``force`` e falso, nada e baixado. O download vai para um ``.part`` e so
    vira definitiva apos concluir, para que uma queda de conexao nao deixe
    um CSV truncado com cara de completo.
    """
    if destination.exists() and not force:
        return (destination.stat().st_size, False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    request = Request(url, headers={"User-Agent": "grupo-srag/0.3"})
    written = 0

    try:
        with urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                print(
                    f"    {format_megabytes(written)}",
                    end="\r",
                    flush=True,
                )
    except HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Falha HTTP {exc.code} ao baixar {url}") from exc
    except URLError as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Falha de rede ao baixar {url}: {exc.reason}") from exc
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    os.replace(partial, destination)
    print(f"    {format_megabytes(written)} pronto")

    return (written, True)


def copy_to_stable_name(source: Path, destination: Path, force: bool) -> Path:
    """Normaliza o nome do CSV para ``INFLUD{yy}.csv``.

    O ``ingest_all`` exige exatamente um CSV por ano, e o portal nomeia os
    arquivos com a data de extracao. A copia evita renomear o arquivo original.
    """
    if destination.exists() and not force:
        return destination

    shutil.copyfile(source, destination)
    return destination
