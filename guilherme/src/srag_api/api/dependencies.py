from pathlib import Path

from srag_api.data.repository import SragRepository
from srag_api.services.epidemiology import SragService


def get_parquet_root() -> Path:
    return Path("data/parquet")


def get_repository() -> SragRepository:
    return SragRepository(get_parquet_root())


def get_service() -> SragService:
    return SragService(get_repository())
