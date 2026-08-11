from pathlib import Path

from srag_api.api.dependencies import get_parquet_root


def test_default_parquet_root_points_to_local_data():
    assert get_parquet_root() == Path("data/parquet")
