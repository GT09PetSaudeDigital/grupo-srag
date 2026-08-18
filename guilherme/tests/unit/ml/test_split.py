import importlib

import pandas as pd


def _load_split_module():
    try:
        return importlib.import_module("srag_api.ml.split")
    except ModuleNotFoundError:
        return None


def test_temporal_split_reserves_2026_for_test():
    split_module = _load_split_module()

    assert split_module is not None, "srag_api.ml.split ainda nao foi implementado"

    years = pd.Series([2019, 2020, 2024, 2025, 2026])

    split = split_module.temporal_split(years)

    assert years.iloc[split.test_idx].tolist() == [2026]
    assert 2026 not in years.iloc[split.train_idx].tolist()


def test_validation_is_more_recent_than_training():
    split_module = _load_split_module()

    assert split_module is not None, "srag_api.ml.split ainda nao foi implementado"

    years = pd.Series([2021, 2023, 2024, 2025, 2026])

    split = split_module.temporal_split(years)

    assert max(years.iloc[split.train_idx]) < min(years.iloc[split.validation_idx])


def test_split_has_no_overlapping_indices():
    split_module = _load_split_module()

    assert split_module is not None, "srag_api.ml.split ainda nao foi implementado"

    years = pd.Series([2023, 2024, 2025, 2026])

    split = split_module.temporal_split(years)

    train = set(split.train_idx)
    validation = set(split.validation_idx)
    test = set(split.test_idx)

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_split_rejects_missing_years():
    split_module = _load_split_module()

    assert split_module is not None, "srag_api.ml.split ainda nao foi implementado"

    years = pd.Series([2024, None, 2025, 2026])

    try:
        split_module.temporal_split(years)
    except ValueError as exc:
        assert "ano" in str(exc).lower()
    else:
        raise AssertionError("Era esperado ValueError para ano ausente")


def test_split_requires_validation_and_test_partitions():
    split_module = _load_split_module()

    assert split_module is not None, "srag_api.ml.split ainda nao foi implementado"

    years = pd.Series([2021, 2022, 2023, 2024])

    try:
        split_module.temporal_split(years)
    except ValueError as exc:
        message = str(exc).lower()
        assert "2025" in message or "2026" in message
    else:
        raise AssertionError("Era esperado ValueError quando validacao/teste nao existem")
