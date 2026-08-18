import importlib

import pandas as pd


def _load_target_module():
    try:
        return importlib.import_module("srag_api.ml.target")
    except ModuleNotFoundError:
        return None


def test_target_maps_cure_and_srag_death():
    target_module = _load_target_module()

    assert target_module is not None, "srag_api.ml.target ainda nao foi implementado"

    df = pd.DataFrame(
        {"DESFECHO_NORMALIZADO": ["CURA", "OBITO_SRAG"]}
    )

    target = target_module.build_mortality_target(df)

    assert target.tolist() == [0, 1]


def test_other_cause_death_is_not_eligible():
    target_module = _load_target_module()

    assert target_module is not None, "srag_api.ml.target ainda nao foi implementado"

    df = pd.DataFrame(
        {"DESFECHO_NORMALIZADO": ["OBITO_OUTRAS_CAUSAS"]}
    )

    assert target_module.eligible_outcome_mask(df).tolist() == [False]


def test_missing_and_ignored_are_not_eligible():
    target_module = _load_target_module()

    assert target_module is not None, "srag_api.ml.target ainda nao foi implementado"

    df = pd.DataFrame(
        {"DESFECHO_NORMALIZADO": ["AUSENTE", "IGNORADO", None]}
    )

    assert target_module.eligible_outcome_mask(df).tolist() == [False, False, False]


def test_ineligible_outcome_becomes_nullable_target():
    target_module = _load_target_module()

    assert target_module is not None, "srag_api.ml.target ainda nao foi implementado"

    df = pd.DataFrame(
        {
            "DESFECHO_NORMALIZADO": [
                "CURA",
                "OBITO_SRAG",
                "OBITO_OUTRAS_CAUSAS",
                None,
            ]
        }
    )

    target = target_module.build_mortality_target(df)

    assert target.iloc[0] == 0
    assert target.iloc[1] == 1
    assert pd.isna(target.iloc[2])
    assert pd.isna(target.iloc[3])
