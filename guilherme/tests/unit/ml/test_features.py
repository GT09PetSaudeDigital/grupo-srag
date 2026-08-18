import importlib


def _load_features_module():
    try:
        return importlib.import_module("srag_api.ml.features")
    except ModuleNotFoundError:
        return None


def test_admission_features_do_not_contain_known_leakage():
    features = _load_features_module()

    assert features is not None, "srag_api.ml.features ainda nao foi implementado"
    assert set(features.ADMISSION_FEATURES).isdisjoint(features.LEAKAGE_FEATURES)


def test_extended_comorbidities_are_available_as_candidates():
    features = _load_features_module()

    assert features is not None, "srag_api.ml.features ainda nao foi implementado"
    expected = {
        "CARDIOPATI",
        "DIABETES",
        "PNEUMOPATI",
        "RENAL",
        "HEPATICA",
        "IMUNODEPRE",
        "OBESIDADE",
        "OUT_MORBI",
    }
    assert expected.issubset(set(features.COMORBIDITY_FEATURES))


def test_validate_feature_registry_accepts_current_registry():
    features = _load_features_module()

    assert features is not None, "srag_api.ml.features ainda nao foi implementado"
    features.validate_feature_registry()
