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


def test_admission_features_use_age_in_years():
    """O contrato do ML usa idade em anos, nao o numero cru.

    No SIVEP, ``NU_IDADE_N`` so tem significado junto com ``TP_IDADE``: o
    mesmo 6 pode ser 6 dias, 6 meses ou 6 anos. Em 2025, 27,49% dos
    elegiveis estavam em dias ou meses. A ingestao ja calcula
    ``IDADE_ANOS``, que e a unica coluna com unidade.
    """
    features = _load_features_module()
    assert features is not None

    assert "IDADE_ANOS" in features.DEMOGRAPHIC_FEATURES
    assert "IDADE_ANOS" in features.ADMISSION_FEATURES


def test_admission_features_drop_the_raw_age_and_its_unit():
    features = _load_features_module()
    assert features is not None

    assert "NU_IDADE_N" not in features.ADMISSION_FEATURES
    assert "TP_IDADE" not in features.ADMISSION_FEATURES
    assert "NU_IDADE_N" not in features.DEMOGRAPHIC_FEATURES


def test_age_in_years_stays_disjoint_from_leakage():
    features = _load_features_module()
    assert features is not None

    assert "IDADE_ANOS" not in features.LEAKAGE_FEATURES
    features.validate_feature_registry()
