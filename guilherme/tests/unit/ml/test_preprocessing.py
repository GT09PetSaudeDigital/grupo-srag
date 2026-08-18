import importlib

import pandas as pd


def _load_preprocessing_module():
    try:
        return importlib.import_module("srag_api.ml.preprocessing")
    except ModuleNotFoundError:
        return None


def test_numeric_imputation_is_learned_from_train_only():
    preprocessing = _load_preprocessing_module()

    assert preprocessing is not None, "srag_api.ml.preprocessing ainda nao foi implementado"

    X_train = pd.DataFrame({"NU_IDADE_N": [20.0, 40.0, None]})
    X_validation = pd.DataFrame({"NU_IDADE_N": [1000.0]})
    X_test = pd.DataFrame({"NU_IDADE_N": [2000.0]})

    preprocessor = preprocessing.build_preprocessor(
        numeric_features=["NU_IDADE_N"],
        categorical_features=[],
    )
    fitted = preprocessing.fit_preprocessor_on_train(preprocessor, X_train)

    imputer = fitted.named_transformers_["numeric"].named_steps["imputer"]

    assert imputer.statistics_[0] == 30.0

    train, validation, test = preprocessing.transform_partitions(
        fitted,
        X_train,
        X_validation,
        X_test,
    )

    assert train.shape[0] == 3
    assert validation.shape[0] == 1
    assert test.shape[0] == 1


def test_unknown_category_in_future_partition_does_not_refit_encoder():
    preprocessing = _load_preprocessing_module()

    assert preprocessing is not None, "srag_api.ml.preprocessing ainda nao foi implementado"

    X_train = pd.DataFrame({"CS_SEXO": ["F", "M"]})
    X_validation = pd.DataFrame({"CS_SEXO": ["I"]})
    X_test = pd.DataFrame({"CS_SEXO": ["X"]})

    preprocessor = preprocessing.build_preprocessor(
        numeric_features=[],
        categorical_features=["CS_SEXO"],
    )
    fitted = preprocessing.fit_preprocessor_on_train(preprocessor, X_train)

    encoder = fitted.named_transformers_["categorical"].named_steps["encoder"]
    assert encoder.categories_[0].tolist() == ["F", "M"]

    train, validation, test = preprocessing.transform_partitions(
        fitted,
        X_train,
        X_validation,
        X_test,
    )

    assert train.shape[0] == 2
    assert validation.shape[0] == 1
    assert test.shape[0] == 1


def test_build_preprocessor_accepts_numeric_only_features():
    preprocessing = _load_preprocessing_module()

    assert preprocessing is not None, "srag_api.ml.preprocessing ainda nao foi implementado"

    preprocessor = preprocessing.build_preprocessor(
        numeric_features=["NU_IDADE_N"],
        categorical_features=[],
    )

    assert preprocessor is not None


def test_build_preprocessor_accepts_categorical_only_features():
    preprocessing = _load_preprocessing_module()

    assert preprocessing is not None, "srag_api.ml.preprocessing ainda nao foi implementado"

    preprocessor = preprocessing.build_preprocessor(
        numeric_features=[],
        categorical_features=["CS_SEXO"],
    )

    assert preprocessor is not None
