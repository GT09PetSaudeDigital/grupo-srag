import importlib

import numpy as np
import pandas as pd
import pytest


def _load_evaluation_module():
    try:
        return importlib.import_module("srag_api.ml.evaluation")
    except ModuleNotFoundError:
        return None


def _require_module():
    module = _load_evaluation_module()
    assert module is not None, "srag_api.ml.evaluation ainda nao foi implementado"
    return module


# --------------------------------------------------------------------------
# Brier score
# --------------------------------------------------------------------------


def test_brier_score_is_zero_for_perfect_probabilities():
    module = _require_module()

    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.0, 1.0, 0.0, 1.0])

    assert module.brier_score(y_true, probabilities) == pytest.approx(0.0)


def test_brier_score_is_one_for_maximally_wrong_confident_predictions():
    module = _require_module()

    y_true = pd.Series([0, 1])
    probabilities = np.array([1.0, 0.0])

    assert module.brier_score(y_true, probabilities) == pytest.approx(1.0)


def test_brier_score_uses_mean_squared_error():
    module = _require_module()

    y_true = pd.Series([0, 1])
    probabilities = np.array([0.2, 0.6])

    esperado = ((0.2 - 0.0) ** 2 + (0.6 - 1.0) ** 2) / 2

    assert module.brier_score(y_true, probabilities) == pytest.approx(esperado)


# --------------------------------------------------------------------------
# Curva de confiabilidade
# --------------------------------------------------------------------------


def test_calibration_metrics_group_predictions_into_declared_bins():
    module = _require_module()

    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.85, 0.95])

    result = module.calibration_metrics(y_true, probabilities, n_bins=10)

    assert len(result.bins) == 4
    assert result.bins[0].count == 1
    assert result.bins[0].mean_predicted == pytest.approx(0.05)
    assert result.bins[0].observed_frequency == pytest.approx(0.0)
    assert result.bins[-1].mean_predicted == pytest.approx(0.95)
    assert result.bins[-1].observed_frequency == pytest.approx(1.0)


def test_calibration_metrics_skip_empty_bins():
    module = _require_module()

    result = module.calibration_metrics(
        pd.Series([0, 1]),
        np.array([0.01, 0.99]),
        n_bins=10,
    )

    assert [b.count for b in result.bins] == [1, 1]
    assert all(b.count > 0 for b in result.bins)


def test_calibration_metrics_expose_brier_score():
    module = _require_module()

    result = module.calibration_metrics(
        pd.Series([0, 1]),
        np.array([0.2, 0.8]),
        n_bins=10,
    )

    # (0.2 - 0)^2 = 0.04 e (0.8 - 1)^2 = 0.04, media 0.04.
    assert result.brier_score == pytest.approx(0.04)


def test_calibration_metrics_reject_n_bins_below_one():
    module = _require_module()

    with pytest.raises(ValueError, match="n_bins"):
        module.calibration_metrics(pd.Series([0, 1]), np.array([0.2, 0.8]), n_bins=0)


# --------------------------------------------------------------------------
# Volume de alertas e cobertura da fila
# --------------------------------------------------------------------------


def test_alert_volume_counts_the_confusion_matrix_by_hand():
    module = _require_module()

    y_true = pd.Series([0, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.9, 0.3, 0.8])

    result = module.alert_volume(y_true, probabilities, threshold=0.5)

    assert result.total == 5
    assert result.alerts == 2
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.true_negatives == 2
    assert result.sensitivity == pytest.approx(0.5)
    assert result.specificity == pytest.approx(2 / 3)
    assert result.ppv == pytest.approx(0.5)
    assert result.npv == pytest.approx(2 / 3)


def test_alert_volume_reports_alerts_per_thousand_admissions():
    module = _require_module()

    y_true = pd.Series([0] * 999 + [1])
    probabilities = np.array([0.1] * 1000)

    result = module.alert_volume(y_true, probabilities, threshold=0.5)

    assert result.alerts == 0
    assert result.alerts_per_1000 == pytest.approx(0.0)


def test_alert_volume_converts_to_a_rate_even_with_rare_events():
    module = _require_module()

    y_true = pd.Series([0] * 999 + [1])
    probabilities = np.array([0.9] + [0.1] * 999)

    result = module.alert_volume(y_true, probabilities, threshold=0.5)

    assert result.alerts == 1
    assert result.alerts_per_1000 == pytest.approx(1.0)


def test_queue_coverage_reports_how_many_deaths_reach_the_top_of_the_queue():
    module = _require_module()

    y_true = pd.Series([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.5, 0.4, 0.3, 0.8, 0.2, 0.1, 0.05, 0.02, 0.01])

    result = module.queue_coverage(y_true, probabilities, fractions=(0.1, 0.2))

    top_one = result[0]
    assert top_one.queue_size == 1
    assert top_one.alerts == 1
    assert top_one.covered_positives == 1
    assert top_one.total_positives == 2
    assert top_one.coverage == pytest.approx(0.5)
    assert top_one.precision == pytest.approx(1.0)

    top_two = result[1]
    assert top_two.queue_size == 2
    assert top_two.covered_positives == 2
    assert top_two.coverage == pytest.approx(1.0)
    assert top_two.precision == pytest.approx(1.0)


def test_queue_coverage_orders_by_probability_not_by_input_position():
    module = _require_module()

    y_true = pd.Series([0, 0, 0, 1])
    probabilities = np.array([0.99, 0.98, 0.97, 0.10])

    result = module.queue_coverage(y_true, probabilities, fractions=(0.25,))

    point = result[0]
    assert point.alerts == 1
    assert point.covered_positives == 0
    assert point.coverage == pytest.approx(0.0)
    assert point.precision == pytest.approx(0.0)


def test_queue_coverage_rejects_fractions_outside_zero_one():
    module = _require_module()

    with pytest.raises(ValueError, match="fracao"):
        module.queue_coverage(
            pd.Series([0, 1]),
            np.array([0.1, 0.9]),
            fractions=(0.0, 1.5),
        )


# --------------------------------------------------------------------------
# Incerteza por bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_interval_contains_the_point_estimate_most_of_the_time():
    module = _require_module()

    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.integers(0, 2, size=400))
    probabilities = np.clip(rng.normal(y_true.to_numpy(), 0.25), 0, 1)

    point = module.average_precision(y_true, probabilities)
    interval = module.bootstrap_confidence_interval(
        y_true,
        probabilities,
        metric="auc_pr",
        n_resamples=120,
        random_state=42,
    )

    assert interval.level == pytest.approx(0.95)
    assert interval.lower <= interval.upper
    assert interval.lower <= point <= interval.upper


def test_bootstrap_interval_reports_resamples_it_could_not_use():
    module = _require_module()

    y_true = pd.Series([0] * 30 + [1] * 2)
    probabilities = np.linspace(0.05, 0.95, 32)

    interval = module.bootstrap_confidence_interval(
        y_true,
        probabilities,
        metric="roc_auc",
        n_resamples=60,
        random_state=7,
    )

    assert interval.total_resamples == 60
    assert 0 < interval.valid_resamples <= 60


def test_bootstrap_interval_is_deterministic_for_a_fixed_seed():
    module = _require_module()

    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.integers(0, 2, size=200))
    probabilities = np.clip(rng.normal(y_true.to_numpy(), 0.3), 0, 1)

    primeiro = module.bootstrap_confidence_interval(
        y_true, probabilities, metric="auc_pr", n_resamples=50, random_state=11
    )
    segundo = module.bootstrap_confidence_interval(
        y_true, probabilities, metric="auc_pr", n_resamples=50, random_state=11
    )

    assert (primeiro.lower, primeiro.upper) == (segundo.lower, segundo.upper)


def test_bootstrap_interval_rejects_unknown_metric():
    module = _require_module()

    with pytest.raises(ValueError, match="Metrica desconhecida"):
        module.bootstrap_confidence_interval(
            pd.Series([0, 1]),
            np.array([0.1, 0.9]),
            metric="inexistente",
        )


# --------------------------------------------------------------------------
# Subgrupos
# --------------------------------------------------------------------------


def test_subgroup_metrics_split_by_a_metadata_column():
    module = _require_module()

    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    metadata = pd.DataFrame({"SG_UF": ["PR", "PR", "MT", "MT", "PR", "MT"]})

    result = module.subgroup_metrics(
        metadata,
        y_true,
        probabilities,
        threshold=0.5,
        by="SG_UF",
    )

    por_uf = {item.key: item for item in result}

    assert set(por_uf) == {"PR", "MT"}
    assert por_uf["PR"].total == 3
    assert por_uf["PR"].positives == 1
    assert por_uf["MT"].total == 3
    assert por_uf["MT"].positives == 2
    assert por_uf["PR"].observed_rate == pytest.approx(1 / 3)
    assert por_uf["MT"].observed_rate == pytest.approx(2 / 3)


def test_subgroup_metrics_leave_auc_undefined_for_single_class_group():
    module = _require_module()

    # 2024 recebe as duas classes e tem AUC; 2025 recebe so sobreviventes,
    # nao ha o que ordenar, mas Brier e volume continuam definidos.
    y_true = pd.Series([0, 1, 0, 0])
    probabilities = np.array([0.1, 0.9, 0.2, 0.3])
    metadata = pd.DataFrame({"ano": [2024, 2024, 2025, 2025]})

    result = module.subgroup_metrics(
        metadata,
        y_true,
        probabilities,
        threshold=0.5,
        by="ano",
    )

    por_ano = {item.key: item for item in result}

    assert por_ano["2025"].roc_auc is None
    assert por_ano["2025"].auc_pr is None
    assert por_ano["2025"].brier_score is not None
    assert por_ano["2025"].total == 2
    assert por_ano["2024"].roc_auc is not None
    assert por_ano["2024"].auc_pr is not None


def test_subgroup_metrics_report_missing_values_as_a_group():
    module = _require_module()

    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])
    metadata = pd.DataFrame({"regiao": ["Sul", None, "Sul", None]})

    result = module.subgroup_metrics(
        metadata,
        y_true,
        probabilities,
        threshold=0.5,
        by="regiao",
    )

    chaves = {item.key for item in result}

    assert "Sul" in chaves
    assert any("AUSENTE" in chave for chave in chaves)


def test_subgroup_metrics_reject_missing_column():
    module = _require_module()

    with pytest.raises(ValueError, match="Coluna de metadados ausente"):
        module.subgroup_metrics(
            pd.DataFrame({"uf": ["PR"]}),
            pd.Series([0, 1]),
            np.array([0.1, 0.9]),
            threshold=0.5,
            by="SG_UF",
        )


# --------------------------------------------------------------------------
# Relatorio agregado
# --------------------------------------------------------------------------


def test_evaluate_partition_aggregates_everything_the_audit_found_missing():
    module = _require_module()

    rng = np.random.default_rng(3)
    y_true = pd.Series(rng.integers(0, 2, size=300))
    probabilities = np.clip(rng.normal(y_true.to_numpy(), 0.3), 0, 1)
    metadata = pd.DataFrame({"SG_UF": ["PR"] * 150 + ["MT"] * 150})

    report = module.evaluate_partition(
        y_true,
        probabilities,
        threshold=0.6,
        partition="teste",
        metadata=metadata,
        by=("SG_UF",),
        queue_fractions=(0.1, 0.25),
        n_resamples=40,
        random_state=5,
    )

    assert report.partition == "teste"
    assert report.total == 300
    assert report.positives == int(y_true.sum())
    assert report.prevalence == pytest.approx(y_true.mean())
    assert 0.0 <= report.auc_pr <= 1.0
    assert 0.0 <= report.roc_auc <= 1.0
    assert report.calibration.brier_score >= 0.0
    assert report.volume.threshold == pytest.approx(0.6)
    assert len(report.queue) == 2
    assert report.auc_pr_interval is not None
    assert report.roc_auc_interval is not None
    assert len(report.subgroups) == 2


def test_evaluate_partition_without_metadata_still_reports_the_core_metrics():
    module = _require_module()

    rng = np.random.default_rng(4)
    y_true = pd.Series(rng.integers(0, 2, size=120))
    probabilities = np.clip(rng.normal(y_true.to_numpy(), 0.3), 0, 1)

    report = module.evaluate_partition(y_true, probabilities, threshold=0.5)

    assert report.total == 120
    assert report.subgroups == ()
    assert report.queue


def test_evaluate_partition_rejects_single_class_partition():
    module = _require_module()

    with pytest.raises(ValueError, match="duas classes"):
        module.evaluate_partition(
            pd.Series([0, 0, 0]),
            np.array([0.1, 0.2, 0.3]),
            threshold=0.5,
        )


def test_evaluate_partition_rejects_probabilities_outside_zero_one():
    module = _require_module()

    with pytest.raises(ValueError, match="probabilidades"):
        module.evaluate_partition(
            pd.Series([0, 1]),
            np.array([0.1, 1.4]),
            threshold=0.5,
        )
