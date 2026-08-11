from srag_api.api.models import MetricResponse


def test_metric_response_accepts_null_value():
    model = MetricResponse(valor=None, numerador=0, denominador=0, ignorados=10)
    assert model.valor is None
    assert model.denominador == 0
