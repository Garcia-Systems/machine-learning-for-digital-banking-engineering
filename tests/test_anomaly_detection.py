from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from harbor_ml import (
    ANOMALY_CONTAMINATION,
    ANOMALY_FEATURE_NAMES,
    AnomalyResult,
    build_anomaly_detector,
    build_anomaly_features,
    load_anomaly_scenarios,
    load_normal_telemetry,
    observation_features,
    score_observation,
    train_anomaly_detector,
)

DATA = Path(__file__).parents[1] / "data"
NORMAL = DATA / "harbor_normal_telemetry.csv"
SCENARIOS = DATA / "harbor_anomaly_scenarios.csv"


@pytest.fixture
def baseline():
    return load_normal_telemetry(NORMAL)


@pytest.fixture
def fitted(baseline):
    return train_anomaly_detector(build_anomaly_detector(), build_anomaly_features(baseline))


def test_normal_fixture_loads_expected_complete_observations(baseline) -> None:
    assert len(baseline) == 200
    assert baseline[0].timestamp.isoformat() == "2026-01-05T08:00:00+00:00"
    assert not np.isnan(build_anomaly_features(baseline)).any()
    assert all(getattr(row, name) >= 0 for row in baseline for name in ANOMALY_FEATURE_NAMES)


def test_feature_matrix_has_declared_order_and_shape(baseline) -> None:
    features = build_anomaly_features(baseline)
    assert ANOMALY_FEATURE_NAMES == (
        "api_latency_ms", "error_rate", "db_connections", "queue_depth",
        "vendor_latency_ms", "requests_per_minute",
    )
    assert features.shape == (200, 6)
    assert features[0].tolist() == [188.0, 0.0045, 32.0, 10.0, 238.0, 446.0]


def test_model_creation_and_fitting() -> None:
    model = build_anomaly_detector()
    assert isinstance(model, IsolationForest)
    assert model.contamination == ANOMALY_CONTAMINATION
    assert not hasattr(model, "estimators_")
    fitted = train_anomaly_detector(model, build_anomaly_features(load_normal_telemetry(NORMAL)))
    assert len(fitted.estimators_) > 0


def test_training_and_scores_are_deterministic(baseline) -> None:
    features = build_anomaly_features(baseline)
    first = train_anomaly_detector(build_anomaly_detector(), features)
    second = train_anomaly_detector(build_anomaly_detector(), features)
    assert np.array_equal(first.predict(features), second.predict(features))
    assert np.allclose(first.decision_function(features), second.decision_function(features))


def test_scenario_fixture_and_key_classifications(fitted) -> None:
    scenarios = load_anomaly_scenarios(SCENARIOS)
    assert [scenario.name for scenario in scenarios] == [
        "normal_morning", "normal_busy_period", "vendor_slowdown", "queue_pressure",
        "broad_system_pressure", "subtle_combination",
    ]
    results = {
        scenario.name: score_observation(fitted, observation_features(scenario.observation))
        for scenario in scenarios
    }
    assert results["normal_morning"].is_anomaly is False
    for name in ("vendor_slowdown", "queue_pressure", "broad_system_pressure", "subtle_combination"):
        assert results[name].is_anomaly is True
        assert np.isfinite(results[name].score)
    assert isinstance(results["vendor_slowdown"], AnomalyResult)
    assert isinstance(results["vendor_slowdown"].score, float)


def test_scores_one_new_observation(fitted) -> None:
    result = score_observation(fitted, {
        "api_latency_ms": 190, "error_rate": 0.004, "db_connections": 31,
        "queue_depth": 11, "vendor_latency_ms": 250, "requests_per_minute": 440,
    })
    assert isinstance(result, AnomalyResult)
    assert np.isfinite(result.score)
    assert isinstance(result.is_anomaly, bool)


@pytest.mark.parametrize("observation, message", [
    ({"api_latency_ms": 190}, "missing required telemetry features"),
    ({"api_latency_ms": 190, "error_rate": .004, "db_connections": 31, "queue_depth": 11,
      "vendor_latency_ms": 250, "requests_per_minute": 440, "label": 0}, "unexpected telemetry features"),
    ({"api_latency_ms": -1, "error_rate": .004, "db_connections": 31, "queue_depth": 11,
      "vendor_latency_ms": 250, "requests_per_minute": 440}, "api_latency_ms"),
    ({"api_latency_ms": 190, "error_rate": float("nan"), "db_connections": 31, "queue_depth": 11,
      "vendor_latency_ms": 250, "requests_per_minute": 440}, "error_rate"),
])
def test_rejects_malformed_inference_features(fitted, observation, message) -> None:
    with pytest.raises(ValueError, match=message):
        score_observation(fitted, observation)


def test_rejects_malformed_training_matrix() -> None:
    with pytest.raises(ValueError, match="6 columns"):
        train_anomaly_detector(build_anomaly_detector(), np.ones((2, 5)))
    with pytest.raises(ValueError, match="finite observations"):
        train_anomaly_detector(build_anomaly_detector(), np.full((2, 6), np.nan))
