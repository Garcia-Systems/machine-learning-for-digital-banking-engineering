from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harbor_ml.dashboard import (
    INVESTIGATION_GUIDANCE, MLAvailability, build_teaching_service,
    calculate_severity, classify_availability, is_ambiguous,
)
from harbor_ml.dashboard.app import create_dashboard_app
from harbor_ml.dashboard.models import TelemetrySnapshot
from harbor_ml.dashboard.service import teaching_scenarios

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2025, 1, 15, 14, 35, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def service():
    return build_teaching_service(ROOT)


@pytest.fixture(scope="module")
def scenarios(service):
    result = {}
    for name, telemetry, request in teaching_scenarios(NOW):
        result[name] = service.assemble(
            telemetry, request, now=NOW,
            prediction_timestamp=NOW - timedelta(minutes=20) if name == "stale_prediction" else NOW,
            ml_available=name != "ml_unavailable", remember=False,
        )
    return result


@pytest.mark.parametrize(
    ("latency", "error_rate", "expected"),
    [(200, .001, "normal"), (800, .001, "warning"), (200, .02, "warning"), (2000, 0, "critical"), (200, .05, "critical")],
)
def test_severity_uses_deterministic_telemetry_thresholds(latency, error_rate, expected):
    telemetry = TelemetrySnapshot(NOW, latency, error_rate, 20, 10, 100, 500)
    assert calculate_severity(telemetry) == expected


def test_snapshot_construction_contains_observations_and_actual_model_outputs(scenarios):
    snapshot = scenarios["vendor_degradation"]
    assert snapshot.telemetry.vendor_latency_ms == 1320
    assert snapshot.telemetry_anomaly is True
    assert snapshot.anomaly_score is not None
    assert snapshot.predicted_incident_class in snapshot.incident_probabilities
    assert sum(snapshot.incident_probabilities.values()) == pytest.approx(1)
    assert snapshot.integration_failure_prediction is (
        snapshot.integration_failure_probability >= snapshot.integration_failure_threshold
    )
    assert snapshot.model_name == "harbor-integration-failure"
    assert snapshot.model_version


def test_probabilities_are_ordered_descending_with_stable_ties(scenarios):
    ordered = scenarios["vendor_degradation"].ordered_incident_probabilities
    assert ordered == tuple(sorted(ordered, key=lambda item: (-item[1], item[0])))


def test_healthy_vendor_and_database_scenarios(scenarios):
    assert scenarios["healthy"].severity == "normal"
    assert scenarios["healthy"].telemetry_anomaly is False
    assert scenarios["healthy"].predicted_incident_class == "normal"
    assert scenarios["healthy"].integration_failure_probability < .50
    assert scenarios["vendor_degradation"].predicted_incident_class == "vendor_degradation"
    assert scenarios["database_pressure"].predicted_incident_class == "database_pressure"


def test_investigation_mapping_is_deterministic(scenarios):
    snapshot = scenarios["vendor_degradation"]
    assert snapshot.investigation_guidance == INVESTIGATION_GUIDANCE["vendor_degradation"]
    assert "Inspect vendor latency and timeout telemetry." in snapshot.investigation_guidance


def test_ambiguity_uses_configurable_top_two_gap():
    assert is_ambiguous({"vendor": .27, "database": .24, "normal": .10})
    assert not is_ambiguous({"vendor": .70, "database": .20}, gap=.10)


def test_availability_current_stale_boundary_and_missing():
    assert classify_availability(NOW - timedelta(minutes=5), NOW, timedelta(minutes=5))[0] is MLAvailability.AVAILABLE
    assert classify_availability(NOW - timedelta(minutes=5, seconds=1), NOW, timedelta(minutes=5))[0] is MLAvailability.STALE
    assert classify_availability(None, NOW, timedelta(minutes=5)) == (MLAvailability.UNAVAILABLE, None)


def test_unavailable_keeps_telemetry_and_does_not_invent_zero(scenarios):
    snapshot = scenarios["ml_unavailable"]
    assert snapshot.telemetry.api_latency_ms == 640
    assert snapshot.ml_status is MLAvailability.UNAVAILABLE
    assert snapshot.integration_failure_probability is None
    assert snapshot.anomaly_score is None
    assert snapshot.model_version is None


def test_stale_prediction_is_preserved_but_explicitly_labeled(scenarios):
    snapshot = scenarios["stale_prediction"]
    assert snapshot.ml_status is MLAvailability.STALE
    assert snapshot.prediction_age_seconds == 1200
    assert snapshot.integration_failure_probability is not None


@pytest.mark.parametrize("name", ["healthy", "vendor_degradation", "database_pressure", "ml_unavailable", "stale_prediction"])
def test_dashboard_route_renders_semantic_sections_without_certainty_language(service, scenarios, name):
    service.history = list(scenarios.values())
    response = TestClient(create_dashboard_app(service, scenarios[name])).get("/dashboard")
    assert response.status_code == 200
    text = response.text
    for label in ("Observed — system health", "Model suggests — ML signals", "Investigate — deterministic guidance", "Recent prediction history"):
        assert label in text
    assert "API latency" in text and "Anomaly score" in text
    assert "Model-assigned incident-class probabilities" in text or name == "ml_unavailable"
    assert "ROOT CAUSE" not in text.upper()
    assert "confirmed diagnosis" in text


def test_html_displays_failure_probability_version_and_anomaly_explanation(service, scenarios):
    response = TestClient(create_dashboard_app(service, scenarios["vendor_degradation"])).get("/dashboard")
    text = response.text
    assert "Failure probability" in text
    assert scenarios["vendor_degradation"].model_version in text
    assert "unusualness score; not a probability" in text
    assert "not confirmed causes" in text


def test_html_handles_none_ml_fields_and_labels_unavailable(service, scenarios):
    response = TestClient(create_dashboard_app(service, scenarios["ml_unavailable"])).get("/dashboard")
    assert response.status_code == 200
    assert "ML prediction unavailable" in response.text
    assert "missing values are not zero risk" in response.text


def test_html_labels_stale_output(service, scenarios):
    text = TestClient(create_dashboard_app(service, scenarios["stale_prediction"])).get("/dashboard").text
    assert "Prediction status: <span class=\"status\">STALE</span>" in text
    assert "must not be treated as current evidence" in text
