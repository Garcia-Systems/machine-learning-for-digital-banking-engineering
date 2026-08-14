from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import math
import pytest
from fastapi.testclient import TestClient

from harbor_ml.capstone_anomaly import (
    build_capstone_anomaly_timeline, create_anomaly_metadata,
    evaluate_detection_behavior, save_capstone_anomaly_artifact,
    select_anomaly_baseline, train_capstone_anomaly_detector,
)
from harbor_ml.capstone_dataset import load_capstone_sources
from harbor_ml.capstone_incident_classifier import (
    create_capstone_incident_metadata, evaluate_capstone_incident_classifier,
    load_capstone_classification_data, save_capstone_incident_artifact,
    split_capstone_incident_data, train_capstone_incident_classifier,
)
from harbor_ml.service.app import create_app
from harbor_ml.service.artifact_loader import (
    ArtifactPaths, ServiceConfig, load_configured_runtimes,
)
from harbor_ml.service.schemas import CapstoneTelemetryRequest
from harbor_ml.training import (
    save_model_artifact, save_training_metadata, train_integration_failure_model,
)

ROOT = Path(__file__).resolve().parents[1]
TELEMETRY = {
    "api_latency_ms": 1480, "error_rate": 0.047, "db_connections": 78,
    "queue_depth": 96, "vendor_latency_ms": 1320, "requests_per_minute": 760,
    "retry_count": 3,
}
INTEGRATION = {
    "vendor": "ClearVerify", "endpoint": "identity_verify",
    "recent_vendor_latency_ms": 940, "recent_vendor_error_rate": 0.031,
    "queue_depth": 42, "retry_count": 1, "request_size_bytes": 2400,
    "hour_of_day": 14,
}


@pytest.fixture(scope="module")
def runtimes(tmp_path_factory):
    root = tmp_path_factory.mktemp("capstone-service")
    timeline = build_capstone_anomaly_timeline(load_capstone_sources(ROOT / "data"))
    baseline = select_anomaly_baseline(timeline)
    anomaly_model = train_capstone_anomaly_detector(baseline)
    anomaly_metadata = create_anomaly_metadata(
        baseline, timeline, evaluate_detection_behavior(tuple()),
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc))
    anomaly = save_capstone_anomaly_artifact(anomaly_model, anomaly_metadata, root / "anomaly")

    data = load_capstone_classification_data(ROOT / "data/harbor_incident_classes.csv")
    split = split_capstone_incident_data(data)
    incident_model = train_capstone_incident_classifier(split)
    incident_metadata = create_capstone_incident_metadata(
        data, split, evaluate_capstone_incident_classifier(incident_model, split),
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc))
    incident = save_capstone_incident_artifact(incident_model, incident_metadata, root / "incident")

    integration_result = train_integration_failure_model(
        ROOT / "data/harbor_integration_requests.csv",
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc))
    integration = (save_model_artifact(integration_result.pipeline, root / "integration"),
                   save_training_metadata(integration_result.metadata, root / "integration"))
    config = ServiceConfig(*(ArtifactPaths(*paths) for paths in (anomaly, incident, integration)))
    return load_configured_runtimes(config)


def test_all_models_health_and_exact_prediction_contracts(runtimes):
    client = TestClient(create_app(runtimes))
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok" and health.json()["ready"] is True
    assert set(health.json()["models"]) == {
        "capstone_anomaly", "capstone_incident", "integration_failure"}
    assert all(model["loaded"] and model["version"] for model in health.json()["models"].values())

    anomaly = client.post("/api/v1/score/telemetry-anomaly", json=TELEMETRY).json()
    assert set(anomaly) == {"model", "model_version", "anomaly_score", "is_anomaly"}
    assert math.isfinite(anomaly["anomaly_score"])
    assert "anomaly_probability" not in anomaly

    incident = client.post("/api/v1/predict/incident", json=TELEMETRY).json()
    assert set(incident) == {"model", "model_version", "predicted_class", "probabilities",
        "top_probability", "second_probability", "probability_gap", "ambiguous"}
    assert sum(incident["probabilities"].values()) == pytest.approx(1)
    assert incident["predicted_class"] in incident["probabilities"]
    assert incident["probability_gap"] == pytest.approx(
        incident["top_probability"] - incident["second_probability"])
    assert incident["ambiguous"] is (incident["probability_gap"] < runtimes.incident.ambiguity_gap)

    integration = client.post("/api/v1/predict/integration-failure", json=INTEGRATION).json()
    assert set(integration) == {"model", "model_version", "failure_probability",
                               "threshold", "predicted_failure"}
    assert integration["predicted_failure"] is (
        integration["failure_probability"] >= integration["threshold"])


def test_partial_degradation_is_model_specific(runtimes):
    client = TestClient(create_app(replace(runtimes, incident=None)))
    health = client.get("/api/v1/health").json()
    assert health["status"] == "degraded" and health["ready"] is False
    assert health["models"]["capstone_incident"] == {
        "loaded": False, "version": None, "feature_contract_version": "capstone-telemetry-v1"}
    response = client.post("/api/v1/predict/incident", json=TELEMETRY)
    assert response.status_code == 503
    assert response.json() == {"detail": "Incident prediction model is unavailable."}
    assert client.post("/api/v1/score/telemetry-anomaly", json=TELEMETRY).status_code == 200
    assert client.post("/api/v1/predict/integration-failure", json=INTEGRATION).status_code == 200


def test_validation_determinism_and_handlers_do_not_train_or_load(runtimes):
    client = TestClient(create_app(runtimes))
    assert client.post("/api/v1/score/telemetry-anomaly",
                       json=TELEMETRY | {"error_rate": 4.2}).status_code == 422
    assert client.post("/api/v1/predict/incident",
                       json=TELEMETRY | {"member_id": "prohibited"}).status_code == 422
    with patch.object(runtimes.anomaly.model, "fit", side_effect=AssertionError("fit called")) as fit:
        first = client.post("/api/v1/score/telemetry-anomaly", json=TELEMETRY)
        second = client.post("/api/v1/score/telemetry-anomaly", json=TELEMETRY)
    assert first.json() == second.json()
    fit.assert_not_called()
    prohibited = {"member_id", "account_number", "email", "model_path", "training_data_path"}
    assert not prohibited & set(CapstoneTelemetryRequest.model_fields)
