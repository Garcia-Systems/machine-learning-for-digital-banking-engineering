from datetime import datetime, timezone
from pathlib import Path

import pytest

from harbor_ml.anomaly_detection import (
    build_anomaly_detector, build_anomaly_features, load_normal_telemetry,
    train_anomaly_detector,
)
from harbor_ml.capstone_incident import (
    CONFIRMING_EVIDENCE_TIME, FINAL_DIAGNOSIS, build_representative_request,
    dominant_trace_component, evaluate_incident_timeline, load_capstone_incident,
    load_capstone_traces,
)
from harbor_ml.incident_classifier import (
    build_incident_classifier, build_incident_features, build_incident_targets,
    load_incident_dataset, train_incident_classifier,
)
from harbor_ml.training import TrainingConfig, train_integration_failure_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def timeline():
    return load_capstone_incident(ROOT / "data/harbor_capstone_incident.csv")


@pytest.fixture(scope="module")
def evaluated(timeline):
    baseline = load_normal_telemetry(ROOT / "data/harbor_normal_telemetry.csv")
    anomaly = train_anomaly_detector(build_anomaly_detector(), build_anomaly_features(baseline))
    history = load_incident_dataset(ROOT / "data/harbor_incident_classes.csv")
    classifier = train_incident_classifier(
        build_incident_classifier(), build_incident_features(history), build_incident_targets(history))
    integration = train_integration_failure_model(
        ROOT / "data/harbor_integration_requests.csv", TrainingConfig(),
        now=lambda: datetime(2026, 8, 12, tzinfo=timezone.utc))
    return evaluate_incident_timeline(
        timeline, anomaly, classifier, integration.pipeline,
        integration_model_version=integration.metadata.model_version)


def test_timeline_is_chronological_healthy_then_escalates(timeline):
    assert len(timeline) == 20
    assert all(left.timestamp < right.timestamp for left, right in zip(timeline, timeline[1:]))
    assert timeline[0].api_latency_ms < 200
    assert timeline[0].error_rate < .01
    peak = max(timeline, key=lambda row: row.api_latency_ms)
    assert peak.vendor_latency_ms > timeline[0].vendor_latency_ms * 8
    assert peak.queue_depth > timeline[0].queue_depth * 10
    assert len({row.deployment_version for row in timeline}) == 1


def test_representative_request_uses_observed_request_time_features(timeline):
    request = build_representative_request(timeline[10])
    assert request.vendor == "ClearVerify"
    assert request.endpoint == "identity_verify"
    assert request.recent_vendor_latency_ms == timeline[10].vendor_latency_ms
    assert request.retry_count == timeline[10].retry_count
    assert request.hour_of_day == 10


def test_actual_outputs_probabilities_ambiguity_and_versions(evaluated):
    assert all(item.anomaly_result is not None for item in evaluated)
    available = [item for item in evaluated if item.incident_prediction]
    assert available
    for item in available:
        assert sum(item.incident_prediction.probabilities.values()) == pytest.approx(1.0)
    assert all(0 <= item.integration_failure_probability <= 1 for item in evaluated)
    assert evaluated[-8].incident_prediction is None
    assert evaluated[-8].classification == "unavailable"
    assert evaluated[-8].telemetry.api_latency_ms == 2075
    assert all(item.model_versions for item in evaluated)
    assert any(item.classification == "ambiguous" for item in available)


def test_diagnosis_waits_for_trace_evidence_and_matches_fixture(evaluated):
    assert all(not item.diagnosis_confirmed for item in evaluated
               if item.timestamp < CONFIRMING_EVIDENCE_TIME)
    confirmed = next(item for item in evaluated if item.timestamp == CONFIRMING_EVIDENCE_TIME)
    assert confirmed.diagnosis_confirmed
    assert any(note.category == "diagnosis" and note.text == FINAL_DIAGNOSIS
               for note in confirmed.notes)


def test_trace_fixture_computes_dominant_component():
    spans = load_capstone_traces(ROOT / "data/harbor_capstone_traces.csv")
    dominant = dominant_trace_component(spans, "capstone-request-001")
    assert dominant.component == "ClearVerify_call"
    assert dominant.duration_ms > sum(
        span.duration_ms for span in spans
        if span.request_id == dominant.request_id and span.component != dominant.component)


def test_fixtures_are_privacy_minimized():
    contents = "\n".join(path.read_text(encoding="utf-8") for path in (
        ROOT / "data/harbor_capstone_incident.csv", ROOT / "data/harbor_capstone_traces.csv"))
    for prohibited in ("member_id", "account_number", "email", "social_security", "name,"):
        assert prohibited not in contents.lower()
