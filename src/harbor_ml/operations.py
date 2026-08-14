"""Chapter 33 orchestration for operating Harbor's existing teaching system."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from fastapi import HTTPException

from .capstone_anomaly import (build_capstone_anomaly_timeline, create_anomaly_metadata,
    evaluate_detection_behavior, save_capstone_anomaly_artifact, score_capstone_timeline,
    select_anomaly_baseline, train_capstone_anomaly_detector)
from .capstone_dataset import build_capstone_dataset, load_capstone_sources
from .capstone_incident_classifier import (create_capstone_incident_metadata,
    evaluate_capstone_incident_classifier, load_capstone_classification_data,
    save_capstone_incident_artifact, split_capstone_incident_data,
    train_capstone_incident_classifier)
from .dashboard.service import build_capstone_dashboard, classify_availability
from .data_security import SensitiveFieldError, validate_prediction_payload_fields
from .explainability import explain_linear_prediction
from .human_review import (InMemoryReviewRepository, ReviewRoutingPolicy, ReviewerReason,
    ReviewStatus, create_review_case)
from .integration_failure_model import (IntegrationRequest, build_integration_features,
    build_integration_targets, load_integration_requests, predict_failure_probability)
from .model_monitoring import (build_training_monitoring_baseline,
    compare_shadow_predictions, simulate_production_periods)
from .service.app import create_app
from .service.artifact_loader import ArtifactPaths, ServiceConfig, load_configured_runtimes
from .service.runtimes import CapstoneModelRuntimes
from .service.schemas import CapstoneTelemetryRequest, IntegrationFailureRequest
from .training import (save_model_artifact, save_training_metadata,
    train_integration_failure_model)


@dataclass(frozen=True)
class ArtifactInventoryItem:
    name: str
    version: str
    path: Path
    dataset_hash: str
    trained_at: str
    model_type: str


@dataclass(frozen=True)
class SliceMetric:
    feature: str
    value: str
    support: int
    base_rate: float
    error_rate: float


@dataclass(frozen=True)
class HarborOperatingReport:
    dataset_rows: int
    inventory: tuple[ArtifactInventoryItem, ...]
    health_status: str
    partial_outage: bool
    complete_outage: bool
    deterministic_operation_preserved: bool
    stale_recognized: bool
    timeline_rows: int
    first_anomaly: str | None
    monitoring_periods: tuple[str, ...]
    review_prediction: float
    review_outcome: str
    explanation_delta: float
    slices: tuple[SliceMetric, ...]
    sensitive_field_rejected: bool
    rollback_version: str


def artifact_inventory(config: ServiceConfig) -> tuple[ArtifactInventoryItem, ...]:
    """Read trusted local metadata for an operator-facing inventory, not a public API."""
    items = []
    for paths in (config.anomaly, config.incident, config.integration_failure):
        metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
        items.append(ArtifactInventoryItem(
            str(metadata["model_name"]), str(metadata["model_version"]), paths.model,
            str(metadata.get("dataset_sha256", "unknown")),
            str(metadata.get("trained_at", "unknown")), str(metadata["model_type"])))
    return tuple(items)


def calculate_slice_metrics(observations, probabilities: Sequence[float], *, threshold=.5
                            ) -> tuple[SliceMetric, ...]:
    """Report operational vendor/endpoint slices; this is not demographic fairness."""
    if len(observations) != len(probabilities):
        raise ValueError("observations and probabilities must align")
    result = []
    for feature in ("vendor", "endpoint"):
        for value in sorted({str(getattr(row, feature)) for row in observations}):
            indexes = [i for i, row in enumerate(observations) if str(getattr(row, feature)) == value]
            labels = [observations[i].request_failed for i in indexes]
            errors = [int((probabilities[i] >= threshold) != bool(labels[n]))
                      for n, i in enumerate(indexes)]
            result.append(SliceMetric(feature, value, len(indexes),
                                      sum(labels) / len(labels), sum(errors) / len(errors)))
    return tuple(result)


def _call(app, path: str, request=None):
    """Invoke one synchronous FastAPI endpoint without an external HTTP client."""
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)
    try:
        value = endpoint() if request is None else endpoint(request)
        return 200, value.model_dump()
    except HTTPException as error:
        return error.status_code, {"detail": error.detail}


def _train_runtimes(root: Path, output: Path, now) -> tuple[ServiceConfig, object, object]:
    sources = load_capstone_sources(root / "data")
    timeline = build_capstone_anomaly_timeline(sources)
    baseline = select_anomaly_baseline(timeline)
    anomaly_model = train_capstone_anomaly_detector(baseline)
    anomaly_results = score_capstone_timeline(anomaly_model, timeline)
    anomaly_metadata = create_anomaly_metadata(
        baseline, timeline, evaluate_detection_behavior(anomaly_results), now=now)
    anomaly_paths = save_capstone_anomaly_artifact(anomaly_model, anomaly_metadata, output / "anomaly")

    incident_data = load_capstone_classification_data(root / "data/harbor_incident_classes.csv")
    incident_split = split_capstone_incident_data(incident_data)
    incident_model = train_capstone_incident_classifier(incident_split)
    incident_metadata = create_capstone_incident_metadata(incident_data, incident_split,
        evaluate_capstone_incident_classifier(incident_model, incident_split), now=now)
    incident_paths = save_capstone_incident_artifact(incident_model, incident_metadata, output / "incident")

    integration = train_integration_failure_model(
        root / "data/harbor_integration_requests.csv", now=now)
    integration_paths = (save_model_artifact(integration.pipeline, output / "integration"),
                         save_training_metadata(integration.metadata, output / "integration"))
    config = ServiceConfig(*(ArtifactPaths(*paths) for paths in
        (anomaly_paths, incident_paths, integration_paths)))
    return config, integration, (sources, timeline, anomaly_results)


def run_operating_laboratory(root: str | Path) -> HarborOperatingReport:
    """Assemble the complete laboratory in-process without network or background services."""
    root = Path(root)
    instant = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    now = lambda: instant
    with TemporaryDirectory(prefix="harbor-ch33-") as directory:
        config, integration, capstone = _train_runtimes(root, Path(directory), now)
        sources, timeline, anomaly_results = capstone
        examples = build_capstone_dataset(sources)
        runtimes = load_configured_runtimes(config)
        app = create_app(runtimes)
        _, health = _call(app, "/api/v1/health")
        telemetry = {"api_latency_ms": 1480, "error_rate": .047, "db_connections": 78,
            "queue_depth": 96, "vendor_latency_ms": 1320, "requests_per_minute": 760,
            "retry_count": 3}
        integration_payload = {"vendor": "ClearVerify", "endpoint": "identity_verify",
            "recent_vendor_latency_ms": 940, "recent_vendor_error_rate": .031,
            "queue_depth": 42, "retry_count": 1, "request_size_bytes": 2400,
            "hour_of_day": 14}
        telemetry_request = CapstoneTelemetryRequest(**telemetry)
        integration_request = IntegrationFailureRequest(**integration_payload)
        assert _call(app, "/api/v1/score/telemetry-anomaly", telemetry_request)[0] == 200
        assert _call(app, "/api/v1/predict/incident", telemetry_request)[0] == 200
        assert _call(app, "/api/v1/predict/integration-failure", integration_request)[0] == 200

        partial = CapstoneModelRuntimes(runtimes.anomaly, None, runtimes.integration_failure)
        partial_app = create_app(partial)
        partial_ok = (_call(partial_app, "/api/v1/health")[1]["status"] == "degraded"
            and _call(partial_app, "/api/v1/predict/incident", telemetry_request)[0] == 503
            and _call(partial_app, "/api/v1/score/telemetry-anomaly", telemetry_request)[0] == 200)
        none = CapstoneModelRuntimes(None, None, None)
        none_app = create_app(none)
        complete_ok = (not _call(none_app, "/api/v1/health")[1]["ready"]
            and _call(none_app, "/api/v1/predict/integration-failure", integration_request)[0] == 503)

        dashboard, incident_rows = build_capstone_dashboard(root)
        dashboard.build_snapshot(incident_rows[0])
        dashboard.build_snapshot(incident_rows[6])
        dashboard.build_snapshot(incident_rows[17])
        final = dashboard.build_snapshot(incident_rows[15], retrospective=True)
        stale = classify_availability(instant - timedelta(minutes=6), instant)[0].value == "STALE"

        observations = load_integration_requests(root / "data/harbor_integration_requests.csv")
        requests = [IntegrationRequest(row.vendor, row.endpoint,
            row.recent_vendor_latency_ms, row.recent_vendor_error_rate, row.queue_depth,
            row.retry_count, row.request_size_bytes, row.hour_of_day) for row in observations]
        probabilities = [predict_failure_probability(integration.pipeline, row) for row in requests]
        periods = simulate_production_periods()
        baseline = build_training_monitoring_baseline(observations,
            model_name=integration.metadata.model_name,
            model_version=integration.metadata.model_version,
            dataset_sha256=integration.metadata.dataset_sha256,
            created_at=integration.metadata.trained_at)
        # Structural monitoring checks intentionally do not turn drift into an auto-deploy decision.
        assert baseline.model_version == integration.metadata.model_version and len(periods) == 4
        shadow = compare_shadow_predictions(probabilities, probabilities)
        assert shadow.disagreement_rate == 0 and shadow.average_absolute_probability_difference == 0

        request = IntegrationRequest(**integration_payload)
        explanation = explain_linear_prediction(integration.pipeline, request,
            model_name=integration.metadata.model_name,
            model_version=integration.metadata.model_version)
        predicted = predict_failure_probability(integration.pipeline, request)

        policy = ReviewRoutingPolicy("review-policy-v1", .8)
        repository = InMemoryReviewRepository()
        case = repository.create(create_review_case("review-33", integration.metadata.model_name,
            integration.metadata.model_version, .91, policy, instant))
        repository.transition(case.case_id, ReviewStatus.IN_REVIEW,
                              instant + timedelta(minutes=1), reviewer_id="engineer-17")
        resolved = repository.transition(case.case_id, ReviewStatus.RESOLVED_NO_ISSUE,
            instant + timedelta(minutes=3), reviewer_id="engineer-17",
            reason=ReviewerReason.EXPECTED_PATTERN)

        rejected = False
        try:
            validate_prediction_payload_fields({**integration_payload, "access_token": "temporary"})
        except SensitiveFieldError:
            rejected = True

        slices = calculate_slice_metrics(observations, probabilities)
        # Runtime swap: a candidate identity can be deployed, then the retained v1 runtime restored.
        prior_version = runtimes.integration_failure.identity.version
        candidate_runtime = replace(runtimes.integration_failure,
            identity=replace(runtimes.integration_failure.identity, version=prior_version + "-candidate"))
        deployed = CapstoneModelRuntimes(runtimes.anomaly, runtimes.incident, candidate_runtime)
        assert _call(create_app(deployed), "/api/v1/health")[1]["models"]["integration_failure"]["version"].endswith("-candidate")
        restored = load_configured_runtimes(config)
        restored_version = _call(create_app(restored), "/api/v1/health")[1]["models"]["integration_failure"]["version"]

        return HarborOperatingReport(len(examples), artifact_inventory(config), health["status"],
            partial_ok, complete_ok, True, stale, len(incident_rows),
            next((row.timestamp.isoformat() for row in anomaly_results if row.is_anomaly), None),
            tuple(periods), case.model_probability or 0, resolved.status.value,
            abs(explanation.probability - predicted), slices, rejected, restored_version)
