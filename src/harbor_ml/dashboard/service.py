"""Assembly layer that keeps model calls out of dashboard templates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from harbor_ml.anomaly_detection import (
    build_anomaly_detector, build_anomaly_features, load_normal_telemetry,
    score_observation, train_anomaly_detector,
)
from harbor_ml.incident_classifier import (
    build_incident_classifier, build_incident_features, build_incident_targets,
    load_incident_dataset, predict_incident, train_incident_classifier,
)
from harbor_ml.integration_failure_model import IntegrationRequest, predict_failure_probability
from harbor_ml.training import MODEL_NAME, train_integration_failure_model

from .models import DashboardSnapshot, MLAvailability, TelemetrySnapshot

FRESHNESS_WINDOW = timedelta(minutes=5)
AMBIGUITY_GAP = 0.10

INVESTIGATION_GUIDANCE: dict[str, tuple[str, ...]] = {
    "vendor_degradation": (
        "Inspect vendor latency and timeout telemetry.",
        "Review retry behavior.",
        "Inspect distributed traces involving the vendor.",
    ),
    "database_pressure": (
        "Inspect connection-pool pressure.",
        "Review slow-query telemetry.",
        "Inspect lock and queue behavior.",
    ),
    "application_regression": (
        "Review recent deployments and feature changes.",
        "Compare errors and traces across application versions.",
        "Inspect affected endpoints.",
    ),
    "traffic_spike": (
        "Compare request volume with the normal time-of-day baseline.",
        "Inspect rate limits and autoscaling behavior.",
        "Review the endpoints receiving additional traffic.",
    ),
    "normal": (
        "Continue checking direct telemetry for localized problems.",
        "Compare the current observation with recent history.",
    ),
}


def calculate_severity(telemetry: TelemetrySnapshot) -> str:
    """Apply fictional teaching thresholds, independently of ML output."""
    if telemetry.error_rate >= 0.05 or telemetry.api_latency_ms >= 2_000:
        return "critical"
    if telemetry.error_rate >= 0.02 or telemetry.api_latency_ms >= 800:
        return "warning"
    return "normal"


def classify_availability(
    prediction_timestamp: datetime | None,
    now: datetime,
    freshness_window: timedelta,
) -> tuple[MLAvailability, float | None]:
    if prediction_timestamp is None:
        return MLAvailability.UNAVAILABLE, None
    age = max(0.0, (now - prediction_timestamp).total_seconds())
    status = MLAvailability.STALE if age > freshness_window.total_seconds() else MLAvailability.AVAILABLE
    return status, age


def is_ambiguous(probabilities: dict[str, float], *, gap: float = AMBIGUITY_GAP) -> bool:
    ordered = sorted(probabilities.values(), reverse=True)
    return len(ordered) >= 2 and ordered[0] - ordered[1] < gap


class DashboardService:
    """Combine reusable Chapter 4, 5, and 16 model outputs with telemetry."""

    def __init__(
        self,
        anomaly_detector: IsolationForest,
        incident_model: Pipeline,
        integration_model: Pipeline,
        *,
        model_version: str,
        integration_threshold: float,
        freshness_window: timedelta = FRESHNESS_WINDOW,
    ) -> None:
        self.anomaly_detector = anomaly_detector
        self.incident_model = incident_model
        self.integration_model = integration_model
        self.model_version = model_version
        self.integration_threshold = integration_threshold
        self.freshness_window = freshness_window
        self.history: list[DashboardSnapshot] = []

    def assemble(
        self,
        telemetry: TelemetrySnapshot,
        integration_request: IntegrationRequest,
        *,
        now: datetime | None = None,
        prediction_timestamp: datetime | None = None,
        ml_available: bool = True,
        remember: bool = True,
    ) -> DashboardSnapshot:
        now = now or datetime.now(timezone.utc)
        prediction_timestamp = (prediction_timestamp or now) if ml_available else None
        status, age = classify_availability(prediction_timestamp, now, self.freshness_window)
        common = dict(
            telemetry=telemetry, generated_at=now, severity=calculate_severity(telemetry),
            ml_status=status, prediction_timestamp=prediction_timestamp,
            prediction_age_seconds=age,
        )
        if not ml_available:
            snapshot = DashboardSnapshot(**common)
        else:
            features = telemetry.model_features()
            anomaly = score_observation(self.anomaly_detector, features)
            incident = predict_incident(self.incident_model, features)
            failure_probability = predict_failure_probability(self.integration_model, integration_request)
            ambiguous = is_ambiguous(incident.probabilities)
            snapshot = DashboardSnapshot(
                **common,
                telemetry_anomaly=anomaly.is_anomaly,
                anomaly_score=anomaly.score,
                predicted_incident_class=incident.predicted_class,
                incident_probabilities=incident.probabilities,
                classification_ambiguous=ambiguous,
                integration_failure_probability=failure_probability,
                integration_failure_threshold=self.integration_threshold,
                integration_failure_prediction=failure_probability >= self.integration_threshold,
                model_name=MODEL_NAME,
                model_version=self.model_version,
                investigation_guidance=INVESTIGATION_GUIDANCE.get(incident.predicted_class, ()),
            )
        if remember:
            self.history = (self.history + [snapshot])[-8:]
        return snapshot


def build_teaching_service(root: str | Path) -> DashboardService:
    """Fit existing deterministic educational fixtures; no dashboard-specific model."""
    root = Path(root)
    baseline = load_normal_telemetry(root / "data/harbor_normal_telemetry.csv")
    anomaly = train_anomaly_detector(build_anomaly_detector(), build_anomaly_features(baseline))
    incidents = load_incident_dataset(root / "data/harbor_incident_classes.csv")
    incident = train_incident_classifier(
        build_incident_classifier(), build_incident_features(incidents), build_incident_targets(incidents)
    )
    integration = train_integration_failure_model(root / "data/harbor_integration_requests.csv")
    return DashboardService(
        anomaly, incident, integration.pipeline,
        model_version=integration.metadata.model_version,
        integration_threshold=integration.metadata.classification_threshold,
    )


def teaching_scenarios(now: datetime) -> Sequence[tuple[str, TelemetrySnapshot, IntegrationRequest]]:
    """Five deterministic observation/request pairs for the executable lab."""
    def scenario(name: str, minutes: int, values: tuple[float, float, int, int, float, float], request: IntegrationRequest):
        return name, TelemetrySnapshot(now - timedelta(minutes=minutes), *values), request

    healthy_request = IntegrationRequest("ClearVerify", "identity_verify", 240, 0.003, 11, 0, 1200, 14)
    vendor_request = IntegrationRequest("ClearVerify", "identity_verify", 1320, 0.08, 96, 3, 2400, 14)
    database_request = IntegrationRequest("LedgerLink", "account_sync", 210, 0.008, 88, 1, 1900, 14)
    return (
        scenario("healthy", 15, (176, .0039, 31, 11, 245, 428), healthy_request),
        scenario("vendor_degradation", 10, (1480, .047, 78, 96, 1320, 710), vendor_request),
        scenario("database_pressure", 5, (920, .031, 96, 88, 210, 680), database_request),
        scenario("ml_unavailable", 2, (640, .012, 55, 40, 360, 650), healthy_request),
        scenario("stale_prediction", 0, (1480, .047, 78, 96, 1320, 710), vendor_request),
    )
