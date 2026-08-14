"""Assembly services: templates never invoke fitted models or discover evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from harbor_ml.anomaly_detection import (build_anomaly_detector, build_anomaly_features,
    load_normal_telemetry, score_observation, train_anomaly_detector)
from harbor_ml.explainability import explain_linear_prediction, sorted_contributions
from harbor_ml.incident_classifier import (build_incident_classifier, build_incident_features,
    build_incident_targets, load_incident_dataset, predict_incident, train_incident_classifier)
from harbor_ml.integration_failure_model import IntegrationRequest, predict_failure_probability
from harbor_ml.training import MODEL_NAME, TrainingConfig, train_integration_failure_model

from .models import (AnomalyDashboardSignal, DashboardSnapshot, ExplanationFactor,
    IncidentDashboardSignal, IntegrationFailureDashboardSignal, MLAvailability,
    ModelMonitoringSummary, SignalMetadata, TelemetrySnapshot)

FRESHNESS_WINDOW = timedelta(minutes=5)
AMBIGUITY_GAP = 0.10
ANOMALY_IDENTITY = ("harbor-capstone-anomaly", "chapter-28-iforest-v1")
INCIDENT_IDENTITY = ("harbor-capstone-incident", "chapter-29-logreg-v1")
CONFIRMED_TRACE_TEXT = (
    "Trace evidence shows the ClearVerify call consumed the dominant share of request "
    "duration in the sampled failing requests."
)

INVESTIGATION_GUIDANCE: dict[str, tuple[str, ...]] = {
    "vendor_degradation": ("Inspect ClearVerify latency and timeout traces.", "Review retry activity.",
                           "Inspect request queue growth."),
    "database_pressure": ("Inspect the database connection pool and slow-query telemetry.",
                          "Compare database connection pressure with vendor-call duration."),
    "application_regression": ("Review recent deployments and compare endpoint traces by version.",),
    "traffic_spike": ("Compare request volume with its time-of-day baseline and inspect rate limits.",),
    "normal": ("Continue checking direct telemetry and recent history for localized problems.",),
}


def calculate_severity(telemetry: TelemetrySnapshot) -> str:
    if telemetry.error_rate >= .05 or telemetry.api_latency_ms >= 2_000:
        return "critical"
    if telemetry.error_rate >= .02 or telemetry.api_latency_ms >= 800:
        return "warning"
    return "normal"


def classify_availability(prediction_timestamp, now, freshness_window=FRESHNESS_WINDOW):
    if prediction_timestamp is None:
        return MLAvailability.UNAVAILABLE, None
    age = max(0.0, (now - prediction_timestamp).total_seconds())
    status = MLAvailability.STALE if age > freshness_window.total_seconds() else MLAvailability.AVAILABLE
    return status, age


def is_ambiguous(probabilities: Mapping[str, float], *, gap: float = AMBIGUITY_GAP) -> bool:
    ordered = sorted(probabilities.values(), reverse=True)
    return len(ordered) >= 2 and ordered[0] - ordered[1] < gap


def _metadata(identity, timestamp, now, freshness):
    status, age = classify_availability(timestamp, now, freshness)
    return SignalMetadata(status, identity[0], identity[1] if timestamp else None, timestamp, age)


def _capability(*signals) -> str:
    available = sum(signal.metadata.availability is not MLAvailability.UNAVAILABLE for signal in signals)
    return "unavailable" if available == 0 else "all_available" if available == len(signals) else "partially_available"


class DashboardService:
    """Chapter 20-compatible service extended with independent Chapter 32 signals."""

    def __init__(self, anomaly_detector: IsolationForest, incident_model: Pipeline,
                 integration_model: Pipeline, *, model_version: str,
                 integration_threshold: float, freshness_window=FRESHNESS_WINDOW) -> None:
        self.anomaly_detector, self.incident_model = anomaly_detector, incident_model
        self.integration_model, self.model_version = integration_model, model_version
        self.integration_threshold, self.freshness_window = integration_threshold, freshness_window
        self.history: list[DashboardSnapshot] = []

    def assemble(self, telemetry: TelemetrySnapshot, integration_request: IntegrationRequest, *,
                 now=None, prediction_timestamp=None, ml_available=True, remember=True,
                 signal_available: Mapping[str, bool] | None = None,
                 prediction_timestamps: Mapping[str, datetime | None] | None = None,
                 confirmed_evidence: tuple[str, ...] = (), retrospective=False) -> DashboardSnapshot:
        now = now or datetime.now(timezone.utc)
        enabled = {"anomaly": ml_available, "incident": ml_available, "integration": ml_available}
        enabled.update(signal_available or {})
        timestamps = {key: ((prediction_timestamp or now) if value else None) for key, value in enabled.items()}
        timestamps.update(prediction_timestamps or {})

        anomaly_meta = _metadata(ANOMALY_IDENTITY, timestamps["anomaly"], now, self.freshness_window)
        incident_meta = _metadata(INCIDENT_IDENTITY, timestamps["incident"], now, self.freshness_window)
        integration_meta = _metadata((MODEL_NAME, self.model_version), timestamps["integration"], now, self.freshness_window)
        features = telemetry.model_features()
        anomaly_result = score_observation(self.anomaly_detector, features) if enabled["anomaly"] else None
        incident_result = predict_incident(self.incident_model, features) if enabled["incident"] else None
        probability = predict_failure_probability(self.integration_model, integration_request) if enabled["integration"] else None
        explanation = (explain_linear_prediction(self.integration_model, integration_request,
            model_name=MODEL_NAME, model_version=self.model_version) if enabled["integration"] else None)
        factors = lambda positive: tuple(ExplanationFactor(x.feature_name, x.contribution)
            for x in sorted_contributions(explanation, positive=positive, limit=3)) if explanation else ()
        anomaly = AnomalyDashboardSignal(anomaly_meta,
            anomaly_result.is_anomaly if anomaly_result else None, anomaly_result.score if anomaly_result else None)
        incident = IncidentDashboardSignal(incident_meta,
            incident_result.predicted_class if incident_result else None,
            incident_result.probabilities if incident_result else {},
            is_ambiguous(incident_result.probabilities) if incident_result else False)
        integration = IntegrationFailureDashboardSignal(integration_meta, probability,
            self.integration_threshold if probability is not None else None,
            probability >= self.integration_threshold if probability is not None else None,
            factors(True), factors(False))
        guidance = list(INVESTIGATION_GUIDANCE.get(incident.predicted_class, ()))
        if incident.ambiguous:
            for label, _ in incident.ordered_probabilities[:2]:
                guidance.extend(x for x in INVESTIGATION_GUIDANCE.get(label, ()) if x not in guidance)
        if telemetry.queue_depth >= 80 and "Inspect request queue growth." not in guidance:
            guidance.append("Inspect request queue growth.")
        snapshot = DashboardSnapshot(telemetry, now, calculate_severity(telemetry), anomaly,
            incident, integration, _capability(anomaly, incident, integration), tuple(guidance),
            confirmed_evidence, retrospective,
            ModelMonitoringSummary(18.0, 0, "within baseline", "awaiting delayed labels"))
        if remember:
            self.history = (self.history + [snapshot])[-20:]
        return snapshot


class CapstoneDashboardService(DashboardService):
    """Build time-bounded views over the fictional Chapter 26 incident."""

    def build_snapshot(self, observation, *, now=None, all_ml_available=True,
                       prediction_timestamps=None, retrospective=False, remember=False):
        row = getattr(observation, "telemetry", observation)
        availability = {"anomaly": all_ml_available,
                        "incident": all_ml_available and row.incident_classifier_available,
                        "integration": all_ml_available}
        evidence = (CONFIRMED_TRACE_TEXT,) if row.timestamp.minute >= 30 else ()
        return self.assemble(row.dashboard_snapshot().__class__(row.timestamp, row.api_latency_ms,
            row.error_rate, row.db_connections, row.queue_depth, row.vendor_latency_ms,
            row.requests_per_minute, row.retry_count),
            __import__("harbor_ml.capstone_incident", fromlist=["build_representative_request"]).build_representative_request(row),
            now=now or row.timestamp, signal_available=availability,
            prediction_timestamps=prediction_timestamps, confirmed_evidence=evidence,
            retrospective=retrospective, remember=remember)


def build_teaching_service(root: str | Path) -> DashboardService:
    root = Path(root)
    baseline = load_normal_telemetry(root / "data/harbor_normal_telemetry.csv")
    anomaly = train_anomaly_detector(build_anomaly_detector(), build_anomaly_features(baseline))
    incidents = load_incident_dataset(root / "data/harbor_incident_classes.csv")
    incident = train_incident_classifier(build_incident_classifier(), build_incident_features(incidents), build_incident_targets(incidents))
    integration = train_integration_failure_model(root / "data/harbor_integration_requests.csv", TrainingConfig())
    return DashboardService(anomaly, incident, integration.pipeline,
        model_version=integration.metadata.model_version,
        integration_threshold=integration.metadata.classification_threshold)


def build_capstone_dashboard(root: str | Path):
    from harbor_ml.capstone_incident import load_capstone_incident
    base = build_teaching_service(root)
    service = CapstoneDashboardService(base.anomaly_detector, base.incident_model,
        base.integration_model, model_version=base.model_version,
        integration_threshold=base.integration_threshold)
    rows = load_capstone_incident(Path(root) / "data/harbor_capstone_incident.csv")
    return service, rows


def teaching_scenarios(now: datetime) -> Sequence[tuple[str, TelemetrySnapshot, IntegrationRequest]]:
    healthy = IntegrationRequest("ClearVerify", "identity_verify", 240, .003, 11, 0, 1200, 14)
    vendor = IntegrationRequest("ClearVerify", "identity_verify", 1320, .08, 96, 3, 2400, 14)
    database = IntegrationRequest("LedgerLink", "account_sync", 210, .008, 88, 1, 1900, 14)
    def item(name, minutes, values, request): return name, TelemetrySnapshot(now-timedelta(minutes=minutes), *values), request
    return (item("healthy", 15, (176,.0039,31,11,245,428,0), healthy),
            item("vendor_degradation",10,(1480,.047,78,96,1320,710,3),vendor),
            item("database_pressure",5,(920,.031,96,88,210,680,1),database),
            item("ml_unavailable",2,(640,.012,55,40,360,650,0),healthy),
            item("stale_prediction",0,(1480,.047,78,96,1320,710,3),vendor))
