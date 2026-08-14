"""Integrated, evidence-ordered incident laboratory for Chapter 26."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from .anomaly_detection import AnomalyResult, score_observation
from .dashboard.models import TelemetrySnapshot
from .dashboard.service import calculate_severity, is_ambiguous
from .incident_classifier import INCIDENT_CLASSES, IncidentPrediction, predict_incident
from .integration_failure_model import IntegrationRequest, predict_failure_probability

CAPSTONE_FIELDS = (
    "timestamp", "api_latency_ms", "error_rate", "db_connections", "queue_depth",
    "vendor_latency_ms", "requests_per_minute", "retry_count", "vendor_timeout_rate",
    "deployment_version", "incident_classifier_available",
)
TRACE_FIELDS = ("request_id", "timestamp", "component", "duration_ms", "status")
CONFIRMING_EVIDENCE_TIME = datetime.fromisoformat("2026-08-12T10:30:00+00:00")
FINAL_DIAGNOSIS = (
    "ClearVerify degradation is the primary confirmed contributor to the Harbor "
    "identity-verification incident. Increased retries and longer-lived requests "
    "subsequently increased queue depth and database connection pressure, amplifying "
    "application latency."
)
MODEL_IDENTITIES = {
    "telemetry_anomaly": ("harbor-telemetry-anomaly", "chapter-04-iforest-v1"),
    "incident_classifier": ("harbor-incident-classifier", "chapter-05-logreg-v1"),
}


@dataclass(frozen=True)
class CapstoneTelemetry:
    timestamp: datetime
    api_latency_ms: float
    error_rate: float
    db_connections: int
    queue_depth: int
    vendor_latency_ms: float
    requests_per_minute: float
    retry_count: int
    vendor_timeout_rate: float
    deployment_version: str
    incident_classifier_available: bool

    def model_features(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in (
            "api_latency_ms", "error_rate", "db_connections", "queue_depth",
            "vendor_latency_ms", "requests_per_minute",
        )}

    def dashboard_snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(self.timestamp, self.api_latency_ms, self.error_rate,
                                 self.db_connections, self.queue_depth,
                                 self.vendor_latency_ms, self.requests_per_minute)


@dataclass(frozen=True)
class TraceSpan:
    request_id: str
    timestamp: datetime
    component: str
    duration_ms: int
    status: str


@dataclass(frozen=True)
class IncidentNote:
    timestamp: datetime
    category: str
    text: str

    def __post_init__(self) -> None:
        if self.category not in {"observation", "model_signal", "hypothesis", "investigation",
                                "evidence", "diagnosis", "action"}:
            raise ValueError("invalid incident-note category")


@dataclass(frozen=True)
class IncidentObservation:
    timestamp: datetime
    telemetry: CapstoneTelemetry
    phase: str
    severity: str
    anomaly_result: AnomalyResult | None
    incident_prediction: IncidentPrediction | None
    classification: str
    integration_failure_probability: float | None
    model_versions: dict[str, str]
    diagnosis_confirmed: bool
    notes: tuple[IncidentNote, ...]


def _timestamp(value: str, row: int) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"row {row}: invalid timestamp") from error
    if result.tzinfo is None:
        raise ValueError(f"row {row}: timestamp must be timezone-aware")
    return result


def load_capstone_incident(path: str | Path) -> list[CapstoneTelemetry]:
    """Load privacy-minimized fictional telemetry and enforce time ordering."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(CAPSTONE_FIELDS):
            raise ValueError("invalid capstone timeline header")
        rows = list(reader)
    observations: list[CapstoneTelemetry] = []
    for number, row in enumerate(rows, 2):
        try:
            values = {name: float(row[name]) for name in (
                "api_latency_ms", "error_rate", "vendor_latency_ms", "requests_per_minute",
                "vendor_timeout_rate")}
            counts = {name: int(row[name]) for name in
                      ("db_connections", "queue_depth", "retry_count")}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"row {number}: invalid telemetry value") from error
        if any(not isfinite(value) or value < 0 for value in (*values.values(), *counts.values())):
            raise ValueError(f"row {number}: telemetry must be finite and non-negative")
        if values["error_rate"] > 1 or values["vendor_timeout_rate"] > 1:
            raise ValueError(f"row {number}: rates must be between zero and one")
        available = row["incident_classifier_available"].lower()
        if available not in {"true", "false"}:
            raise ValueError(f"row {number}: invalid availability")
        item = CapstoneTelemetry(_timestamp(row["timestamp"], number), **values, **counts,
                                 deployment_version=row["deployment_version"],
                                 incident_classifier_available=available == "true")
        if observations and item.timestamp <= observations[-1].timestamp:
            raise ValueError(f"row {number}: timestamps must be strictly chronological")
        observations.append(item)
    if not observations:
        raise ValueError("capstone timeline must contain observations")
    return observations


def load_capstone_traces(path: str | Path) -> list[TraceSpan]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(TRACE_FIELDS):
            raise ValueError("invalid capstone trace header")
        rows = list(reader)
    spans = [TraceSpan(row["request_id"], _timestamp(row["timestamp"], number),
                       row["component"], int(row["duration_ms"]), row["status"])
             for number, row in enumerate(rows, 2)]
    if not spans or any(span.duration_ms < 0 for span in spans):
        raise ValueError("trace fixture must contain non-negative spans")
    return spans


def dominant_trace_component(spans: Sequence[TraceSpan], request_id: str) -> TraceSpan:
    selected = [span for span in spans if span.request_id == request_id]
    if not selected:
        raise ValueError("request_id not found")
    return max(selected, key=lambda span: span.duration_ms)


def build_representative_request(row: CapstoneTelemetry) -> IntegrationRequest:
    return IntegrationRequest(
        vendor="ClearVerify", endpoint="identity_verify",
        recent_vendor_latency_ms=row.vendor_latency_ms,
        recent_vendor_error_rate=row.vendor_timeout_rate, queue_depth=row.queue_depth,
        retry_count=row.retry_count, request_size_bytes=1800, hour_of_day=row.timestamp.hour,
    )


def classify_incident_phase(timestamp: datetime) -> str:
    minute = timestamp.minute
    if minute <= 4:
        return "healthy"
    if minute <= 10:
        return "early_signal"
    if minute <= 18:
        return "degradation"
    if minute <= 28:
        return "compound_pressure"
    if minute <= 30:
        return "confirmed_incident"
    return "recovery"


def _notes(row: CapstoneTelemetry, confirmed: bool) -> tuple[IncidentNote, ...]:
    if confirmed:
        return (
            IncidentNote(row.timestamp, "evidence", "Trace spans identify ClearVerify_call as dominant request time."),
            IncidentNote(row.timestamp, "diagnosis", FINAL_DIAGNOSIS),
            IncidentNote(row.timestamp, "action", "Reduce retries, activate degraded mode, notify engineering and vendor management, and monitor recovery."),
        )
    if row.timestamp.minute >= 14:
        return (
            IncidentNote(row.timestamp, "hypothesis", "The external verification dependency may be contributing."),
            IncidentNote(row.timestamp, "investigation", "Inspect ClearVerify traces, retries, deployment history, and database pressure."),
        )
    return (IncidentNote(row.timestamp, "observation", "Direct telemetry recorded; diagnosis is not established."),)


def evaluate_incident_timeline(
    rows: Sequence[CapstoneTelemetry], anomaly_detector: IsolationForest,
    incident_model: Pipeline, integration_model: Pipeline, *, integration_model_version: str,
) -> list[IncidentObservation]:
    """Run existing fitted models; availability never erases direct telemetry."""
    evaluated = []
    for row in rows:
        features = row.model_features()
        anomaly = score_observation(anomaly_detector, features)
        incident = predict_incident(incident_model, features) if row.incident_classifier_available else None
        classification = ("unavailable" if incident is None else
                          "ambiguous" if is_ambiguous(incident.probabilities) else incident.predicted_class)
        failure_probability = predict_failure_probability(
            integration_model, build_representative_request(row))
        confirmed = row.timestamp >= CONFIRMING_EVIDENCE_TIME
        versions = {
            MODEL_IDENTITIES["telemetry_anomaly"][0]: MODEL_IDENTITIES["telemetry_anomaly"][1],
            "harbor-integration-failure": integration_model_version,
        }
        if incident is not None:
            versions[MODEL_IDENTITIES["incident_classifier"][0]] = MODEL_IDENTITIES["incident_classifier"][1]
        evaluated.append(IncidentObservation(
            row.timestamp, row, classify_incident_phase(row.timestamp),
            calculate_severity(row.dashboard_snapshot()), anomaly, incident, classification,
            failure_probability, versions, confirmed, _notes(row, confirmed),
        ))
    return evaluated


def format_incident_observation(item: IncidentObservation) -> str:
    """Render evidence layers explicitly instead of collapsing them into diagnosis."""
    row = item.telemetry
    lines = [item.timestamp.strftime("%H:%M"), "", "OBSERVED",
             f"API latency: {row.api_latency_ms:.0f} ms",
             f"Error rate: {row.error_rate:.1%}", f"Queue depth: {row.queue_depth}",
             f"DB connections: {row.db_connections}",
             f"Vendor latency: {row.vendor_latency_ms:.0f} ms",
             f"Requests/minute: {row.requests_per_minute:.0f}",
             f"Retries: {row.retry_count}", f"Deployment: {row.deployment_version}",
             "", "DETERMINISTIC STATUS", item.severity, "", "ML SIGNALS",
             f"Telemetry anomaly: {'yes' if item.anomaly_result and item.anomaly_result.is_anomaly else 'no'} "
             f"(score={item.anomaly_result.score:.3f})",
             f"Integration failure probability: {item.integration_failure_probability:.3f}"]
    if item.incident_prediction is None:
        lines += ["Incident classifier: UNAVAILABLE"]
    else:
        lines += ["Incident probabilities:"] + [
            f"  {label}: {item.incident_prediction.probabilities[label]:.3f}"
            for label in INCIDENT_CLASSES
        ] + [f"classification = {item.classification}"]
    lines += ["Model versions:"] + [f"  {name}: {version}" for name, version in item.model_versions.items()]
    lines += ["", "ENGINEERING INTERPRETATION"]
    if item.diagnosis_confirmed:
        lines += [FINAL_DIAGNOSIS]
    elif item.phase == "healthy":
        lines += ["No incident evidence yet."]
    else:
        lines += ["Hypothesis: the external verification dependency may be contributing.",
                  "Next: inspect traces, retries, deployment history, and database pressure.",
                  "NOT ESTABLISHED: a primary contributor is not yet confirmed."]
    return "\n".join(lines)
