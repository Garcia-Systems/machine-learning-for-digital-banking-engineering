"""Typed presentation models for Harbor's engineering dashboards."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MLAvailability(str, Enum):
    """Whether one independently produced prediction can be used as current context."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"


# Chapter 32's name makes the generic Chapter 20 enum explicit without breaking its API.
SignalAvailability = MLAvailability


@dataclass(frozen=True)
class TelemetrySnapshot:
    timestamp: datetime
    api_latency_ms: float
    error_rate: float
    db_connections: int
    queue_depth: int
    vendor_latency_ms: float
    requests_per_minute: float
    retry_count: int = 0

    def model_features(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in (
            "api_latency_ms", "error_rate", "db_connections", "queue_depth",
            "vendor_latency_ms", "requests_per_minute",
        )}


@dataclass(frozen=True)
class SignalMetadata:
    availability: SignalAvailability
    model_name: str
    model_version: str | None
    prediction_timestamp: datetime | None
    prediction_age_seconds: float | None


@dataclass(frozen=True)
class AnomalyDashboardSignal:
    metadata: SignalMetadata
    is_anomaly: bool | None = None
    score: float | None = None


@dataclass(frozen=True)
class IncidentDashboardSignal:
    metadata: SignalMetadata
    predicted_class: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    ambiguous: bool = False

    @property
    def ordered_probabilities(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self.probabilities.items(), key=lambda item: (-item[1], item[0])))


@dataclass(frozen=True)
class ExplanationFactor:
    feature_name: str
    contribution: float


@dataclass(frozen=True)
class IntegrationFailureDashboardSignal:
    metadata: SignalMetadata
    failure_probability: float | None = None
    threshold: float | None = None
    elevated: bool | None = None
    increased_factors: tuple[ExplanationFactor, ...] = ()
    decreased_factors: tuple[ExplanationFactor, ...] = ()


@dataclass(frozen=True)
class ModelMonitoringSummary:
    prediction_api_latency_ms: float
    unknown_categories: int
    drift_status: str
    labeled_performance_status: str


@dataclass(frozen=True)
class DashboardSnapshot:
    """Evidence available at one instant; future confirmation is never attached."""

    telemetry: TelemetrySnapshot
    generated_at: datetime
    severity: str
    anomaly: AnomalyDashboardSignal
    incident: IncidentDashboardSignal
    integration_failure: IntegrationFailureDashboardSignal
    ml_capability_status: str
    investigation_guidance: tuple[str, ...]
    confirmed_evidence: tuple[str, ...] = ()
    retrospective: bool = False
    monitoring: ModelMonitoringSummary | None = None

    # Compatibility properties retained for the Chapter 20 lab and readers.
    @property
    def ml_status(self) -> MLAvailability:
        statuses = {self.anomaly.metadata.availability, self.incident.metadata.availability,
                    self.integration_failure.metadata.availability}
        return statuses.pop() if len(statuses) == 1 else MLAvailability.UNAVAILABLE

    @property
    def telemetry_anomaly(self): return self.anomaly.is_anomaly
    @property
    def anomaly_score(self): return self.anomaly.score
    @property
    def predicted_incident_class(self): return self.incident.predicted_class
    @property
    def incident_probabilities(self): return self.incident.probabilities
    @property
    def classification_ambiguous(self): return self.incident.ambiguous
    @property
    def integration_failure_probability(self): return self.integration_failure.failure_probability
    @property
    def integration_failure_threshold(self): return self.integration_failure.threshold
    @property
    def integration_failure_prediction(self): return self.integration_failure.elevated
    @property
    def model_name(self): return self.integration_failure.metadata.model_name
    @property
    def model_version(self): return self.integration_failure.metadata.model_version
    @property
    def prediction_timestamp(self): return self.integration_failure.metadata.prediction_timestamp
    @property
    def prediction_age_seconds(self): return self.integration_failure.metadata.prediction_age_seconds
    @property
    def ordered_incident_probabilities(self): return self.incident.ordered_probabilities


CapstoneDashboardSnapshot = DashboardSnapshot
