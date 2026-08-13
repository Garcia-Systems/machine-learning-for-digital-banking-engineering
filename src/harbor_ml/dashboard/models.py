"""Typed state passed from dashboard assembly to presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MLAvailability(str, Enum):
    """Whether prediction evidence is usable as current context."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Direct, deterministic observations from Harbor's fictional system."""

    timestamp: datetime
    api_latency_ms: float
    error_rate: float
    db_connections: int
    queue_depth: int
    vendor_latency_ms: float
    requests_per_minute: float

    def model_features(self) -> dict[str, float]:
        return {
            "api_latency_ms": self.api_latency_ms,
            "error_rate": self.error_rate,
            "db_connections": float(self.db_connections),
            "queue_depth": float(self.queue_depth),
            "vendor_latency_ms": self.vendor_latency_ms,
            "requests_per_minute": self.requests_per_minute,
        }


@dataclass(frozen=True)
class DashboardSnapshot:
    """Observations and explicitly labeled model suggestions for one instant."""

    telemetry: TelemetrySnapshot
    generated_at: datetime
    severity: str
    ml_status: MLAvailability
    prediction_timestamp: datetime | None = None
    prediction_age_seconds: float | None = None
    telemetry_anomaly: bool | None = None
    anomaly_score: float | None = None
    predicted_incident_class: str | None = None
    incident_probabilities: dict[str, float] = field(default_factory=dict)
    classification_ambiguous: bool = False
    integration_failure_probability: float | None = None
    integration_failure_threshold: float | None = None
    integration_failure_prediction: bool | None = None
    model_name: str | None = None
    model_version: str | None = None
    investigation_guidance: tuple[str, ...] = ()

    @property
    def ordered_incident_probabilities(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self.incident_probabilities.items(), key=lambda item: (-item[1], item[0])))
