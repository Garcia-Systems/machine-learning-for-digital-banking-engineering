"""Public JSON contracts for the capstone service."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CapstoneTelemetryRequest(BaseModel):
    """Shared v1 telemetry envelope; individual runtimes select their features."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    api_latency_ms: float = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    db_connections: int = Field(ge=0)
    queue_depth: int = Field(ge=0)
    vendor_latency_ms: float = Field(ge=0)
    requests_per_minute: float = Field(ge=0)
    retry_count: int = Field(ge=0)


class IntegrationFailureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    vendor: NonEmptyText
    endpoint: NonEmptyText
    recent_vendor_latency_ms: float = Field(ge=0)
    recent_vendor_error_rate: float = Field(ge=0, le=1)
    queue_depth: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    request_size_bytes: int = Field(gt=0)
    hour_of_day: int = Field(ge=0, le=23)


class AnomalyResponse(BaseModel):
    model: str
    model_version: str
    anomaly_score: float
    is_anomaly: bool


class IncidentPredictionResponse(BaseModel):
    model: str
    model_version: str
    predicted_class: str
    probabilities: dict[str, float]
    top_probability: float
    second_probability: float
    probability_gap: float
    ambiguous: bool


class FailurePredictionResponse(BaseModel):
    model: str
    model_version: str
    failure_probability: float
    threshold: float
    predicted_failure: bool


class ModelHealth(BaseModel):
    loaded: bool
    version: str | None
    feature_contract_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    models: dict[str, ModelHealth]
