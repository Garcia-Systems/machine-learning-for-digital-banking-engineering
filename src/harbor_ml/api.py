"""Versioned HTTP boundary for Harbor's advisory integration-failure model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sklearn.pipeline import Pipeline

from .data_security import build_safe_log_context
from .integration_failure_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    IntegrationRequest,
    predict_failure_probability,
)
from .training import MODEL_NAME, load_trusted_model_artifact

LOGGER = logging.getLogger("harbor_ml.api")
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class IntegrationFailureRequest(BaseModel):
    """Public v1 JSON contract; only these fields become model features."""

    model_config = ConfigDict(extra="forbid")

    vendor: NonEmptyText
    endpoint: NonEmptyText
    recent_vendor_latency_ms: float = Field(ge=0)
    recent_vendor_error_rate: float = Field(ge=0, le=1)
    queue_depth: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    request_size_bytes: int = Field(gt=0)
    hour_of_day: int = Field(ge=0, le=23)

    def to_model_request(self) -> IntegrationRequest:
        return IntegrationRequest(**self.model_dump())


class FailurePredictionResponse(BaseModel):
    model: str
    model_version: str
    failure_probability: float
    threshold: float
    predicted_failure: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str


@dataclass(frozen=True)
class ModelRuntime:
    """A fitted pipeline and its small, validated serving configuration."""

    pipeline: Pipeline
    model_name: str
    model_version: str
    threshold: float

    @classmethod
    def load(cls, model_path: str | Path, metadata_path: str | Path) -> "ModelRuntime":
        """Load local trusted assets once; callers control their provenance."""
        metadata_file = Path(metadata_path)
        try:
            metadata: Any = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not load model metadata: {metadata_file}") from error
        if not isinstance(metadata, dict):
            raise RuntimeError("model metadata must be a JSON object")
        expected = {
            "model_name": MODEL_NAME,
            "numerical_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "target": "request_failed",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"model metadata has incompatible {key}")
        version = metadata.get("model_version")
        threshold = metadata.get("classification_threshold")
        if not isinstance(version, str) or not version.strip():
            raise RuntimeError("model metadata has invalid model_version")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise RuntimeError("model metadata has invalid classification_threshold")
        if not 0 <= float(threshold) <= 1:
            raise RuntimeError("model metadata has invalid classification_threshold")
        try:
            pipeline = load_trusted_model_artifact(model_path)
        except Exception as error:
            raise RuntimeError(f"could not load trusted model artifact: {model_path}") from error
        return cls(pipeline, MODEL_NAME, version, float(threshold))

    def predict_failure(self, request: IntegrationFailureRequest) -> FailurePredictionResponse:
        probability = predict_failure_probability(self.pipeline, request.to_model_request())
        return FailurePredictionResponse(
            model=self.model_name,
            model_version=self.model_version,
            failure_probability=probability,
            threshold=self.threshold,
            predicted_failure=probability >= self.threshold,
        )


def create_app(
    runtime: ModelRuntime | None = None,
    *,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> FastAPI:
    """Build an app around an injected runtime or trusted local artifact paths."""
    if runtime is None:
        model_path = model_path or os.getenv(
            "HARBOR_MODEL_PATH", "artifacts/integration-failure/model.joblib"
        )
        metadata_path = metadata_path or os.getenv(
            "HARBOR_MODEL_METADATA_PATH", "artifacts/integration-failure/metadata.json"
        )
        runtime = ModelRuntime.load(model_path, metadata_path)

    app = FastAPI(title="Harbor Integration Failure Prediction API", version="1")
    app.state.model_runtime = runtime

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=runtime.model_name,
            model_version=runtime.model_version,
        )

    @app.post(
        "/api/v1/predict/integration-failure",
        response_model=FailurePredictionResponse,
    )
    def predict(request: IntegrationFailureRequest) -> FailurePredictionResponse:
        started = perf_counter()
        try:
            response = runtime.predict_failure(request)
        except Exception:
            LOGGER.exception(
                "prediction_failed endpoint=%s model_version=%s",
                "/api/v1/predict/integration-failure",
                runtime.model_version,
            )
            raise HTTPException(status_code=500, detail="prediction could not be calculated")
        LOGGER.info(
            "prediction_succeeded context=%s latency_ms=%.3f",
            build_safe_log_context(request.model_dump(), model_version=runtime.model_version),
            (perf_counter() - started) * 1_000,
        )
        return response

    return app
