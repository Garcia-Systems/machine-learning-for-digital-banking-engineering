"""FastAPI boundary for Harbor's three advisory capstone models."""

import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException

from .artifact_loader import ServiceConfig, load_configured_runtimes
from .runtimes import CapstoneModelRuntimes, FEATURE_CONTRACT_VERSION
from .schemas import (
    AnomalyResponse, CapstoneTelemetryRequest, FailurePredictionResponse,
    HealthResponse, IncidentPredictionResponse, IntegrationFailureRequest, ModelHealth,
)

LOGGER = logging.getLogger("harbor_ml.service")


def create_app(runtimes: CapstoneModelRuntimes) -> FastAPI:
    """Create an app around startup-loaded runtimes; handlers never load or fit."""
    app = FastAPI(title="Harbor Capstone ML Service", version="1")
    app.state.model_runtimes = runtimes

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        models = {
            "capstone_anomaly": ModelHealth(loaded=runtimes.anomaly is not None,
                version=runtimes.anomaly.identity.version if runtimes.anomaly else None,
                feature_contract_version=FEATURE_CONTRACT_VERSION),
            "capstone_incident": ModelHealth(loaded=runtimes.incident is not None,
                version=runtimes.incident.identity.version if runtimes.incident else None,
                feature_contract_version=FEATURE_CONTRACT_VERSION),
            "integration_failure": ModelHealth(loaded=runtimes.integration_failure is not None,
                version=runtimes.integration_failure.identity.version
                if runtimes.integration_failure else None),
        }
        return HealthResponse(status="ok" if runtimes.ready else "degraded",
                              ready=runtimes.ready, models=models)

    def infer(route: str, runtime, operation):
        if runtime is None:
            descriptions = {"anomaly": "Telemetry anomaly model",
                            "incident": "Incident prediction model",
                            "integration": "Integration failure model"}
            raise HTTPException(503, f"{descriptions[route]} is unavailable.")
        started = perf_counter()
        try:
            response = operation(runtime)
        except Exception:
            LOGGER.exception("inference_failed route=%s model=%s model_version=%s",
                             route, runtime.identity.name, runtime.identity.version)
            raise HTTPException(500, "prediction could not be calculated")
        LOGGER.info("inference_succeeded route=%s model=%s model_version=%s latency_ms=%.3f",
                    route, runtime.identity.name, runtime.identity.version,
                    (perf_counter() - started) * 1_000)
        return response

    @app.post("/api/v1/score/telemetry-anomaly", response_model=AnomalyResponse)
    def score_anomaly(request: CapstoneTelemetryRequest) -> AnomalyResponse:
        return infer("anomaly", runtimes.anomaly, lambda runtime: runtime.score(request))

    @app.post("/api/v1/predict/incident", response_model=IncidentPredictionResponse)
    def predict_incident(request: CapstoneTelemetryRequest) -> IncidentPredictionResponse:
        return infer("incident", runtimes.incident, lambda runtime: runtime.predict(request))

    @app.post("/api/v1/predict/integration-failure", response_model=FailurePredictionResponse)
    def predict_integration(request: IntegrationFailureRequest) -> FailurePredictionResponse:
        return infer("integration", runtimes.integration_failure,
                     lambda runtime: runtime.predict(request))

    return app


# Import-time initialization is process startup, not request-time loading. Missing or
# incompatible configured artifacts yield honest degraded readiness.
app = create_app(load_configured_runtimes(ServiceConfig.from_environment()))
