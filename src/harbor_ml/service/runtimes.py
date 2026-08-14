"""Effectively read-only fitted model runtimes used by request handlers."""

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

import numpy as np

from harbor_ml.capstone_incident_classifier import predict_incident_probabilities
from harbor_ml.integration_failure_model import IntegrationRequest, predict_failure_probability

from .schemas import (
    AnomalyResponse, CapstoneTelemetryRequest, FailurePredictionResponse,
    IncidentPredictionResponse, IntegrationFailureRequest,
)

FEATURE_CONTRACT_VERSION = "capstone-telemetry-v1"


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    version: str


@dataclass(frozen=True)
class AnomalyModelRuntime:
    model: Any
    identity: ModelIdentity
    feature_names: tuple[str, ...]

    def score(self, request: CapstoneTelemetryRequest) -> AnomalyResponse:
        values = request.model_dump()
        row = np.asarray([[values[name] for name in self.feature_names]], dtype=np.float64)
        score = -float(self.model.decision_function(row)[0])
        prediction = int(self.model.predict(row)[0])
        if not isfinite(score):
            raise ValueError("anomaly model returned a non-finite score")
        return AnomalyResponse(model=self.identity.name, model_version=self.identity.version,
                               anomaly_score=score, is_anomaly=prediction == -1)


@dataclass(frozen=True)
class IncidentModelRuntime:
    model: Any
    identity: ModelIdentity
    feature_names: tuple[str, ...]
    ambiguity_gap: float

    def predict(self, request: CapstoneTelemetryRequest) -> IncidentPredictionResponse:
        values = request.model_dump()
        result = predict_incident_probabilities(
            self.model, {name: values[name] for name in self.feature_names},
            ambiguity_gap=self.ambiguity_gap,
        )
        probabilities = result.probabilities
        if (not all(isfinite(value) and 0 <= value <= 1 for value in probabilities.values())
                or not isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6)):
            raise ValueError("incident model returned invalid probabilities")
        return IncidentPredictionResponse(
            model=self.identity.name, model_version=self.identity.version,
            predicted_class=result.predicted_class, probabilities=probabilities,
            top_probability=result.top_probability, second_probability=result.second_probability,
            probability_gap=result.probability_gap, ambiguous=result.ambiguous,
        )


@dataclass(frozen=True)
class IntegrationFailureRuntime:
    """Adapter around Chapter 18's fitted integration-failure model semantics."""

    model: Any
    identity: ModelIdentity
    threshold: float

    def predict(self, request: IntegrationFailureRequest) -> FailurePredictionResponse:
        probability = predict_failure_probability(
            self.model, IntegrationRequest(**request.model_dump()))
        return FailurePredictionResponse(
            model=self.identity.name, model_version=self.identity.version,
            failure_probability=probability, threshold=self.threshold,
            predicted_failure=probability >= self.threshold,
        )


@dataclass(frozen=True)
class CapstoneModelRuntimes:
    anomaly: AnomalyModelRuntime | None
    incident: IncidentModelRuntime | None
    integration_failure: IntegrationFailureRuntime | None

    @property
    def ready(self) -> bool:
        return all((self.anomaly, self.incident, self.integration_failure))
