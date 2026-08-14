"""Explicit configuration and trusted, contract-validating startup loading."""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from harbor_ml.capstone_anomaly import (
    CAPSTONE_ANOMALY_FEATURES, MODEL_NAME as ANOMALY_NAME,
    load_capstone_anomaly_artifact,
)
from harbor_ml.capstone_incident_classifier import (
    CAPSTONE_INCIDENT_CLASSES, CAPSTONE_INCIDENT_FEATURES, MODEL_NAME as INCIDENT_NAME,
    load_capstone_incident_artifact,
)
from harbor_ml.integration_failure_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from harbor_ml.training import MODEL_NAME as INTEGRATION_NAME, load_trusted_model_artifact

from .runtimes import (
    AnomalyModelRuntime, CapstoneModelRuntimes, IncidentModelRuntime,
    IntegrationFailureRuntime, ModelIdentity,
)

LOGGER = logging.getLogger("harbor_ml.service.loading")


@dataclass(frozen=True)
class ArtifactPaths:
    model: Path
    metadata: Path


@dataclass(frozen=True)
class ServiceConfig:
    anomaly: ArtifactPaths
    incident: ArtifactPaths
    integration_failure: ArtifactPaths

    @classmethod
    def from_environment(cls) -> "ServiceConfig":
        def paths(prefix: str, default: str) -> ArtifactPaths:
            return ArtifactPaths(
                Path(os.getenv(f"{prefix}_MODEL_PATH", f"{default}/model.joblib")),
                Path(os.getenv(f"{prefix}_METADATA_PATH", f"{default}/metadata.json")),
            )
        return cls(
            paths("HARBOR_CAPSTONE_ANOMALY", "artifacts/capstone-anomaly"),
            paths("HARBOR_CAPSTONE_INCIDENT", "artifacts/capstone-incident-classifier"),
            paths("HARBOR_INTEGRATION_FAILURE", "artifacts/integration-failure"),
        )


def _metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("could not load model metadata") from error
    if not isinstance(value, dict):
        raise RuntimeError("model metadata must be an object")
    return value


def _expect(metadata: dict[str, Any], **expected: Any) -> str:
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"model metadata has incompatible {key}")
    version = metadata.get("model_version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError("model metadata has invalid model_version")
    return version


def load_anomaly_runtime(paths: ArtifactPaths) -> AnomalyModelRuntime:
    metadata = _metadata(paths.metadata)
    version = _expect(metadata, model_name=ANOMALY_NAME, model_type="IsolationForest",
                      features=list(CAPSTONE_ANOMALY_FEATURES))
    model = load_capstone_anomaly_artifact(paths.model)
    return AnomalyModelRuntime(model, ModelIdentity(ANOMALY_NAME, version),
                               CAPSTONE_ANOMALY_FEATURES)


def load_incident_runtime(paths: ArtifactPaths) -> IncidentModelRuntime:
    metadata = _metadata(paths.metadata)
    version = _expect(metadata, model_name=INCIDENT_NAME,
                      model_type="StandardScaler+LogisticRegression",
                      features=list(CAPSTONE_INCIDENT_FEATURES),
                      class_taxonomy=list(CAPSTONE_INCIDENT_CLASSES))
    gap = metadata.get("ambiguity_gap_threshold")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not 0 <= gap <= 1:
        raise RuntimeError("model metadata has invalid ambiguity_gap_threshold")
    model = load_capstone_incident_artifact(paths.model)
    if tuple(str(value) for value in model.classes_) != tuple(sorted(CAPSTONE_INCIDENT_CLASSES)):
        raise RuntimeError("incident artifact has incompatible fitted classes")
    return IncidentModelRuntime(model, ModelIdentity(INCIDENT_NAME, version),
                                CAPSTONE_INCIDENT_FEATURES, float(gap))


def load_integration_runtime(paths: ArtifactPaths) -> IntegrationFailureRuntime:
    metadata = _metadata(paths.metadata)
    version = _expect(metadata, model_name=INTEGRATION_NAME,
                      numerical_features=list(NUMERIC_FEATURES),
                      categorical_features=list(CATEGORICAL_FEATURES), target="request_failed")
    threshold = metadata.get("classification_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise RuntimeError("model metadata has invalid classification_threshold")
    return IntegrationFailureRuntime(load_trusted_model_artifact(paths.model),
                                     ModelIdentity(INTEGRATION_NAME, version), float(threshold))


def load_configured_runtimes(config: ServiceConfig) -> CapstoneModelRuntimes:
    """Attempt every required model independently so safe capabilities can degrade."""
    def optional(name: str, loader: Callable[[ArtifactPaths], Any], paths: ArtifactPaths):
        try:
            return loader(paths)
        except Exception:
            LOGGER.exception("model_load_failed model=%s", name)
            return None
    return CapstoneModelRuntimes(
        optional(ANOMALY_NAME, load_anomaly_runtime, config.anomaly),
        optional(INCIDENT_NAME, load_incident_runtime, config.incident),
        optional(INTEGRATION_NAME, load_integration_runtime, config.integration_failure),
    )
