"""Run Chapter 30's complete multi-model API laboratory in process."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from harbor_ml.capstone_anomaly import (  # noqa: E402
    build_capstone_anomaly_timeline, create_anomaly_metadata, evaluate_detection_behavior,
    save_capstone_anomaly_artifact, select_anomaly_baseline, train_capstone_anomaly_detector,
)
from harbor_ml.capstone_dataset import load_capstone_sources  # noqa: E402
from harbor_ml.capstone_incident_classifier import (  # noqa: E402
    create_capstone_incident_metadata, evaluate_capstone_incident_classifier,
    load_capstone_classification_data, save_capstone_incident_artifact,
    split_capstone_incident_data, train_capstone_incident_classifier,
)
from harbor_ml.service.app import create_app  # noqa: E402
from harbor_ml.service.artifact_loader import (  # noqa: E402
    ArtifactPaths, ServiceConfig, load_configured_runtimes,
)
from harbor_ml.training import (  # noqa: E402
    save_model_artifact, save_training_metadata, train_integration_failure_model,
)

HEALTHY = {"api_latency_ms": 185, "error_rate": .003, "db_connections": 36,
    "queue_depth": 8, "vendor_latency_ms": 245, "requests_per_minute": 430, "retry_count": 0}
DEGRADED = {"api_latency_ms": 1480, "error_rate": .047, "db_connections": 78,
    "queue_depth": 96, "vendor_latency_ms": 1320, "requests_per_minute": 760, "retry_count": 3}
INTEGRATION = {"vendor": "ClearVerify", "endpoint": "identity_verify",
    "recent_vendor_latency_ms": 940, "recent_vendor_error_rate": .031,
    "queue_depth": 42, "retry_count": 1, "request_size_bytes": 2400, "hour_of_day": 14}


def build_runtimes(directory: Path):
    def now():
        return datetime(2026, 8, 14, tzinfo=timezone.utc)
    timeline = build_capstone_anomaly_timeline(load_capstone_sources(ROOT / "data"))
    baseline = select_anomaly_baseline(timeline)
    anomaly_model = train_capstone_anomaly_detector(baseline)
    anomaly_meta = create_anomaly_metadata(baseline, timeline, evaluate_detection_behavior(tuple()), now=now)
    anomaly = save_capstone_anomaly_artifact(anomaly_model, anomaly_meta, directory / "anomaly")
    data = load_capstone_classification_data(ROOT / "data/harbor_incident_classes.csv")
    split = split_capstone_incident_data(data)
    incident_model = train_capstone_incident_classifier(split)
    incident_meta = create_capstone_incident_metadata(
        data, split, evaluate_capstone_incident_classifier(incident_model, split), now=now)
    incident = save_capstone_incident_artifact(incident_model, incident_meta, directory / "incident")
    result = train_integration_failure_model(ROOT / "data/harbor_integration_requests.csv", now=now)
    integration = (save_model_artifact(result.pipeline, directory / "integration"),
                   save_training_metadata(result.metadata, directory / "integration"))
    return load_configured_runtimes(ServiceConfig(
        ArtifactPaths(*anomaly), ArtifactPaths(*incident), ArtifactPaths(*integration)))


def call(client, method, route, payload=None):
    response = getattr(client, method)(route, json=payload) if payload else getattr(client, method)(route)
    print(f"\n{method.upper()} {route}\n{response.status_code}\n{json.dumps(response.json(), indent=2)}")
    return response


def main() -> None:
    print("Harbor Federal Credit Union\nCapstone ML Service Laboratory")
    with tempfile.TemporaryDirectory(prefix="harbor-ch30-") as temporary:
        runtimes = build_runtimes(Path(temporary))
        client = TestClient(create_app(runtimes))
        call(client, "get", "/api/v1/health")
        call(client, "post", "/api/v1/score/telemetry-anomaly", HEALTHY)
        call(client, "post", "/api/v1/score/telemetry-anomaly", DEGRADED)
        call(client, "post", "/api/v1/predict/incident", DEGRADED)
        call(client, "post", "/api/v1/predict/integration-failure", INTEGRATION)
        print("\nSimulating missing incident model...")
        degraded = TestClient(create_app(replace(runtimes, incident=None)))
        call(degraded, "get", "/api/v1/health")
        call(degraded, "post", "/api/v1/predict/incident", DEGRADED)
        call(degraded, "post", "/api/v1/score/telemetry-anomaly", DEGRADED)
        call(degraded, "post", "/api/v1/score/telemetry-anomaly", DEGRADED | {"error_rate": 4.2})


if __name__ == "__main__":
    main()
