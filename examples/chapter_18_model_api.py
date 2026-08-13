"""Run Chapter 18's in-process model API laboratory."""

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from harbor_ml.api import create_app  # noqa: E402
from harbor_ml.training import (  # noqa: E402
    save_model_artifact, save_training_metadata, train_integration_failure_model,
)

REQUEST = {
    "vendor": "ClearVerify", "endpoint": "identity_verify",
    "recent_vendor_latency_ms": 940, "recent_vendor_error_rate": 0.031,
    "queue_depth": 42, "retry_count": 1, "request_size_bytes": 2400,
    "hour_of_day": 14,
}


def show(label: str, response) -> None:
    print(f"\n{label}\n\n{response.status_code}\n{json.dumps(response.json(), indent=2)}")


def main() -> None:
    print("Harbor Federal Credit Union\nML Prediction API Laboratory")
    result = train_integration_failure_model(ROOT / "data/harbor_integration_requests.csv")
    with tempfile.TemporaryDirectory(prefix="harbor-ch18-") as directory:
        model = save_model_artifact(result.pipeline, directory)
        metadata = save_training_metadata(result.metadata, directory)
        client = TestClient(create_app(model_path=model, metadata_path=metadata))
        show("GET /api/v1/health", client.get("/api/v1/health"))
        show("POST /api/v1/predict/integration-failure", client.post(
            "/api/v1/predict/integration-failure", json=REQUEST
        ))
        show("Invalid request", client.post(
            "/api/v1/predict/integration-failure", json=REQUEST | {"hour_of_day": 28}
        ))
        show("Unknown vendor (accepted, not necessarily understood)", client.post(
            "/api/v1/predict/integration-failure", json=REQUEST | {"vendor": "NovelVendor"}
        ))
    print("\nAll calls ran in process: no TCP server or committed binary artifact was needed.")


if __name__ == "__main__":
    main()
