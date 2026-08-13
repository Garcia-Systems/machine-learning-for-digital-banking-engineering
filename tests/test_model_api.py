import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from harbor_ml.api import ModelRuntime, create_app
from harbor_ml.training import (
    save_model_artifact,
    save_training_metadata,
    train_integration_failure_model,
)

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data/harbor_integration_requests.csv"
VALID = {
    "vendor": "ClearVerify",
    "endpoint": "identity_verify",
    "recent_vendor_latency_ms": 940,
    "recent_vendor_error_rate": 0.031,
    "queue_depth": 42,
    "retry_count": 1,
    "request_size_bytes": 2400,
    "hour_of_day": 14,
}


@pytest.fixture(scope="module")
def trained_result():
    return train_integration_failure_model(DATASET)


@pytest.fixture
def assets(tmp_path, trained_result):
    return (
        save_model_artifact(trained_result.pipeline, tmp_path),
        save_training_metadata(trained_result.metadata, tmp_path),
    )


def test_app_loads_model_once_and_health_reports_metadata(assets):
    with patch("harbor_ml.api.load_trusted_model_artifact", wraps=__import__(
        "harbor_ml.api", fromlist=["load_trusted_model_artifact"]
    ).load_trusted_model_artifact) as loader:
        app = create_app(model_path=assets[0], metadata_path=assets[1])
        client = TestClient(app)
        assert client.get("/api/v1/health").json() == {
            "status": "ok", "model_loaded": True,
            "model_name": "harbor-integration-failure",
            "model_version": app.state.model_runtime.model_version,
        }
        client.post("/api/v1/predict/integration-failure", json=VALID)
        client.post("/api/v1/predict/integration-failure", json=VALID)
        loader.assert_called_once()


def test_valid_prediction_is_actual_deterministic_thresholded_probability(assets):
    client = TestClient(create_app(model_path=assets[0], metadata_path=assets[1]))
    first = client.post("/api/v1/predict/integration-failure", json=VALID)
    second = client.post("/api/v1/predict/integration-failure", json=VALID)
    assert first.status_code == 200 and second.json() == first.json()
    body = first.json()
    assert set(body) == {"model", "model_version", "failure_probability", "threshold", "predicted_failure"}
    assert 0 <= body["failure_probability"] <= 1
    assert body["predicted_failure"] is (body["failure_probability"] >= body["threshold"])


@pytest.mark.parametrize("change", [
    {"hour_of_day": 30}, {"recent_vendor_latency_ms": "slow"}, {"request_size_bytes": 0},
])
def test_invalid_numerical_requests_are_rejected(assets, change):
    client = TestClient(create_app(model_path=assets[0], metadata_path=assets[1]))
    assert client.post("/api/v1/predict/integration-failure", json=VALID | change).status_code == 422


def test_missing_field_is_rejected(assets):
    client = TestClient(create_app(model_path=assets[0], metadata_path=assets[1]))
    invalid = VALID.copy(); invalid.pop("vendor")
    assert client.post("/api/v1/predict/integration-failure", json=invalid).status_code == 422


def test_unknown_category_is_accepted_by_fitted_encoder(assets):
    client = TestClient(create_app(model_path=assets[0], metadata_path=assets[1]))
    response = client.post("/api/v1/predict/integration-failure", json=VALID | {"vendor": "NewVendor"})
    assert response.status_code == 200
    assert 0 <= response.json()["failure_probability"] <= 1


def test_missing_artifact_fails_app_creation_clearly(tmp_path, trained_result):
    metadata = save_training_metadata(trained_result.metadata, tmp_path)
    with pytest.raises(RuntimeError, match="could not load trusted model artifact"):
        create_app(model_path=tmp_path / "missing.joblib", metadata_path=metadata)


def test_inference_error_is_controlled_and_does_not_leak_traceback(assets):
    runtime = ModelRuntime.load(*assets)
    with patch.object(runtime.pipeline, "predict_proba", side_effect=RuntimeError("SECRET TRACE")):
        response = TestClient(create_app(runtime)).post(
            "/api/v1/predict/integration-failure", json=VALID
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "prediction could not be calculated"}
    assert "SECRET TRACE" not in response.text and "Traceback" not in response.text


def test_prediction_endpoint_never_fits(assets):
    runtime = ModelRuntime.load(*assets)
    with patch.object(runtime.pipeline, "fit", side_effect=AssertionError("fit called")) as fit:
        response = TestClient(create_app(runtime)).post(
            "/api/v1/predict/integration-failure", json=VALID
        )
    assert response.status_code == 200
    fit.assert_not_called()


def test_metadata_feature_contract_is_validated(assets, tmp_path):
    metadata = json.loads(assets[1].read_text())
    metadata["categorical_features"] = ["vendor"]
    path = tmp_path / "bad-metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="incompatible categorical_features"):
        ModelRuntime.load(assets[0], path)
