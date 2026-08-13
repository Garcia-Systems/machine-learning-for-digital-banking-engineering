from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from harbor_ml.integration_failure_model import IntegrationRequest, predict_failure_probability
from harbor_ml.training import (
    TrainingConfig, TrainingResult, calculate_file_sha256, load_trusted_model_artifact,
    save_model_artifact, save_training_metadata, train_integration_failure_model,
    validate_training_dataset,
)

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data/harbor_integration_requests.csv"


def test_validation_accepts_fixture_and_hash_is_deterministic():
    assert len(validate_training_dataset(DATASET)) == 600
    assert calculate_file_sha256(DATASET) == calculate_file_sha256(DATASET)
    assert len(calculate_file_sha256(DATASET)) == 64


@pytest.mark.parametrize("change, message", [
    ("missing", "missing required"), ("empty", "header and observations"),
    ("target", "invalid request_failed"),
])
def test_invalid_datasets_fail(tmp_path, change, message):
    path = tmp_path / "data.csv"
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    if change == "missing":
        lines = [line.replace(",endpoint,", ",") for line in lines]
    elif change == "empty":
        lines = []
    else:
        cells = lines[1].split(",")
        cells[-1] = "maybe"
        lines[1] = ",".join(cells)
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_training_dataset(path)


@pytest.mark.parametrize("kwargs", [
    {"test_size": 0}, {"test_size": 1}, {"classification_threshold": -0.1},
    {"classification_threshold": 1.1}, {"max_iter": 0},
])
def test_training_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


@pytest.fixture(scope="module")
def result():
    def fixed():
        return datetime(2026, 8, 13, 12, tzinfo=timezone.utc)

    return train_integration_failure_model(DATASET, now=fixed)


def test_training_result_pipeline_metadata_and_baseline(result):
    assert isinstance(result, TrainingResult)
    assert isinstance(result.pipeline, Pipeline)
    assert set(result.pipeline.named_steps) == {"preprocessor", "classifier"}
    assert hasattr(result.pipeline.named_steps["preprocessor"], "transformers_")
    assert result.training_rows + result.test_rows == 600
    assert result.metadata.trained_at == "2026-08-13T12:00:00Z"
    assert result.metadata.dataset_sha256 == result.dataset_sha256
    assert "baseline_accuracy" in result.metadata.metrics
    assert all(np.isfinite(list(asdict(result.metrics).values())))
    json.dumps(asdict(result.metadata))


def test_artifact_and_metadata_round_trip_in_new_output_directory(tmp_path, result):
    output = tmp_path / "nested" / "integration-failure"
    model_path = save_model_artifact(result.pipeline, output)
    metadata_path = save_training_metadata(result.metadata, output)
    assert model_path.is_file() and metadata_path.is_file()
    assert json.loads(metadata_path.read_text())["model_version"].endswith(result.dataset_sha256[:8])
    request = IntegrationRequest("ClearVerify", "identity_verify", 250, .03, 20, 0, 1400, 11)
    before = predict_failure_probability(result.pipeline, request)
    loaded = load_trusted_model_artifact(model_path)
    assert set(loaded.named_steps) == {"preprocessor", "classifier"}
    assert predict_failure_probability(loaded, request) == before


def test_same_data_and_configuration_reproduce_metrics():
    first = train_integration_failure_model(DATASET, TrainingConfig(random_state=17))
    second = train_integration_failure_model(DATASET, TrainingConfig(random_state=17))
    assert first.metrics == second.metrics


def test_repository_does_not_contain_committed_model_artifacts():
    assert not list(ROOT.glob("**/*.joblib"))
    assert "artifacts/" in (ROOT / ".gitignore").read_text()
