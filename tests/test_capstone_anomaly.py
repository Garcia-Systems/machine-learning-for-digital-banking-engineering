from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from harbor_ml.capstone_anomaly import (
    CAPSTONE_ANOMALY_FEATURES, PROHIBITED_ANOMALY_FIELDS,
    build_anomaly_feature_matrix, build_capstone_anomaly_timeline,
    create_anomaly_metadata, evaluate_detection_behavior, load_capstone_anomaly_artifact,
    save_capstone_anomaly_artifact, score_capstone_timeline, select_anomaly_baseline,
    train_capstone_anomaly_detector, validate_anomaly_baseline,
)
from harbor_ml.capstone_dataset import load_capstone_sources

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def timeline():
    return build_capstone_anomaly_timeline(load_capstone_sources(ROOT / "data"))


def test_feature_contract_excludes_labels_targets_and_sensitive_fields(timeline):
    assert CAPSTONE_ANOMALY_FEATURES == (
        "api_latency_ms", "error_rate", "db_connections", "queue_depth",
        "vendor_latency_ms", "requests_per_minute", "retry_count")
    assert not (PROHIBITED_ANOMALY_FIELDS & set(CAPSTONE_ANOMALY_FEATURES))
    matrix = build_anomaly_feature_matrix(timeline)
    assert matrix.shape == (20, 7)
    assert np.isfinite(matrix).all()


def test_baseline_selection_is_healthy_and_chronologically_before_incident(timeline):
    baseline = select_anomaly_baseline(timeline)
    incident = tuple(row for row in timeline if row.phase != "healthy")
    assert len(baseline) == 3
    assert {row.phase for row in baseline} == {"healthy"}
    assert max(row.timestamp for row in baseline) < min(row.timestamp for row in incident)
    validate_anomaly_baseline(baseline, evaluation=incident)
    with pytest.raises(ValueError, match="known degraded"):
        validate_anomaly_baseline((*baseline, incident[0]), minimum_rows=3)
    with pytest.raises(ValueError, match="at least 4"):
        validate_anomaly_baseline(baseline, minimum_rows=4)


def test_fit_scores_orientation_evaluation_and_reproducibility(timeline):
    baseline = select_anomaly_baseline(timeline)
    first = train_capstone_anomaly_detector(baseline)
    second = train_capstone_anomaly_detector(baseline)
    results = score_capstone_timeline(first, timeline)
    repeated = score_capstone_timeline(second, timeline)
    assert np.isfinite([row.anomaly_score for row in results]).all()
    assert np.allclose([row.anomaly_score for row in results],
                       [row.anomaly_score for row in repeated])
    raw = first.decision_function(build_anomaly_feature_matrix(timeline))
    assert np.allclose([row.anomaly_score for row in results], -raw)
    behavior = evaluate_detection_behavior(
        results, reference_incident_timestamp=datetime(2026, 8, 12, 10, 20, tzinfo=timezone.utc))
    assert behavior.healthy_eval_rows > 0
    assert behavior.first_anomaly_timestamp == next(
        row.timestamp for row in results if row.phase != "healthy" and row.is_anomaly)
    assert behavior.lead_time_seconds == (
        behavior.reference_incident_timestamp - behavior.first_anomaly_timestamp).total_seconds()


def test_artifact_metadata_and_round_trip(timeline, tmp_path):
    baseline = select_anomaly_baseline(timeline)
    model = train_capstone_anomaly_detector(baseline)
    before = score_capstone_timeline(model, timeline)
    behavior = evaluate_detection_behavior(before)
    metadata = create_anomaly_metadata(baseline, timeline, behavior,
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc))
    model_path, metadata_path = save_capstone_anomaly_artifact(model, metadata, tmp_path)
    after = score_capstone_timeline(load_capstone_anomaly_artifact(model_path), timeline)
    assert np.allclose([row.anomaly_score for row in before],
                       [row.anomaly_score for row in after])
    assert metadata_path.exists()
    assert metadata.model_version.startswith("harbor-capstone-anomaly-")
    assert metadata.dataset_sha256 and metadata.baseline_rows == 3
