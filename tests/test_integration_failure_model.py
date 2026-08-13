from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from harbor_ml import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PREDICTION_FEATURES,
    IntegrationRequest,
    build_integration_features,
    build_integration_pipeline,
    build_integration_targets,
    evaluate_integration_model,
    load_integration_requests,
    predict_failure,
    predict_failure_probability,
    split_integration_dataset,
    train_integration_model,
)

DATASET = Path(__file__).parents[1] / "data/harbor_integration_requests.csv"
VENDORS = {
    "ClearVerify", "Northstar Payments", "HarborLink Core Gateway",
    "BlueCurrent Documents",
}
ENDPOINTS = {
    "identity_verify", "identity_document_upload", "transfer_submit",
    "transfer_status", "account_summary", "transaction_history",
    "statement_fetch", "notice_fetch",
}


@pytest.fixture(scope="module")
def observations():
    return load_integration_requests(DATASET)


@pytest.fixture(scope="module")
def fitted(observations):
    X = build_integration_features(observations)
    y = build_integration_targets(observations)
    split = split_integration_dataset(X, y)
    return train_integration_model(build_integration_pipeline(), split.X_train, split.y_train)


def sample_request(**changes):
    values = {
        "vendor": "ClearVerify", "endpoint": "identity_verify",
        "recent_vendor_latency_ms": 250, "recent_vendor_error_rate": 0.03,
        "queue_depth": 20, "retry_count": 0, "request_size_bytes": 1400,
        "hour_of_day": 11,
    }
    values.update(changes)
    return IntegrationRequest(**values)


def test_fixture_contains_expected_features_categories_and_binary_targets(observations):
    assert len(observations) == 600
    assert NUMERIC_FEATURES == (
        "recent_vendor_latency_ms", "recent_vendor_error_rate", "queue_depth",
        "retry_count", "request_size_bytes", "hour_of_day",
    )
    assert CATEGORICAL_FEATURES == ("vendor", "endpoint")
    assert {row.vendor for row in observations} == VENDORS
    assert {row.endpoint for row in observations} == ENDPOINTS
    assert set(build_integration_targets(observations)) == {0, 1}


def test_feature_matrix_uses_only_prediction_time_contract(observations):
    X = build_integration_features(observations)
    assert X.shape == (600, len(PREDICTION_FEATURES))
    assert "request_failed" not in PREDICTION_FEATURES
    assert not {"final_http_status", "failure_reason", "response_duration_ms"} & set(PREDICTION_FEATURES)
    assert list(X[0]) == [getattr(observations[0], name) for name in PREDICTION_FEATURES]


def test_split_is_deterministic_and_stratified(observations):
    X, y = build_integration_features(observations), build_integration_targets(observations)
    first, second = split_integration_dataset(X, y), split_integration_dataset(X, y)
    assert np.array_equal(first.X_train, second.X_train)
    assert np.array_equal(first.y_test, second.y_test)
    assert set(first.y_train) == set(first.y_test) == {0, 1}


def test_pipeline_constructs_and_one_hot_encodes_during_fit(fitted):
    assert isinstance(fitted, Pipeline)
    assert isinstance(fitted.named_steps["preprocessor"], ColumnTransformer)
    transformed = fitted.named_steps["preprocessor"].transform(
        np.asarray([[250, 0.03, 20, 0, 1400, 11, "ClearVerify", "identity_verify"]], dtype=object)
    )
    assert transformed.shape[1] > len(PREDICTION_FEATURES)
    encoder = fitted.named_steps["preprocessor"].named_transformers_["categorical"]
    assert encoder.handle_unknown == "ignore"


def test_fit_evaluate_probabilities_and_binary_predictions(observations, fitted):
    X, y = build_integration_features(observations), build_integration_targets(observations)
    split = split_integration_dataset(X, y)
    evaluation = evaluate_integration_model(fitted, split.X_test, split.y_test)
    assert 0 <= evaluation.accuracy <= 1
    assert evaluation.confusion_matrix.shape == (2, 2)
    assert set(evaluation.predictions) <= {0, 1}
    probabilities = fitted.predict_proba(split.X_test)
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    assert np.sum(probabilities, axis=1) == pytest.approx(np.ones(len(probabilities)))


def test_probability_and_custom_threshold_are_separate(fitted):
    request = sample_request(retry_count=2, queue_depth=70)
    probability = predict_failure_probability(fitted, request)
    predictions = [predict_failure(fitted, request, threshold=value) for value in (0.3, 0.5, 0.7)]
    assert all(item.probability == probability for item in predictions)
    assert all(item.predicted_failure == (probability >= item.threshold) for item in predictions)


def test_unseen_category_does_not_crash(fitted):
    probability = predict_failure_probability(
        fitted, sample_request(vendor="Harbor Experimental Sandbox", endpoint="new_endpoint")
    )
    assert 0 <= probability <= 1


@pytest.mark.parametrize(
    "candidate, message",
    [
        ({}, "missing required"),
        ({**asdict(sample_request()), "request_failed": 1}, "unexpected"),
        ({**asdict(sample_request()), "queue_depth": -1}, "non-negative"),
        ({**asdict(sample_request()), "vendor": ""}, "non-empty"),
        ({**asdict(sample_request()), "hour_of_day": 24}, "outside valid"),
    ],
)
def test_malformed_prediction_input_is_rejected(fitted, candidate, message):
    with pytest.raises(ValueError, match=message):
        predict_failure_probability(fitted, candidate)


def test_malformed_fixture_is_rejected(tmp_path):
    malformed = tmp_path / "bad.csv"
    malformed.write_text(
        "timestamp,vendor,endpoint,recent_vendor_latency_ms,recent_vendor_error_rate,queue_depth,retry_count,request_size_bytes,hour_of_day,request_failed\n"
        "not-a-date,ClearVerify,identity_verify,200,0.02,3,0,1000,10,2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid request_failed"):
        load_integration_requests(malformed)


def test_invalid_threshold_is_rejected(fitted):
    with pytest.raises(ValueError, match="between 0 and 1"):
        predict_failure(fitted, sample_request(), threshold=1.1)
