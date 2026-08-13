from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from harbor_ml import (
    FEATURE_NAMES,
    EvaluationResult,
    build_feature_matrix,
    build_model,
    build_target_vector,
    evaluate_model,
    load_request_outcomes,
    predict_request_failure,
    split_dataset,
    train_model,
)

FIXTURE = Path(__file__).parents[1] / "data/harbor_request_outcomes.csv"
NEW_REQUEST = {
    "vendor_latency_ms": 1650,
    "queue_depth": 103,
    "db_connections": 74,
    "retry_count": 3,
}


@pytest.fixture
def prepared_data():
    outcomes = load_request_outcomes(FIXTURE)
    return outcomes, build_feature_matrix(outcomes), build_target_vector(outcomes)


@pytest.fixture
def fitted_model(prepared_data):
    _, X, y = prepared_data
    split = split_dataset(X, y)
    return train_model(build_model(), split.X_train, split.y_train), split


def test_loading_and_extraction_preserve_feature_order(prepared_data) -> None:
    outcomes, X, y = prepared_data

    assert len(outcomes) == 30
    assert FEATURE_NAMES == (
        "vendor_latency_ms",
        "queue_depth",
        "db_connections",
        "retry_count",
    )
    assert X.shape == (30, 4)
    assert X[0].tolist() == [205.0, 9.0, 28.0, 0.0]
    assert y.shape == (30,)
    assert y.tolist() == [outcome.request_failed for outcome in outcomes]


def test_split_is_deterministic_and_stratified(prepared_data) -> None:
    _, X, y = prepared_data
    first = split_dataset(X, y)
    second = split_dataset(X, y)

    assert len(first.y_train) == 22
    assert len(first.y_test) == 8
    assert np.array_equal(first.X_train, second.X_train)
    assert np.array_equal(first.X_test, second.X_test)
    assert np.array_equal(first.y_train, second.y_train)
    assert set(first.y_train) == {0, 1}
    assert set(first.y_test) == {0, 1}


def test_model_has_scaler_and_logistic_classifier() -> None:
    model = build_model()

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)


def test_model_fits_and_predicts_expected_shapes(fitted_model) -> None:
    model, split = fitted_model
    predictions = model.predict(split.X_test)
    probabilities = model.predict_proba(split.X_test)

    assert predictions.shape == split.y_test.shape
    assert predictions.dtype.kind in "iu"
    assert probabilities.shape == (len(split.y_test), 2)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_evaluation_reports_consistent_counts(fitted_model) -> None:
    model, split = fitted_model
    result = evaluate_model(model, split.X_test, split.y_test)

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.correct + result.incorrect == len(split.y_test)
    assert (
        result.true_negatives
        + result.false_positives
        + result.false_negatives
        + result.true_positives
        == len(split.y_test)
    )
    assert result.predictions.shape == split.y_test.shape


def test_predicts_one_valid_new_observation(fitted_model) -> None:
    model, _ = fitted_model
    result = predict_request_failure(model, NEW_REQUEST)

    assert result.predicted_class in (0, 1)
    assert isinstance(result.predicted_class, int)
    assert 0.0 <= result.failure_probability <= 1.0
    assert isinstance(result.failure_probability, float)


@pytest.mark.parametrize(
    "candidate, message",
    [
        (
            {"vendor_latency_ms": 100, "queue_depth": 10, "db_connections": 20},
            "missing required request features: retry_count",
        ),
        ({**NEW_REQUEST, "outcome": 1}, "unexpected request features: outcome"),
        ({**NEW_REQUEST, "queue_depth": -1}, "queue_depth"),
        ({**NEW_REQUEST, "retry_count": "three"}, "retry_count"),
        ({**NEW_REQUEST, "vendor_latency_ms": float("nan")}, "vendor_latency_ms"),
    ],
)
def test_rejects_malformed_new_observations(
    fitted_model, candidate, message: str
) -> None:
    model, _ = fitted_model
    with pytest.raises(ValueError, match=message):
        predict_request_failure(model, candidate)
