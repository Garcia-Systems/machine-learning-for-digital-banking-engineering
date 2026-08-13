from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from harbor_ml import (
    INCIDENT_CLASSES, INCIDENT_FEATURES, IncidentPrediction,
    build_incident_classifier, build_incident_features, build_incident_targets,
    evaluate_incident_classifier, load_incident_dataset, predict_incident,
    predict_incident_probabilities, split_incident_dataset, train_incident_classifier,
)

DATASET = Path(__file__).parents[1] / "data/harbor_incident_classes.csv"


@pytest.fixture
def observations():
    return load_incident_dataset(DATASET)


@pytest.fixture
def split(observations):
    return split_incident_dataset(
        build_incident_features(observations), build_incident_targets(observations)
    )


@pytest.fixture
def fitted(split):
    return train_incident_classifier(
        build_incident_classifier(), split.X_train, split.y_train
    )


def test_dataset_loading_features_labels_and_class_representation(observations) -> None:
    assert len(observations) == 300
    assert INCIDENT_FEATURES == (
        "api_latency_ms", "error_rate", "db_connections", "queue_depth",
        "vendor_latency_ms", "requests_per_minute",
    )
    X = build_incident_features(observations)
    y = build_incident_targets(observations)
    assert X.shape == (300, 6)
    assert set(y) == set(INCIDENT_CLASSES)
    assert {label: int(np.count_nonzero(y == label)) for label in INCIDENT_CLASSES} == {
        label: 60 for label in INCIDENT_CLASSES
    }
    assert np.isfinite(X).all()


def test_split_is_deterministic_stratified_and_contains_every_class(observations) -> None:
    X, y = build_incident_features(observations), build_incident_targets(observations)
    first, second = split_incident_dataset(X, y), split_incident_dataset(X, y)
    assert np.array_equal(first.X_train, second.X_train)
    assert np.array_equal(first.y_test, second.y_test)
    assert first.X_train.shape == (225, 6)
    assert first.X_test.shape == (75, 6)
    assert set(first.y_train) == set(first.y_test) == set(INCIDENT_CLASSES)


def test_model_fits_predicts_known_classes_and_evaluates(split, fitted) -> None:
    assert isinstance(fitted, Pipeline)
    assert set(fitted.classes_) == set(INCIDENT_CLASSES)
    evaluation = evaluate_incident_classifier(fitted, split.X_test, split.y_test)
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert evaluation.confusion_matrix.shape == (5, 5)
    assert evaluation.confusion_matrix.sum() == len(split.y_test)
    assert set(evaluation.predictions) <= set(INCIDENT_CLASSES)


def test_scenario_probability_api_maps_model_classes_and_sums_to_one(fitted) -> None:
    scenario = {
        "api_latency_ms": 1840, "error_rate": 0.061, "db_connections": 58,
        "queue_depth": 112, "vendor_latency_ms": 1720,
        "requests_per_minute": 710,
    }
    result = predict_incident(fitted, scenario)
    assert isinstance(result, IncidentPrediction)
    assert result.predicted_class in INCIDENT_CLASSES
    assert set(result.probabilities) == set(INCIDENT_CLASSES) == set(fitted.classes_)
    assert all(np.isfinite(list(result.probabilities.values())))
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    row = np.asarray([[scenario[name] for name in INCIDENT_FEATURES]])
    expected = dict(zip(fitted.classes_, fitted.predict_proba(row)[0], strict=True))
    assert result.probabilities == pytest.approx(expected)
    assert predict_incident_probabilities(fitted, scenario) == result.probabilities


@pytest.mark.parametrize("scenario, message", [
    ({"api_latency_ms": 100}, "missing required incident features"),
    ({"api_latency_ms": 100, "error_rate": .01, "db_connections": 30,
      "queue_depth": 10, "vendor_latency_ms": 200, "requests_per_minute": 500,
      "incident_type": "normal"}, "unexpected incident features"),
    ({"api_latency_ms": float("nan"), "error_rate": .01, "db_connections": 30,
      "queue_depth": 10, "vendor_latency_ms": 200, "requests_per_minute": 500},
     "api_latency_ms"),
])
def test_prediction_rejects_malformed_features(fitted, scenario, message) -> None:
    with pytest.raises(ValueError, match=message):
        predict_incident(fitted, scenario)


def test_training_rejects_wrong_or_nonfinite_features(split) -> None:
    with pytest.raises(ValueError, match="6 columns"):
        train_incident_classifier(build_incident_classifier(), np.ones((5, 5)), split.y_train[:5])
    with pytest.raises(ValueError, match="finite observations"):
        train_incident_classifier(build_incident_classifier(), np.full((5, 6), np.nan), split.y_train[:5])
