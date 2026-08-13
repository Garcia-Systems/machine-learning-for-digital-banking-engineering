"""A small, deterministic supervised-learning pipeline for Chapter 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .problem_framing import REQUEST_FAILURE, RequestOutcome

FEATURE_NAMES = REQUEST_FAILURE.features
RANDOM_STATE = 42

FeatureMatrix = NDArray[np.float64]
TargetVector = NDArray[np.int64]
RequestFeatures = Mapping[str, int | float]


@dataclass(frozen=True)
class DatasetSplit:
    """Training examples and held-out examples from one deterministic split."""

    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector


@dataclass(frozen=True)
class EvaluationResult:
    """The deliberately small set of evaluation results introduced in Chapter 3."""

    accuracy: float
    correct: int
    incorrect: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    predictions: TargetVector


@dataclass(frozen=True)
class FailurePrediction:
    """A class and failure probability produced for one new request."""

    predicted_class: int
    failure_probability: float


def build_feature_matrix(outcomes: Sequence[RequestOutcome]) -> FeatureMatrix:
    """Select Chapter 3 features in the problem definition's declared order."""
    return np.asarray(
        [[getattr(outcome, name) for name in FEATURE_NAMES] for outcome in outcomes],
        dtype=np.float64,
    ).reshape(len(outcomes), len(FEATURE_NAMES))


def build_target_vector(outcomes: Sequence[RequestOutcome]) -> TargetVector:
    """Extract the known historical binary outcome from every observation."""
    return np.asarray(
        [outcome.request_failed for outcome in outcomes], dtype=np.int64
    )


def split_dataset(
    X: FeatureMatrix,
    y: TargetVector,
    *,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Create a reproducible, class-stratified training/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)


def build_model() -> Pipeline:
    """Build a scaling and logistic-regression pipeline without fitting it."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(random_state=RANDOM_STATE, max_iter=1_000),
            ),
        ]
    )


def train_model(
    model: Pipeline, X_train: FeatureMatrix, y_train: TargetVector
) -> Pipeline:
    """Fit model parameters using examples with known outcomes."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline, X_test: FeatureMatrix, y_test: TargetVector
) -> EvaluationResult:
    """Evaluate fitted-model behavior on examples held out from training."""
    predictions = np.asarray(model.predict(X_test), dtype=np.int64)
    correct = int(np.count_nonzero(predictions == y_test))
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return EvaluationResult(
        accuracy=correct / len(y_test),
        correct=correct,
        incorrect=len(y_test) - correct,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        predictions=predictions,
    )


def _validated_request_row(request: RequestFeatures) -> FeatureMatrix:
    missing = [name for name in FEATURE_NAMES if name not in request]
    unexpected = [name for name in request if name not in FEATURE_NAMES]
    if missing:
        raise ValueError("missing required request features: " + ", ".join(missing))
    if unexpected:
        raise ValueError("unexpected request features: " + ", ".join(unexpected))

    values: list[float] = []
    for name in FEATURE_NAMES:
        value = request[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        numeric_value = float(value)
        if not isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        values.append(numeric_value)
    return np.asarray([values], dtype=np.float64)


def predict_request_failure(
    model: Pipeline, request: RequestFeatures
) -> FailurePrediction:
    """Apply a fitted model to one validated prediction-time observation."""
    row = _validated_request_row(request)
    predicted_class = int(model.predict(row)[0])
    failure_probability = float(model.predict_proba(row)[0, 1])
    return FailurePrediction(predicted_class, failure_probability)
