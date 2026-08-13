"""Request-level integration failure classification for Chapter 7."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = (
    "recent_vendor_latency_ms",
    "recent_vendor_error_rate",
    "queue_depth",
    "retry_count",
    "request_size_bytes",
    "hour_of_day",
)
CATEGORICAL_FEATURES = ("vendor", "endpoint")
PREDICTION_FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
TARGET = "request_failed"
INTEGRATION_RANDOM_STATE = 42

FeatureMatrix = NDArray[np.object_]
TargetVector = NDArray[np.int64]


@dataclass(frozen=True)
class IntegrationObservation:
    """One historical request with request-time values and its later outcome."""

    timestamp: datetime
    vendor: str
    endpoint: str
    recent_vendor_latency_ms: float
    recent_vendor_error_rate: float
    queue_depth: int
    retry_count: int
    request_size_bytes: int
    hour_of_day: int
    request_failed: int


@dataclass(frozen=True)
class IntegrationRequest:
    """Only the fields guaranteed by Harbor's prediction-time contract."""

    vendor: str
    endpoint: str
    recent_vendor_latency_ms: float
    recent_vendor_error_rate: float
    queue_depth: int
    retry_count: int
    request_size_bytes: int
    hour_of_day: int


@dataclass(frozen=True)
class IntegrationDatasetSplit:
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector


@dataclass(frozen=True)
class IntegrationEvaluation:
    accuracy: float
    confusion_matrix: NDArray[np.int64]
    false_positives: int
    false_negatives: int
    predictions: TargetVector


@dataclass(frozen=True)
class IntegrationFailurePrediction:
    probability: float
    predicted_failure: bool
    threshold: float


def _timestamp(value: str, row_number: int) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"row {row_number}: invalid timestamp") from error


def load_integration_requests(path: str | Path) -> list[IntegrationObservation]:
    """Load and validate the committed fictional request-level CSV."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except OSError:
        raise
    if not rows:
        raise ValueError("integration dataset must contain observations")

    observations = []
    for number, row in enumerate(rows, 2):
        numeric: dict[str, float] = {}
        for name in NUMERIC_FEATURES:
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"row {number}: invalid {name}") from error
            if not isfinite(value) or value < 0:
                raise ValueError(f"row {number}: invalid {name}")
            numeric[name] = value
        vendor = row.get("vendor", "").strip()
        endpoint = row.get("endpoint", "").strip()
        if not vendor or not endpoint:
            raise ValueError(f"row {number}: vendor and endpoint are required")
        try:
            target = int(row.get(TARGET, ""))
        except (TypeError, ValueError) as error:
            raise ValueError(f"row {number}: invalid {TARGET}") from error
        if target not in (0, 1):
            raise ValueError(f"row {number}: invalid {TARGET}")
        integer_names = ("queue_depth", "retry_count", "request_size_bytes", "hour_of_day")
        if any(not numeric[name].is_integer() for name in integer_names):
            raise ValueError(f"row {number}: count fields must be integers")
        if numeric["recent_vendor_error_rate"] > 1 or numeric["hour_of_day"] > 23:
            raise ValueError(f"row {number}: rate or hour outside valid range")
        observations.append(
            IntegrationObservation(
                timestamp=_timestamp(row.get("timestamp", ""), number),
                vendor=vendor,
                endpoint=endpoint,
                request_failed=target,
                **{name: int(numeric[name]) for name in integer_names},
                recent_vendor_latency_ms=numeric["recent_vendor_latency_ms"],
                recent_vendor_error_rate=numeric["recent_vendor_error_rate"],
            )
        )
    return observations


def build_integration_features(
    observations: Sequence[IntegrationObservation],
) -> FeatureMatrix:
    """Build only declared prediction-time columns; the target cannot leak in."""
    return np.asarray(
        [[getattr(row, name) for name in PREDICTION_FEATURES] for row in observations],
        dtype=object,
    ).reshape(len(observations), len(PREDICTION_FEATURES))


def build_integration_targets(
    observations: Sequence[IntegrationObservation],
) -> TargetVector:
    return np.asarray([row.request_failed for row in observations], dtype=np.int64)


def split_integration_dataset(
    X: FeatureMatrix,
    y: TargetVector,
    *,
    test_size: float = 0.25,
    random_state: int = INTEGRATION_RANDOM_STATE,
) -> IntegrationDatasetSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return IntegrationDatasetSplit(X_train, X_test, y_train, y_test)


def build_integration_pipeline() -> Pipeline:
    """Build named numeric/categorical preprocessing and binary classification."""
    numeric_indices = tuple(range(len(NUMERIC_FEATURES)))
    categorical_indices = tuple(range(len(NUMERIC_FEATURES), len(PREDICTION_FEATURES)))
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_indices),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_indices,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(random_state=INTEGRATION_RANDOM_STATE, max_iter=1_000),
            ),
        ]
    )


def train_integration_model(
    model: Pipeline, X_train: FeatureMatrix, y_train: TargetVector
) -> Pipeline:
    if X_train.ndim != 2 or X_train.shape[1] != len(PREDICTION_FEATURES):
        raise ValueError(f"features must have {len(PREDICTION_FEATURES)} columns")
    if X_train.shape[0] == 0 or set(np.unique(y_train)) - {0, 1}:
        raise ValueError("training data must contain valid binary observations")
    model.fit(X_train, y_train)
    return model


def evaluate_integration_model(
    model: Pipeline, X_test: FeatureMatrix, y_test: TargetVector
) -> IntegrationEvaluation:
    predictions = np.asarray(model.predict(X_test), dtype=np.int64)
    matrix = np.asarray(confusion_matrix(y_test, predictions, labels=(0, 1)), dtype=np.int64)
    return IntegrationEvaluation(
        accuracy=float(np.mean(predictions == y_test)),
        confusion_matrix=matrix,
        false_positives=int(matrix[0, 1]),
        false_negatives=int(matrix[1, 0]),
        predictions=predictions,
    )


def _request_row(request: IntegrationRequest | Mapping[str, object]) -> FeatureMatrix:
    values = asdict(request) if isinstance(request, IntegrationRequest) else dict(request)
    missing = [name for name in PREDICTION_FEATURES if name not in values]
    unexpected = [name for name in values if name not in PREDICTION_FEATURES]
    if missing:
        raise ValueError("missing required integration features: " + ", ".join(missing))
    if unexpected:
        raise ValueError("unexpected integration features: " + ", ".join(unexpected))
    for name in CATEGORICAL_FEATURES:
        if not isinstance(values[name], str) or not str(values[name]).strip():
            raise ValueError(f"{name} must be a non-empty string")
    for name in NUMERIC_FEATURES:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        if not isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if float(values["recent_vendor_error_rate"]) > 1 or float(values["hour_of_day"]) > 23:
        raise ValueError("recent_vendor_error_rate or hour_of_day outside valid range")
    return np.asarray([[values[name] for name in PREDICTION_FEATURES]], dtype=object)


def predict_failure_probability(
    model: Pipeline, request: IntegrationRequest | Mapping[str, object]
) -> float:
    """Return the fitted model's probability for the failure class (label 1)."""
    row = _request_row(request)
    classes = tuple(model.named_steps["classifier"].classes_)
    failure_index = classes.index(1)
    return float(model.predict_proba(row)[0, failure_index])


def predict_failure(
    model: Pipeline,
    request: IntegrationRequest | Mapping[str, object],
    *,
    threshold: float = 0.5,
) -> IntegrationFailurePrediction:
    """Apply an explicit decision threshold to one unchanged probability."""
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    probability = predict_failure_probability(model, request)
    return IntegrationFailurePrediction(probability, probability >= threshold, threshold)
