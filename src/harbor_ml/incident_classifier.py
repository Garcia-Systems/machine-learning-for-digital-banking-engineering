"""Deterministic multi-class incident classification for Chapter 5."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INCIDENT_FEATURES = (
    "api_latency_ms",
    "error_rate",
    "db_connections",
    "queue_depth",
    "vendor_latency_ms",
    "requests_per_minute",
)
INCIDENT_CLASSES = (
    "normal",
    "vendor_degradation",
    "database_pressure",
    "traffic_spike",
    "application_regression",
)
INCIDENT_RANDOM_STATE = 42

FeatureMatrix = NDArray[np.float64]
TargetVector = NDArray[np.str_]
IncidentFeatures = Mapping[str, int | float]


@dataclass(frozen=True)
class IncidentObservation:
    """One labeled, fictional historical telemetry observation."""

    timestamp: datetime
    api_latency_ms: float
    error_rate: float
    db_connections: float
    queue_depth: float
    vendor_latency_ms: float
    requests_per_minute: float
    incident_type: str


@dataclass(frozen=True)
class IncidentDatasetSplit:
    """A deterministic, stratified training/test split."""

    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector


@dataclass(frozen=True)
class IncidentEvaluation:
    """Held-out accuracy, predictions, and a labeled confusion matrix."""

    accuracy: float
    predictions: TargetVector
    confusion_matrix: NDArray[np.int64]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class IncidentPrediction:
    """The selected known class and model-assigned probabilities."""

    predicted_class: str
    probabilities: dict[str, float]


def _timestamp(value: str, row_number: int) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"row {row_number}: invalid timestamp") from error


def load_incident_dataset(path: str | Path) -> list[IncidentObservation]:
    """Load and validate labeled fictional incident observations from CSV."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("incident dataset must contain observations")

    observations: list[IncidentObservation] = []
    for number, row in enumerate(rows, 2):
        values: dict[str, float] = {}
        for name in INCIDENT_FEATURES:
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"row {number}: invalid {name}") from error
            if not isfinite(value) or value < 0:
                raise ValueError(f"row {number}: invalid {name}")
            values[name] = value
        label = row.get("incident_type", "").strip()
        if label not in INCIDENT_CLASSES:
            raise ValueError(f"row {number}: invalid incident_type: {label}")
        observations.append(
            IncidentObservation(
                timestamp=_timestamp(row.get("timestamp", ""), number),
                incident_type=label,
                **values,
            )
        )
    return observations


def build_incident_features(
    observations: Sequence[IncidentObservation],
) -> FeatureMatrix:
    """Select numeric predictors in the declared inference-time order."""
    return np.asarray(
        [[getattr(row, name) for name in INCIDENT_FEATURES] for row in observations],
        dtype=np.float64,
    ).reshape(len(observations), len(INCIDENT_FEATURES))


def build_incident_targets(
    observations: Sequence[IncidentObservation],
) -> TargetVector:
    """Extract the known category, which is never included in model features."""
    return np.asarray([row.incident_type for row in observations], dtype=np.str_)


def split_incident_dataset(
    X: FeatureMatrix,
    y: TargetVector,
    *,
    test_size: float = 0.25,
    random_state: int = INCIDENT_RANDOM_STATE,
) -> IncidentDatasetSplit:
    """Split reproducibly while preserving every class's relative frequency."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return IncidentDatasetSplit(X_train, X_test, y_train, y_test)


def build_incident_classifier() -> Pipeline:
    """Build an unfitted scaling and multi-class logistic-regression pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(random_state=INCIDENT_RANDOM_STATE, max_iter=1_000),
            ),
        ]
    )


def train_incident_classifier(
    model: Pipeline, X_train: FeatureMatrix, y_train: TargetVector
) -> Pipeline:
    """Fit the classifier on labeled historical observations."""
    if X_train.ndim != 2 or X_train.shape[1] != len(INCIDENT_FEATURES):
        raise ValueError(f"features must have {len(INCIDENT_FEATURES)} columns")
    if X_train.shape[0] == 0 or not np.isfinite(X_train).all():
        raise ValueError("features must contain finite observations")
    model.fit(X_train, y_train)
    return model


def evaluate_incident_classifier(
    model: Pipeline, X_test: FeatureMatrix, y_test: TargetVector
) -> IncidentEvaluation:
    """Calculate held-out accuracy and a matrix in the declared label order."""
    predictions = np.asarray(model.predict(X_test), dtype=np.str_)
    matrix = confusion_matrix(y_test, predictions, labels=INCIDENT_CLASSES)
    return IncidentEvaluation(
        accuracy=float(np.mean(predictions == y_test)),
        predictions=predictions,
        confusion_matrix=np.asarray(matrix, dtype=np.int64),
        labels=INCIDENT_CLASSES,
    )


def _validated_row(observation: IncidentFeatures) -> FeatureMatrix:
    missing = [name for name in INCIDENT_FEATURES if name not in observation]
    unexpected = [name for name in observation if name not in INCIDENT_FEATURES]
    if missing:
        raise ValueError("missing required incident features: " + ", ".join(missing))
    if unexpected:
        raise ValueError("unexpected incident features: " + ", ".join(unexpected))
    values: list[float] = []
    for name in INCIDENT_FEATURES:
        value = observation[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        numeric = float(value)
        if not isfinite(numeric) or numeric < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        values.append(numeric)
    return np.asarray([values], dtype=np.float64)


def predict_incident(
    model: Pipeline, observation: IncidentFeatures
) -> IncidentPrediction:
    """Classify one observation and map probabilities through ``model.classes_``."""
    row = _validated_row(observation)
    predicted_class = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    # Pipeline exposes its fitted classifier's classes_ property. Never assume that
    # alphabetical, taxonomy, or display order matches predict_proba's columns.
    mapped = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities, strict=True)
    }
    return IncidentPrediction(predicted_class, mapped)


def predict_incident_probabilities(
    model: Pipeline, observation: IncidentFeatures
) -> dict[str, float]:
    """Return all model-assigned probabilities for a validated observation."""
    return predict_incident(model, observation).probabilities
