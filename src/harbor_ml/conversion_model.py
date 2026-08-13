"""Leakage-safe product-session conversion classification for Chapter 10."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .member_behavior import MemberEvent

NUMERIC_CONVERSION_FEATURES = (
    "elapsed_seconds", "events_so_far", "search_count", "help_events",
    "prior_login_failures", "hour_of_day",
)
CATEGORICAL_CONVERSION_FEATURES = ("channel", "landing_source", "device_category")
CONVERSION_FEATURES = (*NUMERIC_CONVERSION_FEATURES, *CATEGORICAL_CONVERSION_FEATURES)
CONVERSION_TARGET = "application_started"
CONVERSION_RANDOM_STATE = 42

FeatureMatrix = NDArray[np.object_]
TargetVector = NDArray[np.int64]


@dataclass(frozen=True)
class ConversionSnapshot:
    session_id: str
    snapshot_timestamp: datetime
    channel: str
    landing_source: str
    device_category: str
    elapsed_seconds: float
    events_so_far: int
    search_count: int
    help_events: int
    prior_login_failures: int
    hour_of_day: int
    application_started: bool


@dataclass(frozen=True)
class PartialConversionState:
    channel: str
    landing_source: str
    device_category: str
    elapsed_seconds: float
    events_so_far: int
    search_count: int
    help_events: int
    prior_login_failures: int
    hour_of_day: int


@dataclass(frozen=True)
class ConversionDatasetSplit:
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector


@dataclass(frozen=True)
class ConversionEvaluation:
    accuracy: float
    confusion_matrix: NDArray[np.int64]
    false_positives: int
    false_negatives: int
    predictions: TargetVector


@dataclass(frozen=True)
class ConversionPrediction:
    probability: float
    predicted_conversion: bool


def build_conversion_snapshots(
    sessions: Mapping[str, Sequence[MemberEvent]],
) -> list[ConversionSnapshot]:
    """Snapshot eligible sessions at their first product-details view.

    The prefix through ``product_details_viewed`` supplies features. Only the
    strictly later suffix supplies the historical label.
    """
    snapshots: list[ConversionSnapshot] = []
    for session_id, raw_events in sorted(sessions.items()):
        if not raw_events:
            continue
        events = sorted(raw_events, key=lambda event: (event.timestamp, event.event_name))
        if {event.session_id for event in events} != {session_id}:
            raise ValueError("session mapping key must match every event session_id")
        if len({event.channel for event in events}) != 1:
            raise ValueError("a conversion session must use one channel")
        prediction_index = next(
            (index for index, event in enumerate(events)
             if event.event_name == "product_details_viewed"), None,
        )
        if prediction_index is None:
            continue
        prefix = events[: prediction_index + 1]
        if prefix[0].event_name != "session_started":
            raise ValueError("an eligible conversion session must begin with session_started")
        landing_sources = {event.landing_source for event in events if event.landing_source}
        device_categories = {event.device_category for event in events if event.device_category}
        if len(landing_sources) != 1 or len(device_categories) != 1:
            raise ValueError("an eligible conversion session requires one landing source and device category")
        prediction = events[prediction_index]
        snapshots.append(ConversionSnapshot(
            session_id=session_id,
            snapshot_timestamp=prediction.timestamp,
            channel=prediction.channel,
            landing_source=landing_sources.pop(),
            device_category=device_categories.pop(),
            elapsed_seconds=(prediction.timestamp - prefix[0].timestamp).total_seconds(),
            events_so_far=len(prefix),
            search_count=sum(event.event_name == "search_performed" for event in prefix),
            help_events=sum(event.event_name == "help_opened" for event in prefix),
            prior_login_failures=sum(event.event_name == "login_failed" for event in prefix),
            hour_of_day=prediction.timestamp.hour,
            application_started=any(
                event.event_name == "application_started"
                for event in events[prediction_index + 1 :]
            ),
        ))
    return snapshots


def build_conversion_features(
    rows: Sequence[ConversionSnapshot | PartialConversionState],
) -> FeatureMatrix:
    return np.asarray(
        [[getattr(row, feature) for feature in CONVERSION_FEATURES] for row in rows],
        dtype=object,
    ).reshape(len(rows), len(CONVERSION_FEATURES))


def build_conversion_targets(rows: Sequence[ConversionSnapshot]) -> TargetVector:
    return np.asarray([int(row.application_started) for row in rows], dtype=np.int64)


def calculate_conversion_rate(rows: Sequence[ConversionSnapshot]) -> float:
    if not rows:
        raise ValueError("cannot calculate conversion rate without eligible sessions")
    return sum(row.application_started for row in rows) / len(rows)


def calculate_majority_baseline(targets: TargetVector) -> float:
    if len(targets) == 0:
        raise ValueError("cannot calculate a baseline without targets")
    counts = np.bincount(targets, minlength=2)
    return float(counts.max() / len(targets))


def split_conversion_dataset(
    X: FeatureMatrix, y: TargetVector, *, test_size: float = 0.25,
    random_state: int = CONVERSION_RANDOM_STATE,
) -> ConversionDatasetSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    return ConversionDatasetSplit(X_train, X_test, y_train, y_test)


def build_conversion_pipeline() -> Pipeline:
    numeric = tuple(range(len(NUMERIC_CONVERSION_FEATURES)))
    categorical = tuple(range(len(NUMERIC_CONVERSION_FEATURES), len(CONVERSION_FEATURES)))
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        ("classifier", LogisticRegression(random_state=CONVERSION_RANDOM_STATE, max_iter=1_000)),
    ])


def train_conversion_model(
    model: Pipeline, X_train: FeatureMatrix, y_train: TargetVector,
) -> Pipeline:
    if X_train.ndim != 2 or X_train.shape[1] != len(CONVERSION_FEATURES):
        raise ValueError(f"features must have {len(CONVERSION_FEATURES)} columns")
    if len(X_train) == 0 or set(np.unique(y_train)) != {0, 1}:
        raise ValueError("training data must contain both binary target classes")
    model.fit(X_train, y_train)
    return model


def evaluate_conversion_model(
    model: Pipeline, X_test: FeatureMatrix, y_test: TargetVector,
) -> ConversionEvaluation:
    predictions = np.asarray(model.predict(X_test), dtype=np.int64)
    matrix = np.asarray(confusion_matrix(y_test, predictions, labels=(0, 1)), dtype=np.int64)
    return ConversionEvaluation(
        float(np.mean(predictions == y_test)), matrix, int(matrix[0, 1]),
        int(matrix[1, 0]), predictions,
    )


def classify_conversion(probability: float, *, threshold: float = 0.5) -> bool:
    if not 0 <= probability <= 1 or not 0 <= threshold <= 1:
        raise ValueError("probability and threshold must be between 0 and 1")
    return probability >= threshold


def predict_conversion_probability(model: Pipeline, state: PartialConversionState) -> float:
    return float(model.predict_proba(build_conversion_features([state]))[0, 1])


def predict_conversion(
    model: Pipeline, state: PartialConversionState, *, threshold: float = 0.5,
) -> ConversionPrediction:
    probability = predict_conversion_probability(model, state)
    return ConversionPrediction(probability, classify_conversion(probability, threshold=threshold))
