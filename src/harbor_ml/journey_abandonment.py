"""Leakage-safe transfer-journey abandonment classification for Chapter 9."""

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

NUMERIC_JOURNEY_FEATURES = ("elapsed_seconds", "events_so_far", "error_count", "search_count")
BOOLEAN_JOURNEY_FEATURES = ("help_opened", "previous_failed_login")
CATEGORICAL_JOURNEY_FEATURES = ("channel",)
JOURNEY_FEATURES = (*NUMERIC_JOURNEY_FEATURES, *BOOLEAN_JOURNEY_FEATURES, *CATEGORICAL_JOURNEY_FEATURES)
JOURNEY_TARGET = "journey_abandoned"
JOURNEY_RANDOM_STATE = 42

FeatureMatrix = NDArray[np.object_]
TargetVector = NDArray[np.int64]


@dataclass(frozen=True)
class JourneySnapshot:
    snapshot_timestamp: datetime
    session_id: str
    channel: str
    elapsed_seconds: float
    events_so_far: int
    help_opened: bool
    error_count: int
    search_count: int
    previous_failed_login: bool
    journey_abandoned: bool


@dataclass(frozen=True)
class PartialJourneyState:
    channel: str
    elapsed_seconds: float
    events_so_far: int
    help_opened: bool
    error_count: int
    search_count: int
    previous_failed_login: bool


@dataclass(frozen=True)
class JourneyDatasetSplit:
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector


@dataclass(frozen=True)
class JourneyEvaluation:
    accuracy: float
    confusion_matrix: NDArray[np.int64]
    false_positives: int
    false_negatives: int
    predictions: TargetVector


def build_transfer_snapshots(
    sessions: Mapping[str, Sequence[MemberEvent]],
) -> list[JourneySnapshot]:
    """Freeze each eligible session at its first recipient_selected event.

    Features inspect events no later than the prediction point. Only label
    construction inspects subsequent events.
    """
    snapshots: list[JourneySnapshot] = []
    for session_id, raw_events in sorted(sessions.items()):
        if not raw_events:
            continue
        events = sorted(raw_events, key=lambda event: (event.timestamp, event.event_name))
        if {event.session_id for event in events} != {session_id}:
            raise ValueError("session mapping key must match every event session_id")
        channels = {event.channel for event in events}
        if len(channels) != 1:
            raise ValueError("a journey must use one channel")
        start_index = next((i for i, event in enumerate(events) if event.event_name == "transfer_started"), None)
        if start_index is None:
            continue
        prediction_index = next(
            (i for i in range(start_index + 1, len(events)) if events[i].event_name == "recipient_selected"),
            None,
        )
        if prediction_index is None:
            continue
        start = events[start_index]
        prediction = events[prediction_index]
        feature_window = events[start_index : prediction_index + 1]
        known_session_history = events[: prediction_index + 1]
        label_window = events[prediction_index + 1 :]
        snapshots.append(JourneySnapshot(
            snapshot_timestamp=prediction.timestamp,
            session_id=session_id,
            channel=prediction.channel,
            elapsed_seconds=(prediction.timestamp - start.timestamp).total_seconds(),
            events_so_far=len(feature_window),
            help_opened=any(event.event_name == "help_opened" for event in feature_window),
            error_count=sum(event.event_name == "transfer_failed" for event in feature_window),
            search_count=sum(event.event_name == "search_performed" for event in feature_window),
            previous_failed_login=any(event.event_name == "login_failed" for event in known_session_history),
            journey_abandoned=not any(event.event_name == "transfer_completed" for event in label_window),
        ))
    return snapshots


def build_journey_features(rows: Sequence[JourneySnapshot | PartialJourneyState]) -> FeatureMatrix:
    return np.asarray(
        [[getattr(row, name) for name in JOURNEY_FEATURES] for row in rows], dtype=object
    ).reshape(len(rows), len(JOURNEY_FEATURES))


def build_journey_targets(rows: Sequence[JourneySnapshot]) -> TargetVector:
    return np.asarray([int(row.journey_abandoned) for row in rows], dtype=np.int64)


def split_journey_dataset(X: FeatureMatrix, y: TargetVector, *, test_size: float = 0.25,
                          random_state: int = JOURNEY_RANDOM_STATE) -> JourneyDatasetSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return JourneyDatasetSplit(X_train, X_test, y_train, y_test)


def build_abandonment_pipeline() -> Pipeline:
    numeric = tuple(range(len(NUMERIC_JOURNEY_FEATURES)))
    booleans = tuple(range(len(NUMERIC_JOURNEY_FEATURES), len(NUMERIC_JOURNEY_FEATURES) + len(BOOLEAN_JOURNEY_FEATURES)))
    categorical = (len(JOURNEY_FEATURES) - 1,)
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("boolean", "passthrough", booleans),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", LogisticRegression(random_state=JOURNEY_RANDOM_STATE, max_iter=1_000)),
    ])


def train_abandonment_model(model: Pipeline, X_train: FeatureMatrix, y_train: TargetVector) -> Pipeline:
    if X_train.ndim != 2 or X_train.shape[1] != len(JOURNEY_FEATURES):
        raise ValueError(f"features must have {len(JOURNEY_FEATURES)} columns")
    if len(X_train) == 0 or set(np.unique(y_train)) != {0, 1}:
        raise ValueError("training data must contain both binary target classes")
    model.fit(X_train, y_train)
    return model


def evaluate_abandonment_model(model: Pipeline, X_test: FeatureMatrix,
                               y_test: TargetVector) -> JourneyEvaluation:
    predictions = np.asarray(model.predict(X_test), dtype=np.int64)
    matrix = np.asarray(confusion_matrix(y_test, predictions, labels=(0, 1)), dtype=np.int64)
    return JourneyEvaluation(float(np.mean(predictions == y_test)), matrix,
                             int(matrix[0, 1]), int(matrix[1, 0]), predictions)


def predict_abandonment_probability(model: Pipeline, state: PartialJourneyState) -> float:
    return float(model.predict_proba(build_journey_features([state]))[0, 1])


def classify_abandonment(probability: float, *, threshold: float = 0.5) -> bool:
    if not 0 <= probability <= 1 or not 0 <= threshold <= 1:
        raise ValueError("probability and threshold must be between 0 and 1")
    return probability >= threshold
