"""Privacy-minimized behavioral session clustering for Chapter 11."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .member_behavior import MemberEvent

SEGMENTATION_FEATURES = (
    "session_duration_seconds",
    "event_count",
    "account_view_count",
    "search_count",
    "transfer_count",
    "statement_view_count",
    "help_event_count",
    "verification_event_count",
)
SEGMENTATION_RANDOM_STATE = 42

FeatureMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class BehavioralSession:
    """One pseudonymous session represented only by interaction measurements."""

    session_id: str
    session_duration_seconds: float
    event_count: int
    account_view_count: int
    search_count: int
    transfer_count: int
    statement_view_count: int
    help_event_count: int
    verification_event_count: int


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: int
    session_count: int
    means: Mapping[str, float]


def build_behavioral_sessions(
    sessions: Mapping[str, Sequence[MemberEvent]],
) -> list[BehavioralSession]:
    """Derive one behavioral row per nonempty session from Chapter 8 events."""
    rows: list[BehavioralSession] = []
    for session_id, raw_events in sorted(sessions.items()):
        if not raw_events:
            raise ValueError(f"session {session_id!r} contains no events")
        events = sorted(raw_events, key=lambda event: (event.timestamp, event.event_name))
        if {event.session_id for event in events} != {session_id}:
            raise ValueError("session mapping key must match every event session_id")
        if len({event.channel for event in events}) != 1:
            raise ValueError("a behavioral session must use one channel")
        duration = (events[-1].timestamp - events[0].timestamp).total_seconds()
        if duration < 0:  # defensive contract if sorting behavior ever changes
            raise ValueError("session duration cannot be negative")
        names = [event.event_name for event in events]
        rows.append(BehavioralSession(
            session_id=session_id,
            session_duration_seconds=duration,
            event_count=len(events),
            account_view_count=names.count("account_viewed"),
            search_count=names.count("search_performed"),
            transfer_count=sum(name.startswith("transfer_") for name in names),
            statement_view_count=names.count("statement_viewed"),
            help_event_count=names.count("help_opened"),
            verification_event_count=sum(name.startswith("verification_") for name in names),
        ))
    if not rows:
        raise ValueError("segmentation requires at least one session")
    return rows


def build_segmentation_features(rows: Sequence[BehavioralSession]) -> FeatureMatrix:
    """Build X only: the returned matrix intentionally has no target column."""
    if not rows:
        raise ValueError("segmentation requires at least one behavioral session")
    expected_fields = {field.name for field in fields(BehavioralSession)}
    if not set(SEGMENTATION_FEATURES) < expected_fields:
        raise RuntimeError("segmentation feature contract does not match BehavioralSession")
    matrix = np.asarray(
        [[getattr(row, feature) for feature in SEGMENTATION_FEATURES] for row in rows],
        dtype=float,
    )
    if matrix.shape != (len(rows), len(SEGMENTATION_FEATURES)):
        raise ValueError("malformed segmentation feature matrix")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("segmentation features must be finite and nonnegative")
    return matrix


def build_segmentation_pipeline(
    k: int = 4, *, random_state: int = SEGMENTATION_RANDOM_STATE,
) -> Pipeline:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=random_state, n_init=10)),
    ])


def train_segmentation_model(pipeline: Pipeline, X: FeatureMatrix) -> Pipeline:
    if X.ndim != 2 or X.shape[1] != len(SEGMENTATION_FEATURES):
        raise ValueError("X must contain the documented segmentation features in order")
    if len(X) < pipeline.named_steps["kmeans"].n_clusters:
        raise ValueError("k cannot exceed the number of sessions")
    pipeline.fit(X)
    return pipeline


def assign_clusters(model: Pipeline, rows: Sequence[BehavioralSession]) -> NDArray[np.int32]:
    return np.asarray(model.predict(build_segmentation_features(rows)), dtype=np.int32)


def calculate_inertia_by_k(
    X: FeatureMatrix, k_values: Sequence[int], *, random_state: int = SEGMENTATION_RANDOM_STATE,
) -> dict[int, float]:
    if not k_values:
        raise ValueError("provide at least one k value")
    results: dict[int, float] = {}
    for k in k_values:
        model = train_segmentation_model(
            build_segmentation_pipeline(k, random_state=random_state), X,
        )
        results[k] = float(model.named_steps["kmeans"].inertia_)
    return results


def inverse_transformed_centers(model: Pipeline) -> FeatureMatrix:
    """Return fitted K-means centers in the original behavioral units."""
    scaler = model.named_steps["scaler"]
    kmeans = model.named_steps["kmeans"]
    return np.asarray(scaler.inverse_transform(kmeans.cluster_centers_), dtype=float)


def summarize_clusters(
    rows: Sequence[BehavioralSession], assignments: Sequence[int],
) -> list[ClusterSummary]:
    if len(rows) != len(assignments):
        raise ValueError("every behavioral session requires one cluster assignment")
    if not rows:
        raise ValueError("cannot summarize zero sessions")
    matrix = build_segmentation_features(rows)
    labels = np.asarray(assignments)
    summaries = []
    for cluster_id in sorted(int(value) for value in np.unique(labels)):
        selected = matrix[labels == cluster_id]
        summaries.append(ClusterSummary(
            cluster_id=cluster_id,
            session_count=len(selected),
            means=dict(zip(SEGMENTATION_FEATURES, selected.mean(axis=0), strict=True)),
        ))
    return summaries
