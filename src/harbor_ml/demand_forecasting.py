"""Leakage-safe, one-step system-demand regression for Chapter 14."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FORECAST_HORIZON_MINUTES = 10
OBSERVATION_INTERVAL_MINUTES = 5
DEMAND_FEATURES = ("requests_now", "requests_5m_ago", "requests_10m_ago",
    "requests_15m_ago", "recent_average_requests", "recent_growth",
    "api_latency_ms", "error_rate", "queue_depth", "hour_of_day", "day_of_week")
DEMAND_TARGET = "future_requests_per_minute"
FeatureMatrix = NDArray[np.float64]
TargetVector = NDArray[np.float64]


@dataclass(frozen=True)
class DemandObservation:
    timestamp: datetime
    requests_per_minute: float
    api_latency_ms: float
    error_rate: float
    queue_depth: int


@dataclass(frozen=True)
class DemandExample:
    timestamp: datetime
    requests_now: float
    requests_5m_ago: float
    requests_10m_ago: float
    requests_15m_ago: float
    recent_average_requests: float
    recent_growth: float
    api_latency_ms: float
    error_rate: float
    queue_depth: int
    hour_of_day: int
    day_of_week: int
    future_requests_per_minute: float


@dataclass(frozen=True)
class DemandSplit:
    train: tuple[DemandExample, ...]
    test: tuple[DemandExample, ...]


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float


def load_demand_observations(path: str | Path) -> list[DemandObservation]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("demand dataset must contain observations")
    result: list[DemandObservation] = []
    for number, row in enumerate(rows, 2):
        try:
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            requests = float(row["requests_per_minute"])
            latency = float(row["api_latency_ms"])
            error_rate = float(row["error_rate"])
            queue_depth = int(row["queue_depth"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"row {number}: malformed demand observation") from error
        if not all(isfinite(value) for value in (requests, latency, error_rate)) \
                or requests < 0 or latency < 0 or not 0 <= error_rate <= 1 or queue_depth < 0:
            raise ValueError(f"row {number}: invalid demand observation")
        result.append(DemandObservation(timestamp, requests, latency, error_rate, queue_depth))
    if any(left.timestamp >= right.timestamp for left, right in zip(result, result[1:])):
        raise ValueError("demand observations must be strictly chronological")
    return result


def build_demand_examples(observations: Sequence[DemandObservation]) -> list[DemandExample]:
    """Align three past lags at t with the target exactly t + 10 minutes."""
    examples: list[DemandExample] = []
    interval = timedelta(minutes=OBSERVATION_INTERVAL_MINUTES)
    for index in range(3, len(observations) - 2):
        window = observations[index - 3:index + 1]
        target = observations[index + 2]
        if any(right.timestamp - left.timestamp != interval
               for left, right in zip(observations[index - 3:index + 2],
                                      observations[index - 2:index + 3])):
            continue
        now = window[-1]
        examples.append(DemandExample(now.timestamp, now.requests_per_minute,
            window[-2].requests_per_minute, window[-3].requests_per_minute,
            window[-4].requests_per_minute,
            float(np.mean([item.requests_per_minute for item in window])),
            now.requests_per_minute - window[0].requests_per_minute,
            now.api_latency_ms, now.error_rate, now.queue_depth, now.timestamp.hour,
            now.timestamp.weekday(), target.requests_per_minute))
    return examples


def chronological_split(examples: Sequence[DemandExample], *, train_fraction: float = .8) -> DemandSplit:
    if not 0 < train_fraction < 1 or len(examples) < 2:
        raise ValueError("split requires examples and a fraction between zero and one")
    if any(left.timestamp >= right.timestamp for left, right in zip(examples, examples[1:])):
        raise ValueError("examples must be strictly chronological")
    boundary = int(len(examples) * train_fraction)
    return DemandSplit(tuple(examples[:boundary]), tuple(examples[boundary:]))


def build_demand_features(examples: Sequence[DemandExample]) -> FeatureMatrix:
    return np.asarray([[getattr(item, name) for name in DEMAND_FEATURES]
                       for item in examples], dtype=float).reshape(len(examples), len(DEMAND_FEATURES))


def build_demand_targets(examples: Sequence[DemandExample]) -> TargetVector:
    return np.asarray([item.future_requests_per_minute for item in examples], dtype=float)


def build_demand_model() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=10.0))])


def train_demand_model(model: Pipeline, examples: Sequence[DemandExample]) -> Pipeline:
    if not examples:
        raise ValueError("training examples are required")
    model.fit(build_demand_features(examples), build_demand_targets(examples))
    return model


def predict_future_demand(model: Pipeline, examples: Sequence[DemandExample]) -> TargetVector:
    return np.asarray(model.predict(build_demand_features(examples)), dtype=float)


def calculate_metrics(actual: Sequence[float], predicted: Sequence[float]) -> RegressionMetrics:
    actual_array, predicted_array = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(actual_array) == 0 or actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted values must have equal nonzero length")
    errors = actual_array - predicted_array
    return RegressionMetrics(float(np.mean(np.abs(errors))), float(sqrt(np.mean(errors ** 2))))


def persistence_predictions(examples: Sequence[DemandExample]) -> TargetVector:
    return np.asarray([item.requests_now for item in examples], dtype=float)


def residuals(actual: Sequence[float], predicted: Sequence[float]) -> TargetVector:
    actual_array, predicted_array = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted values must have equal length")
    return actual_array - predicted_array
