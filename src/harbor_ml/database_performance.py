"""Pre-execution database-duration regression for Chapter 15."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

QUERY_FAMILIES = ("account_summary", "transaction_history", "member_search",
                  "statement_lookup", "transfer_history", "verification_audit")
ROW_BANDS = ("tiny", "small", "medium", "large", "very_large")
QUERY_FEATURES = ("query_family", "rows_expected_band", "join_count", "filter_count",
                  "uses_sort", "uses_aggregation", "uses_grouping", "current_db_connections",
                  "current_queue_depth", "recent_db_latency_ms", "requests_per_minute")
QUERY_TARGET = "query_duration_ms"


@dataclass(frozen=True)
class QueryContext:
    query_family: str
    rows_expected_band: str
    join_count: int
    filter_count: int
    uses_sort: bool
    uses_aggregation: bool
    uses_grouping: bool
    current_db_connections: int
    current_queue_depth: int
    recent_db_latency_ms: float
    requests_per_minute: float

    def __post_init__(self) -> None:
        if self.query_family not in QUERY_FAMILIES or self.rows_expected_band not in ROW_BANDS:
            raise ValueError("unknown query family or row band")
        counts = (self.join_count, self.filter_count, self.current_db_connections,
                  self.current_queue_depth)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("count features must be nonnegative integers")
        if (not isfinite(self.recent_db_latency_ms) or self.recent_db_latency_ms < 0
                or not isfinite(self.requests_per_minute) or self.requests_per_minute < 0):
            raise ValueError("continuous features must be finite and nonnegative")
        if any(type(value) is not bool for value in
               (self.uses_sort, self.uses_aggregation, self.uses_grouping)):
            raise ValueError("query flags must be booleans")


@dataclass(frozen=True)
class QueryObservation:
    timestamp: datetime
    context: QueryContext
    query_duration_ms: float


@dataclass(frozen=True)
class QuerySplit:
    train: tuple[QueryObservation, ...]
    test: tuple[QueryObservation, ...]


@dataclass(frozen=True)
class QueryDurationPrediction:
    predicted_duration_ms: float


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    median_absolute_error: float


def _boolean(value: str) -> bool:
    if value not in ("true", "false"):
        raise ValueError("boolean must be true or false")
    return value == "true"


def load_query_performance(path: str | Path) -> list[QueryObservation]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("query dataset must contain observations")
    result = []
    for number, row in enumerate(rows, 2):
        try:
            context = QueryContext(row["query_family"], row["rows_expected_band"],
                int(row["join_count"]), int(row["filter_count"]), _boolean(row["uses_sort"]),
                _boolean(row["uses_aggregation"]), _boolean(row["uses_grouping"]),
                int(row["current_db_connections"]), int(row["current_queue_depth"]),
                float(row["recent_db_latency_ms"]), float(row["requests_per_minute"]))
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            duration = float(row[QUERY_TARGET])
            if not isfinite(duration) or duration < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"row {number}: malformed query observation") from error
        result.append(QueryObservation(timestamp, context, duration))
    if any(a.timestamp >= b.timestamp for a, b in zip(result, result[1:])):
        raise ValueError("query observations must be strictly chronological")
    return result


def chronological_split(observations: Sequence[QueryObservation], *, train_fraction: float = .8) -> QuerySplit:
    if len(observations) < 2 or not 0 < train_fraction < 1:
        raise ValueError("split requires observations and a fraction between zero and one")
    if any(a.timestamp >= b.timestamp for a, b in zip(observations, observations[1:])):
        raise ValueError("observations must be strictly chronological")
    boundary = int(len(observations) * train_fraction)
    return QuerySplit(tuple(observations[:boundary]), tuple(observations[boundary:]))


def build_query_features(contexts: Sequence[QueryContext]) -> NDArray[np.object_]:
    return np.asarray([[getattr(item, name) for name in QUERY_FEATURES] for item in contexts],
                      dtype=object).reshape(len(contexts), len(QUERY_FEATURES))


def build_database_performance_pipeline(*, random_state: int = 1515) -> Pipeline:
    transformer = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), [0, 1]),
        ("values", "passthrough", list(range(2, len(QUERY_FEATURES)))),
    ])
    forest = RandomForestRegressor(n_estimators=160, min_samples_leaf=2, n_jobs=1,
                                   random_state=random_state)
    return Pipeline([("preprocessor", transformer), ("regressor", forest)])


def train_database_performance_model(model: Pipeline,
                                     observations: Sequence[QueryObservation]) -> Pipeline:
    if not observations:
        raise ValueError("training observations are required")
    model.fit(build_query_features([x.context for x in observations]),
              [x.query_duration_ms for x in observations])
    return model


def predict_query_duration(model: Pipeline, contexts: Sequence[QueryContext]) -> NDArray[np.float64]:
    if not contexts:
        return np.asarray([], dtype=float)
    return np.maximum(0, np.asarray(model.predict(build_query_features(contexts)), dtype=float))


def query_family_baseline(train: Sequence[QueryObservation], contexts: Sequence[QueryContext]) -> NDArray[np.float64]:
    if not train:
        raise ValueError("baseline requires training observations")
    overall = float(np.median([x.query_duration_ms for x in train]))
    medians = {family: float(np.median([x.query_duration_ms for x in train
                                        if x.context.query_family == family]))
               for family in QUERY_FAMILIES if any(x.context.query_family == family for x in train)}
    return np.asarray([medians.get(x.query_family, overall) for x in contexts], dtype=float)


def calculate_metrics(actual: Sequence[float], predicted: Sequence[float]) -> RegressionMetrics:
    actual_array, predicted_array = np.asarray(actual), np.asarray(predicted)
    if not len(actual_array) or actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted values must have equal nonzero length")
    absolute = np.abs(actual_array - predicted_array)
    return RegressionMetrics(float(np.mean(absolute)),
                             float(sqrt(np.mean((actual_array - predicted_array) ** 2))),
                             float(np.median(absolute)))


def residuals(actual: Sequence[float], predicted: Sequence[float]) -> NDArray[np.float64]:
    actual_array, predicted_array = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted values must have equal length")
    return actual_array - predicted_array
