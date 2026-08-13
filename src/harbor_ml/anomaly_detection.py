"""Deterministic application-telemetry anomaly detection for Chapter 4."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURE_NAMES = (
    "api_latency_ms",
    "error_rate",
    "db_connections",
    "queue_depth",
    "vendor_latency_ms",
    "requests_per_minute",
)
ANOMALY_RANDOM_STATE = 42
ANOMALY_CONTAMINATION = 0.05

FeatureMatrix = NDArray[np.float64]
TelemetryFeatures = Mapping[str, int | float]


@dataclass(frozen=True)
class TelemetryObservation:
    """One fictional point-in-time application observation."""

    timestamp: datetime
    api_latency_ms: float
    error_rate: float
    db_connections: float
    queue_depth: float
    vendor_latency_ms: float
    requests_per_minute: float


@dataclass(frozen=True)
class AnomalyScenario:
    """A named observation used to make the laboratory output readable."""

    name: str
    observation: TelemetryObservation


@dataclass(frozen=True)
class AnomalyResult:
    """An Isolation Forest score and the binary decision derived from it."""

    is_anomaly: bool
    score: float


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid timestamp: {value}") from error


def _parse_observation(row: Mapping[str, str], row_number: int) -> TelemetryObservation:
    values: dict[str, float] = {}
    for name in ANOMALY_FEATURE_NAMES:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"row {row_number}: invalid {name}") from error
        if not isfinite(value) or value < 0:
            raise ValueError(f"row {row_number}: invalid {name}")
        values[name] = value
    try:
        timestamp = _parse_timestamp(row["timestamp"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"row {row_number}: invalid timestamp") from error
    return TelemetryObservation(timestamp=timestamp, **values)


def load_normal_telemetry(path: str | Path) -> list[TelemetryObservation]:
    """Load and validate the committed normal-operation baseline."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("normal telemetry dataset must contain observations")
    return [_parse_observation(row, number) for number, row in enumerate(rows, 2)]


def load_anomaly_scenarios(path: str | Path) -> list[AnomalyScenario]:
    """Load named, fictional evaluation scenarios without using them to train."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    scenarios: list[AnomalyScenario] = []
    for number, row in enumerate(rows, 2):
        name = row.get("scenario", "").strip()
        if not name:
            raise ValueError(f"row {number}: scenario is required")
        scenarios.append(AnomalyScenario(name, _parse_observation(row, number)))
    if not scenarios:
        raise ValueError("scenario dataset must contain observations")
    return scenarios


def build_anomaly_features(
    observations: Sequence[TelemetryObservation],
) -> FeatureMatrix:
    """Select numerical values in the model's declared feature order."""
    return np.asarray(
        [
            [getattr(observation, name) for name in ANOMALY_FEATURE_NAMES]
            for observation in observations
        ],
        dtype=np.float64,
    ).reshape(len(observations), len(ANOMALY_FEATURE_NAMES))


def build_anomaly_detector(
    *,
    contamination: float = ANOMALY_CONTAMINATION,
    random_state: int = ANOMALY_RANDOM_STATE,
) -> IsolationForest:
    """Build an unfitted Isolation Forest with explicit teaching assumptions."""
    return IsolationForest(contamination=contamination, random_state=random_state)


def train_anomaly_detector(
    detector: IsolationForest, features: FeatureMatrix
) -> IsolationForest:
    """Fit expected behavior from the normal-operation feature matrix."""
    if features.ndim != 2 or features.shape[1] != len(ANOMALY_FEATURE_NAMES):
        raise ValueError(f"features must have {len(ANOMALY_FEATURE_NAMES)} columns")
    if features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("features must contain finite observations")
    detector.fit(features)
    return detector


def _validated_row(observation: TelemetryFeatures) -> FeatureMatrix:
    missing = [name for name in ANOMALY_FEATURE_NAMES if name not in observation]
    unexpected = [name for name in observation if name not in ANOMALY_FEATURE_NAMES]
    if missing:
        raise ValueError("missing required telemetry features: " + ", ".join(missing))
    if unexpected:
        raise ValueError("unexpected telemetry features: " + ", ".join(unexpected))
    values: list[float] = []
    for name in ANOMALY_FEATURE_NAMES:
        value = observation[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        numeric = float(value)
        if not isfinite(numeric) or numeric < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        values.append(numeric)
    return np.asarray([values], dtype=np.float64)


def score_observation(
    detector: IsolationForest, observation: TelemetryFeatures
) -> AnomalyResult:
    """Score one observation; larger transformed scores mean more unusual.

    scikit-learn's ``decision_function`` is negative for observations beyond its
    learned threshold. Negation makes the teaching score increase with unusualness;
    it remains an uncalibrated model score, not a probability.
    """
    row = _validated_row(observation)
    score = -float(detector.decision_function(row)[0])
    prediction = int(detector.predict(row)[0])
    return AnomalyResult(is_anomaly=prediction == -1, score=score)


def observation_features(observation: TelemetryObservation) -> dict[str, float]:
    """Convert a loaded observation to inference-time feature names and values."""
    return {name: float(getattr(observation, name)) for name in ANOMALY_FEATURE_NAMES}
