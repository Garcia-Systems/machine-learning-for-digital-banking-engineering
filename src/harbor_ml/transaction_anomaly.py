"""Privacy-minimized transaction anomaly detection for Chapter 12."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TRANSACTION_TYPES = (
    "debit_purchase", "bill_payment", "internal_transfer", "external_transfer",
    "atm_withdrawal", "deposit",
)
TRANSACTION_CHANNELS = ("web", "mobile", "atm", "branch_assisted")
AMOUNT_BANDS = (
    "under_25", "25_to_99", "100_to_499", "500_to_999",
    "1000_to_2499", "2500_plus",
)
CATEGORICAL_TRANSACTION_FEATURES = ("transaction_type", "channel", "amount_band")
NUMERIC_TRANSACTION_FEATURES = (
    "hour_of_day", "recent_transaction_count", "minutes_since_previous_transaction",
    "distance_from_recent_pattern",
)
BOOLEAN_TRANSACTION_FEATURES = ("device_change",)
TRANSACTION_FEATURES = (
    *CATEGORICAL_TRANSACTION_FEATURES, *NUMERIC_TRANSACTION_FEATURES,
    *BOOLEAN_TRANSACTION_FEATURES,
)
TRANSACTION_RANDOM_STATE = 1212
TRANSACTION_CONTAMINATION = 0.04
FeatureMatrix = NDArray[np.object_]


@dataclass(frozen=True)
class TransactionObservation:
    timestamp: datetime
    transaction_type: str
    channel: str
    amount_band: str
    hour_of_day: int
    recent_transaction_count: int
    minutes_since_previous_transaction: float
    device_change: bool
    distance_from_recent_pattern: float


@dataclass(frozen=True)
class TransactionScenario:
    name: str
    observation: TransactionObservation


@dataclass(frozen=True)
class TransactionAnomalyResult:
    """An uncalibrated unusualness score and the fitted detector's decision."""

    raw_score: float
    is_anomaly: bool


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError("device_change must be true or false")
    return normalized == "true"


def _parse_row(row: Mapping[str, str], number: int) -> TransactionObservation:
    try:
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        transaction_type = row["transaction_type"].strip()
        channel = row["channel"].strip()
        amount_band = row["amount_band"].strip()
        hour = int(row["hour_of_day"])
        count = int(row["recent_transaction_count"])
        minutes = float(row["minutes_since_previous_transaction"])
        changed = _parse_bool(row["device_change"])
        distance = float(row["distance_from_recent_pattern"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {number}: malformed transaction observation") from error
    if transaction_type not in TRANSACTION_TYPES:
        raise ValueError(f"row {number}: invalid transaction_type")
    if channel not in TRANSACTION_CHANNELS:
        raise ValueError(f"row {number}: invalid channel")
    if amount_band not in AMOUNT_BANDS:
        raise ValueError(f"row {number}: invalid amount_band")
    if not 0 <= hour <= 23 or count < 0:
        raise ValueError(f"row {number}: invalid transaction count or hour")
    if not isfinite(minutes) or minutes < 0 or not isfinite(distance) or not 0 <= distance <= 1:
        raise ValueError(f"row {number}: invalid numeric transaction value")
    return TransactionObservation(
        timestamp, transaction_type, channel, amount_band, hour, count, minutes,
        changed, distance,
    )


def _load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("transaction dataset must contain observations")
    return rows


def load_transaction_observations(path: str | Path) -> list[TransactionObservation]:
    """Load the unlabeled synthetic baseline used for unsupervised fitting."""
    return [_parse_row(row, number) for number, row in enumerate(_load_rows(path), 2)]


def load_transaction_scenarios(path: str | Path) -> list[TransactionScenario]:
    """Load named evaluation scenarios that remain separate from training."""
    scenarios = []
    for number, row in enumerate(_load_rows(path), 2):
        name = row.get("scenario", "").strip()
        if not name:
            raise ValueError(f"row {number}: scenario is required")
        scenarios.append(TransactionScenario(name, _parse_row(row, number)))
    return scenarios


def build_transaction_features(observations: Sequence[TransactionObservation]) -> FeatureMatrix:
    if not observations:
        raise ValueError("transaction features require at least one observation")
    expected = {field.name for field in fields(TransactionObservation)}
    if not set(TRANSACTION_FEATURES) < expected:
        raise RuntimeError("transaction feature contract does not match observation schema")
    return np.asarray(
        [[getattr(row, feature) for feature in TRANSACTION_FEATURES] for row in observations],
        dtype=object,
    )


def _boolean_to_float(values: NDArray[np.object_]) -> NDArray[np.float64]:
    return values.astype(float)


def build_transaction_preprocessor() -> ColumnTransformer:
    """Encode categories densely, scale numbers, and pass the Boolean as 0/1."""
    categorical = tuple(range(3))
    numeric = tuple(range(3, 7))
    boolean = (7,)
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", StandardScaler(), numeric),
        ("boolean", FunctionTransformer(_boolean_to_float), boolean),
    ])


def build_transaction_anomaly_pipeline(
    *, contamination: float = TRANSACTION_CONTAMINATION,
    random_state: int = TRANSACTION_RANDOM_STATE,
) -> Pipeline:
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5")
    return Pipeline([
        ("preprocessor", build_transaction_preprocessor()),
        ("detector", IsolationForest(contamination=contamination, random_state=random_state)),
    ])


def train_transaction_anomaly_detector(
    model: Pipeline, observations: Sequence[TransactionObservation],
) -> Pipeline:
    """Fit only the supplied baseline; evaluation scenarios are not accepted here."""
    model.fit(build_transaction_features(observations))
    return model


def score_transactions(
    model: Pipeline, observations: Sequence[TransactionObservation],
) -> list[TransactionAnomalyResult]:
    """Return scores where larger values mean more unusual, never probabilities.

    scikit-learn's decision function is positive on the inlier side of its learned
    cutoff. Negating it puts more unusual observations in the positive direction;
    zero is the detector's fitted normal/anomaly boundary.
    """
    matrix = build_transaction_features(observations)
    scores = -np.asarray(model.decision_function(matrix), dtype=float)
    predictions = np.asarray(model.predict(matrix), dtype=int)
    return [
        TransactionAnomalyResult(float(score), prediction == -1)
        for score, prediction in zip(scores, predictions, strict=True)
    ]


def score_transaction(
    model: Pipeline, observation: TransactionObservation,
) -> TransactionAnomalyResult:
    return score_transactions(model, [observation])[0]
