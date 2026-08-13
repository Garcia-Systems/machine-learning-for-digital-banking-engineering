"""Supervised, leakage-safe operational review routing for Chapter 13."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .transaction_anomaly import AMOUNT_BANDS, TRANSACTION_CHANNELS, TRANSACTION_TYPES

NUMERIC_REVIEW_FEATURES = ("hour_of_day", "recent_transaction_count",
    "minutes_since_previous_transaction", "distance_from_recent_pattern",
    "recent_failed_transaction_count")
BOOLEAN_REVIEW_FEATURES = ("device_change",)
CATEGORICAL_REVIEW_FEATURES = ("transaction_type", "channel", "amount_band")
REVIEW_FEATURES = (*NUMERIC_REVIEW_FEATURES, *BOOLEAN_REVIEW_FEATURES,
                   *CATEGORICAL_REVIEW_FEATURES)
REVIEW_TARGET = "manual_review_required"
REVIEW_RANDOM_STATE = 1313
LEAKAGE_FIELDS = frozenset({"review_outcome", "review_notes", "case_closed_reason",
                            "confirmed_issue", REVIEW_TARGET, "investigator_disposition"})
FeatureMatrix = NDArray[np.object_]
TargetVector = NDArray[np.int64]


@dataclass(frozen=True)
class ReviewObservation:
    transaction_type: str
    channel: str
    amount_band: str
    hour_of_day: int
    recent_transaction_count: int
    minutes_since_previous_transaction: float
    device_change: bool
    distance_from_recent_pattern: float
    recent_failed_transaction_count: int


@dataclass(frozen=True)
class LabeledReviewObservation:
    timestamp: datetime
    observation: ReviewObservation
    manual_review_required: int


@dataclass(frozen=True)
class ReviewDatasetSplit:
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector


@dataclass(frozen=True)
class ReviewEvaluation:
    threshold: float
    accuracy: float
    confusion_matrix: NDArray[np.int64]
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    precision: float
    recall: float
    predicted_review_count: int


@dataclass(frozen=True)
class ReviewPrediction:
    """A historical review-routing estimate, never a wrongdoing probability."""
    probability: float
    predicted_review: bool


def load_review_dataset(path: str | Path) -> list[LabeledReviewObservation]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("review dataset must contain observations")
    result = []
    for number, row in enumerate(rows, 2):
        try:
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            transaction_type, channel = row["transaction_type"].strip(), row["channel"].strip()
            amount_band = row["amount_band"].strip()
            hour, count = int(row["hour_of_day"]), int(row["recent_transaction_count"])
            minutes = float(row["minutes_since_previous_transaction"])
            raw_bool = row["device_change"].strip().lower()
            if raw_bool not in {"true", "false"}:
                raise ValueError
            distance = float(row["distance_from_recent_pattern"])
            failures, label = int(row["recent_failed_transaction_count"]), int(row[REVIEW_TARGET])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"row {number}: malformed review observation") from error
        if transaction_type not in TRANSACTION_TYPES or channel not in TRANSACTION_CHANNELS \
                or amount_band not in AMOUNT_BANDS:
            raise ValueError(f"row {number}: invalid categorical review value")
        if not 0 <= hour <= 23 or count < 0 or failures < 0 or label not in {0, 1} \
                or not isfinite(minutes) or minutes < 0 or not isfinite(distance) \
                or not 0 <= distance <= 1:
            raise ValueError(f"row {number}: invalid numeric review value")
        result.append(LabeledReviewObservation(timestamp, ReviewObservation(
            transaction_type, channel, amount_band, hour, count, minutes,
            raw_bool == "true", distance, failures), label))
    return result


def build_review_features(rows: Sequence[LabeledReviewObservation | ReviewObservation]) -> FeatureMatrix:
    observations = [row.observation if isinstance(row, LabeledReviewObservation) else row for row in rows]
    return np.asarray([[getattr(row, feature) for feature in REVIEW_FEATURES]
                       for row in observations], dtype=object).reshape(len(rows), len(REVIEW_FEATURES))


def build_review_targets(rows: Sequence[LabeledReviewObservation]) -> TargetVector:
    return np.asarray([row.manual_review_required for row in rows], dtype=np.int64)


def calculate_majority_baseline(targets: TargetVector) -> float:
    if len(targets) == 0:
        raise ValueError("cannot calculate a baseline without targets")
    return float(np.bincount(targets, minlength=2).max() / len(targets))


def split_review_dataset(X: FeatureMatrix, y: TargetVector, *, test_size: float = .25,
                         random_state: int = REVIEW_RANDOM_STATE) -> ReviewDatasetSplit:
    values = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return ReviewDatasetSplit(*values)


def build_review_pipeline() -> Pipeline:
    numeric = tuple(range(5)); boolean = (5,); categorical = (6, 7, 8)
    return Pipeline([("preprocessor", ColumnTransformer([
        ("numeric", StandardScaler(), numeric), ("boolean", "passthrough", boolean),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)])),
        ("classifier", LogisticRegression(random_state=REVIEW_RANDOM_STATE, max_iter=1_000))])


def train_review_model(model: Pipeline, X_train: FeatureMatrix, y_train: TargetVector) -> Pipeline:
    if X_train.ndim != 2 or X_train.shape[1] != len(REVIEW_FEATURES):
        raise ValueError(f"features must have {len(REVIEW_FEATURES)} columns")
    if len(X_train) == 0 or set(np.unique(y_train)) != {0, 1}:
        raise ValueError("training data must contain both binary target classes")
    model.fit(X_train, y_train)
    return model


def predict_review_probabilities(model: Pipeline, X: FeatureMatrix) -> NDArray[np.float64]:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def apply_review_threshold(probability: float, *, threshold: float = .5) -> bool:
    if not 0 <= probability <= 1 or not 0 <= threshold <= 1:
        raise ValueError("probability and threshold must be between 0 and 1")
    return probability >= threshold


def evaluate_review_model(y_true: TargetVector, probabilities: NDArray[np.float64],
                          *, threshold: float = .5) -> ReviewEvaluation:
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("targets and probabilities must have equal nonzero length")
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = np.asarray(confusion_matrix(y_true, predictions, labels=(0, 1)), dtype=np.int64)
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return ReviewEvaluation(threshold, float(np.mean(predictions == y_true)), matrix,
        tn, fp, fn, tp, float(precision_score(y_true, predictions, zero_division=0)),
        float(recall_score(y_true, predictions, zero_division=0)), int(predictions.sum()))


def predict_review_probability(model: Pipeline, observation: ReviewObservation) -> float:
    return float(predict_review_probabilities(model, build_review_features([observation]))[0])


def predict_review(model: Pipeline, observation: ReviewObservation, *, threshold: float = .5) -> ReviewPrediction:
    probability = predict_review_probability(model, observation)
    return ReviewPrediction(probability, apply_review_threshold(probability, threshold=threshold))
