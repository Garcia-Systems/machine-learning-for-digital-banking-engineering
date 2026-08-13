"""Deterministic evaluation helpers for binary engineering signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def observations(self) -> int:
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    predicted_failures: int
    metrics: ClassificationMetrics


@dataclass(frozen=True)
class SliceMetrics:
    value: str
    count: int
    failure_rate: float
    accuracy: float
    precision: float | None
    recall: float | None


@dataclass(frozen=True)
class ProbabilityBin:
    lower: float
    upper: float
    count: int
    average_probability: float | None
    actual_failure_rate: float | None


@dataclass(frozen=True)
class EvaluationReport:
    observations: int
    successes: int
    failures: int
    baseline_accuracy: float
    default_threshold: ThresholdMetrics
    roc_auc: float
    average_precision: float
    thresholds: tuple[ThresholdMetrics, ...]
    slices: dict[str, tuple[SliceMetrics, ...]]
    probability_bins: tuple[ProbabilityBin, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _binary(values: ArrayLike, name: str) -> NDArray[np.int64]:
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    if set(np.unique(result)) - {0, 1}:
        raise ValueError(f"{name} must contain only binary labels")
    return result


def _probabilities(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError("probabilities must be finite values between 0 and 1")
    return result


def calculate_classification_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> ClassificationMetrics:
    """Calculate binary metrics with zero for undefined aggregate ratios."""
    actual, predicted = _binary(y_true, "actual labels"), _binary(y_pred, "predictions")
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("actual labels and predictions must have the same non-zero length")
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationMetrics((tp + tn) / len(actual), precision, recall, specificity, f1,
                                 tp, tn, fp, fn)


def apply_threshold(probabilities: ArrayLike, threshold: float) -> NDArray[np.int64]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return (_probabilities(probabilities) >= threshold).astype(np.int64)


def evaluate_threshold(y_true: ArrayLike, probabilities: ArrayLike, threshold: float) -> ThresholdMetrics:
    scores = _probabilities(probabilities)
    predictions = apply_threshold(scores, threshold)
    metrics = calculate_classification_metrics(y_true, predictions)
    return ThresholdMetrics(float(threshold), int(predictions.sum()), metrics)


def evaluate_thresholds(y_true: ArrayLike, probabilities: ArrayLike,
                        thresholds: Sequence[float]) -> tuple[ThresholdMetrics, ...]:
    ordered = sorted(float(value) for value in thresholds)
    return tuple(evaluate_threshold(y_true, probabilities, value) for value in ordered)


def select_threshold_for_minimum_recall(results: Sequence[ThresholdMetrics],
                                        minimum_recall: float) -> ThresholdMetrics:
    """Choose highest precision, breaking ties toward the higher threshold."""
    if not 0 <= minimum_recall <= 1:
        raise ValueError("minimum_recall must be between 0 and 1")
    eligible = [item for item in results if item.metrics.recall >= minimum_recall]
    if not eligible:
        raise ValueError("no evaluated threshold meets minimum_recall")
    return max(eligible, key=lambda item: (item.metrics.precision, item.threshold))


def calculate_majority_baseline(y_true: ArrayLike) -> ClassificationMetrics:
    actual = _binary(y_true, "actual labels")
    if len(actual) == 0:
        raise ValueError("actual labels cannot be empty")
    majority = 1 if int(actual.sum()) > len(actual) / 2 else 0
    return calculate_classification_metrics(actual, np.full(len(actual), majority))


def calculate_ranking_metrics(y_true: ArrayLike, probabilities: ArrayLike) -> tuple[float, float]:
    actual, scores = _binary(y_true, "actual labels"), _probabilities(probabilities)
    if len(actual) != len(scores):
        raise ValueError("actual labels and probabilities must have the same length")
    if len(np.unique(actual)) != 2:
        raise ValueError("ranking metrics require both classes")
    return float(roc_auc_score(actual, scores)), float(average_precision_score(actual, scores))


def collect_error_examples(rows: Sequence[Mapping[str, object]], y_true: ArrayLike,
                           probabilities: ArrayLike, threshold: float = 0.5,
                           limit: int = 3) -> dict[str, tuple[dict[str, object], ...]]:
    """Return the most confident false positives and false negatives."""
    actual, scores = _binary(y_true, "actual labels"), _probabilities(probabilities)
    if limit < 0 or len(rows) != len(actual) or len(actual) != len(scores):
        raise ValueError("rows, labels, and probabilities must align and limit must be non-negative")
    predicted = apply_threshold(scores, threshold)

    def example(index: int) -> dict[str, object]:
        return {**dict(rows[index]), "probability": float(scores[index]),
                "actual": "failure" if actual[index] else "success"}

    false_positive_indices = np.flatnonzero((actual == 0) & (predicted == 1))
    false_negative_indices = np.flatnonzero((actual == 1) & (predicted == 0))
    false_positive_indices = sorted(false_positive_indices, key=lambda i: (-scores[i], int(i)))
    false_negative_indices = sorted(false_negative_indices, key=lambda i: (scores[i], int(i)))
    return {
        "false_positives": tuple(example(int(i)) for i in false_positive_indices[:limit]),
        "false_negatives": tuple(example(int(i)) for i in false_negative_indices[:limit]),
    }


def evaluate_slices(values: Sequence[object], y_true: ArrayLike, probabilities: ArrayLike,
                    threshold: float = 0.5) -> tuple[SliceMetrics, ...]:
    actual, scores = _binary(y_true, "actual labels"), _probabilities(probabilities)
    labels = np.asarray([str(value) for value in values], dtype=object)
    if len(labels) != len(actual) or len(actual) != len(scores):
        raise ValueError("slice values, labels, and probabilities must align")
    predicted = apply_threshold(scores, threshold)
    output = []
    for value in sorted(set(labels)):
        mask = labels == value
        metrics = calculate_classification_metrics(actual[mask], predicted[mask])
        positives = int(actual[mask].sum())
        predicted_positives = int(predicted[mask].sum())
        output.append(SliceMetrics(str(value), int(mask.sum()), float(actual[mask].mean()),
                                  metrics.accuracy,
                                  metrics.precision if predicted_positives else None,
                                  metrics.recall if positives else None))
    return tuple(output)


def build_probability_bins(y_true: ArrayLike, probabilities: ArrayLike,
                           edges: Sequence[float] = (0, .2, .4, .6, .8, 1)) -> tuple[ProbabilityBin, ...]:
    actual, scores = _binary(y_true, "actual labels"), _probabilities(probabilities)
    boundaries = tuple(float(edge) for edge in edges)
    if len(actual) != len(scores) or len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != 1:
        raise ValueError("aligned inputs and bin edges spanning 0 through 1 are required")
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("bin edges must be strictly increasing")
    bins = []
    for index, (lower, upper) in enumerate(zip(boundaries, boundaries[1:])):
        mask = (scores >= lower) & ((scores <= upper) if index == len(boundaries) - 2 else (scores < upper))
        count = int(mask.sum())
        bins.append(ProbabilityBin(lower, upper, count,
                                   float(scores[mask].mean()) if count else None,
                                   float(actual[mask].mean()) if count else None))
    return tuple(bins)


def build_evaluation_report(y_true: ArrayLike, probabilities: ArrayLike,
                            slice_values: Mapping[str, Sequence[object]],
                            threshold: float = .5,
                            thresholds: Sequence[float] = tuple(np.arange(.1, 1, .1))) -> EvaluationReport:
    actual, scores = _binary(y_true, "actual labels"), _probabilities(probabilities)
    sweep = evaluate_thresholds(actual, scores, thresholds)
    default = evaluate_threshold(actual, scores, threshold)
    auc, average_precision = calculate_ranking_metrics(actual, scores)
    slices = {name: evaluate_slices(values, actual, scores, threshold)
              for name, values in slice_values.items()}
    bins = build_probability_bins(actual, scores)
    baseline = calculate_majority_baseline(actual)
    return EvaluationReport(len(actual), int((actual == 0).sum()), int(actual.sum()),
                            baseline.accuracy, default, auc, average_precision,
                            sweep, slices, bins)
