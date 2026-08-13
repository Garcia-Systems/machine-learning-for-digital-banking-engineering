"""Small, deterministic production-model monitoring primitives for Chapter 25."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from math import ceil, isfinite
from typing import Mapping, Sequence

import numpy as np

from .integration_failure_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, IntegrationObservation
from .model_evaluation import ClassificationMetrics, calculate_classification_metrics


@dataclass(frozen=True)
class NumericBaseline:
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    quantiles: dict[str, float]


@dataclass(frozen=True)
class MonitoringBaseline:
    model_name: str
    model_version: str
    dataset_sha256: str
    created_at: str
    numeric_summaries: dict[str, NumericBaseline]
    categorical_frequencies: dict[str, dict[str, float]]


@dataclass(frozen=True)
class NumericDriftResult:
    feature: str
    baseline_mean: float
    current_mean: float
    baseline_std: float
    standardized_mean_shift: float
    investigate: bool


@dataclass(frozen=True)
class CategoryDriftResult:
    feature: str
    category: str
    baseline_frequency: float
    current_frequency: float
    difference: float
    investigate: bool


@dataclass(frozen=True)
class PredictionRecord:
    """Privacy-minimized facts recorded when a prediction is made."""

    prediction_id: str
    prediction_timestamp: datetime
    model_name: str
    model_version: str
    policy_version: str
    vendor: str
    endpoint: str
    failure_probability: float
    predicted_failure: bool
    actual_failure: bool | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.failure_probability <= 1 or not isfinite(self.failure_probability):
            raise ValueError("failure_probability must be finite and between 0 and 1")
        if self.prediction_timestamp.tzinfo is None:
            raise ValueError("prediction_timestamp must be timezone-aware")


@dataclass(frozen=True)
class ServiceHealth:
    prediction_requests_total: int
    prediction_success_total: int
    prediction_error_total: int
    prediction_latency_ms: tuple[float, ...]
    model_loaded: bool


@dataclass(frozen=True)
class PredictionDriftResult:
    baseline_average_probability: float
    current_average_probability: float
    average_probability_difference: float
    baseline_positive_rate: float
    current_positive_rate: float
    positive_rate_difference: float
    investigate: bool


@dataclass(frozen=True)
class ModelMonitoringSummary:
    model_name: str
    model_version: str
    policy_version: str
    prediction_count: int
    labeled_count: int
    average_probability: float
    predicted_positive_rate: float
    actual_positive_rate: float | None
    performance_metrics: ClassificationMetrics | None
    performance_status: str
    unknown_vendor_count: int
    unknown_endpoint_count: int
    unknown_category_count: int
    drift_detected: bool
    status: str


@dataclass(frozen=True)
class ShadowComparison:
    observation_count: int
    average_absolute_probability_difference: float
    disagreement_rate: float


@dataclass(frozen=True)
class PerformanceComparison:
    metric: str
    baseline_value: float
    current_value: float
    decrease: float
    investigate: bool


def build_training_monitoring_baseline(
    observations: Sequence[IntegrationObservation], *, model_name: str, model_version: str,
    dataset_sha256: str, created_at: str
) -> MonitoringBaseline:
    """Derive a versioned baseline from the same observations used for training."""
    if not observations:
        raise ValueError("training observations cannot be empty")
    numeric: dict[str, NumericBaseline] = {}
    for feature in NUMERIC_FEATURES:
        values = np.asarray([getattr(row, feature) for row in observations], dtype=float)
        numeric[feature] = NumericBaseline(
            float(values.mean()), float(values.std(ddof=0)), float(values.min()), float(values.max()),
            {name: float(np.quantile(values, value)) for name, value in
             (("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))},
        )
    categorical = {}
    for feature in CATEGORICAL_FEATURES:
        values = [str(getattr(row, feature)) for row in observations]
        categorical[feature] = {
            category: values.count(category) / len(values) for category in sorted(set(values))
        }
    return MonitoringBaseline(model_name, model_version, dataset_sha256, created_at,
                              numeric, categorical)


def monitoring_baseline_dict(baseline: MonitoringBaseline) -> dict[str, object]:
    """Return a JSON-serializable baseline representation."""
    return asdict(baseline)


def calculate_numeric_drift(feature: str, baseline: NumericBaseline,
                            current_values: Sequence[float], *, threshold: float = 1.0
                            ) -> NumericDriftResult:
    if not current_values or threshold < 0:
        raise ValueError("current values are required and threshold must be non-negative")
    current_mean = float(np.mean(np.asarray(current_values, dtype=float)))
    difference = current_mean - baseline.mean
    if baseline.standard_deviation == 0:
        shift = 0.0 if difference == 0 else (float("inf") if difference > 0 else float("-inf"))
    else:
        shift = difference / baseline.standard_deviation
    return NumericDriftResult(feature, baseline.mean, current_mean,
                              baseline.standard_deviation, shift, abs(shift) >= threshold)


def calculate_categorical_drift(feature: str, baseline_frequencies: Mapping[str, float],
                                current_values: Sequence[str], *, threshold: float = .20
                                ) -> tuple[CategoryDriftResult, ...]:
    if not current_values or threshold < 0:
        raise ValueError("current values are required and threshold must be non-negative")
    categories = sorted(set(baseline_frequencies) | set(current_values))
    size = len(current_values)
    return tuple(CategoryDriftResult(
        feature, category, float(baseline_frequencies.get(category, 0)),
        current_values.count(category) / size,
        current_values.count(category) / size - float(baseline_frequencies.get(category, 0)),
        abs(current_values.count(category) / size - float(baseline_frequencies.get(category, 0))) >= threshold,
    ) for category in categories)


def attach_outcome(records: Sequence[PredictionRecord], prediction_id: str,
                   actual_failure: bool) -> tuple[PredictionRecord, ...]:
    """Return a new collection with one eventual outcome attached; never mutate input."""
    matches = sum(record.prediction_id == prediction_id for record in records)
    if matches != 1:
        raise ValueError("prediction_id must identify exactly one record")
    return tuple(replace(record, actual_failure=bool(actual_failure))
                 if record.prediction_id == prediction_id else record for record in records)


def attach_labels(records: Sequence[PredictionRecord], outcomes: Mapping[str, bool]
                  ) -> tuple[PredictionRecord, ...]:
    unknown = set(outcomes) - {record.prediction_id for record in records}
    if unknown:
        raise ValueError("outcomes contain unknown prediction identifiers")
    return tuple(replace(record, actual_failure=bool(outcomes[record.prediction_id]))
                 if record.prediction_id in outcomes else record for record in records)


def summarize_prediction_window(records: Sequence[PredictionRecord], baseline: MonitoringBaseline,
                                *, minimum_labeled: int = 20,
                                drift_detected: bool = False) -> ModelMonitoringSummary:
    if not records or minimum_labeled < 1:
        raise ValueError("records are required and minimum_labeled must be positive")
    identities = {(r.model_name, r.model_version, r.policy_version) for r in records}
    if len(identities) != 1:
        raise ValueError("a monitoring window cannot mix model or policy versions")
    model_name, model_version, policy_version = identities.pop()
    if (model_name, model_version) != (baseline.model_name, baseline.model_version):
        raise ValueError("monitoring baseline does not match the model version")
    known_vendors = set(baseline.categorical_frequencies["vendor"])
    known_endpoints = set(baseline.categorical_frequencies["endpoint"])
    unknown_vendors = sum(r.vendor not in known_vendors for r in records)
    unknown_endpoints = sum(r.endpoint not in known_endpoints for r in records)
    labeled = [r for r in records if r.actual_failure is not None]
    enough = len(labeled) >= minimum_labeled
    metrics = calculate_classification_metrics(
        [int(bool(r.actual_failure)) for r in labeled], [int(r.predicted_failure) for r in labeled]
    ) if enough else None
    status = "investigate" if drift_detected else ("healthy" if enough else "insufficient_data")
    return ModelMonitoringSummary(
        model_name, model_version, policy_version, len(records), len(labeled),
        float(np.mean([r.failure_probability for r in records])),
        float(np.mean([r.predicted_failure for r in records])),
        float(np.mean([r.actual_failure for r in labeled])) if labeled else None,
        metrics, "available" if enough else "insufficient_labels", unknown_vendors,
        unknown_endpoints, unknown_vendors + unknown_endpoints, drift_detected, status,
    )


def calculate_prediction_drift(baseline_probabilities: Sequence[float],
                               current_probabilities: Sequence[float], *, threshold: float = .15,
                               classification_threshold: float = .5) -> PredictionDriftResult:
    if not baseline_probabilities or not current_probabilities:
        raise ValueError("both probability windows are required")
    before, current = np.asarray(baseline_probabilities), np.asarray(current_probabilities)
    before_mean, current_mean = float(before.mean()), float(current.mean())
    before_rate = float(np.mean(before >= classification_threshold))
    current_rate = float(np.mean(current >= classification_threshold))
    return PredictionDriftResult(before_mean, current_mean, current_mean - before_mean,
                                 before_rate, current_rate, current_rate - before_rate,
                                 abs(current_mean - before_mean) >= threshold or
                                 abs(current_rate - before_rate) >= threshold)


def latency_percentile(values: Sequence[float], percentile: float) -> float:
    """Nearest-rank operational percentile (p50/p95), with no external dependency."""
    if not values or not 0 < percentile <= 100:
        raise ValueError("latencies are required and percentile must be in (0, 100]")
    ordered = sorted(float(value) for value in values)
    if any(value < 0 or not isfinite(value) for value in ordered):
        raise ValueError("latencies must be finite and non-negative")
    return ordered[max(0, ceil(percentile / 100 * len(ordered)) - 1)]


def compare_shadow_predictions(production_probabilities: Sequence[float],
                               candidate_probabilities: Sequence[float], *, threshold: float = .5
                               ) -> ShadowComparison:
    if not production_probabilities or len(production_probabilities) != len(candidate_probabilities):
        raise ValueError("production and candidate scores must align and be non-empty")
    production, candidate = np.asarray(production_probabilities), np.asarray(candidate_probabilities)
    return ShadowComparison(len(production), float(np.mean(np.abs(production - candidate))),
                            float(np.mean((production >= threshold) != (candidate >= threshold))))


def compare_performance_to_baseline(metric: str, baseline_value: float, current_value: float,
                                    *, maximum_decrease: float = .15) -> PerformanceComparison:
    """Apply an explicit educational degradation tolerance to one approved metric."""
    if metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError("unsupported classification metric")
    if not all(0 <= value <= 1 for value in (baseline_value, current_value, maximum_decrease)):
        raise ValueError("performance values and tolerance must be between 0 and 1")
    decrease = baseline_value - current_value
    return PerformanceComparison(metric, baseline_value, current_value, decrease,
                                 decrease >= maximum_decrease)


def drift_persists(windows: Sequence[bool], *, consecutive_windows: int = 3) -> bool:
    if consecutive_windows < 1:
        raise ValueError("consecutive_windows must be positive")
    return len(windows) >= consecutive_windows and all(windows[-consecutive_windows:])


def simulate_production_periods() -> dict[str, tuple[IntegrationObservation, ...]]:
    """Create four reproducible, fictional periods without member identity."""
    start = datetime(2025, 2, 1, tzinfo=timezone.utc)
    periods: dict[str, tuple[IntegrationObservation, ...]] = {}
    for period in "ABCD":
        rows = []
        vendors = ("ClearVerify", "Northstar Payments", "BlueCurrent Documents",
                   "HarborLink Core Gateway")
        endpoints = ("account_summary", "identity_document_upload", "identity_verify",
                     "notice_fetch", "statement_fetch", "transaction_history",
                     "transfer_status", "transfer_submit")
        for index in range(40):
            vendor = ("ClearVerify" if period == "B" and index % 5 < 4
                      else vendors[index % len(vendors)])
            endpoint = endpoints[index % len(endpoints)]
            if period == "C" and index % 4 == 0:
                endpoint = "real-time-payments"
            latency = 310 + (index % 9) * 18 + (520 if period == "B" else 0)
            error_rate = .05 + (index % 5) * .015
            probability_signal = latency > 700 or error_rate > .09 or index % 7 == 0
            outcome = probability_signal
            if period == "D":
                outcome = not probability_signal if index % 2 == 0 else probability_signal
            rows.append(IntegrationObservation(
                start + timedelta(hours=index), vendor, endpoint, latency, error_rate,
                8 + index % 20, index % 3, 900 + index * 17, index % 24, int(outcome)))
        periods[period] = tuple(rows)
        start += timedelta(days=2)
    return periods
