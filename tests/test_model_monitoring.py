from dataclasses import fields
from datetime import datetime, timezone

import pytest

from harbor_ml.integration_failure_model import load_integration_requests
from harbor_ml.model_monitoring import (
    PredictionRecord, attach_labels, build_training_monitoring_baseline,
    calculate_categorical_drift, calculate_numeric_drift, calculate_prediction_drift,
    compare_performance_to_baseline, compare_shadow_predictions, drift_persists, latency_percentile,
    simulate_production_periods, summarize_prediction_window,
)


@pytest.fixture
def baseline():
    rows = load_integration_requests("data/harbor_integration_requests.csv")
    return build_training_monitoring_baseline(
        rows, model_name="harbor-integration-failure", model_version="v1",
        dataset_sha256="abc", created_at="2025-01-01T00:00:00Z")


def record(index, *, actual=None, vendor="ClearVerify", endpoint="account_summary"):
    return PredictionRecord(str(index), datetime(2025, 1, 1, tzinfo=timezone.utc),
                            "harbor-integration-failure", "v1", "review-policy-v2",
                            vendor, endpoint, .8 if index % 2 else .2, bool(index % 2), actual)


def test_baseline_contains_numeric_summaries_and_normalized_categories(baseline):
    assert baseline.numeric_summaries["recent_vendor_latency_ms"].maximum > baseline.numeric_summaries["recent_vendor_latency_ms"].minimum
    assert set(baseline.numeric_summaries["queue_depth"].quantiles) == {"p25", "p50", "p75", "p95"}
    assert sum(baseline.categorical_frequencies["vendor"].values()) == pytest.approx(1)
    assert baseline.model_version == "v1"


def test_numeric_drift_and_zero_variance():
    from harbor_ml.model_monitoring import NumericBaseline
    ordinary = NumericBaseline(10, 2, 5, 15, {})
    assert calculate_numeric_drift("x", ordinary, [12, 12]).standardized_mean_shift == 1
    constant = NumericBaseline(10, 0, 10, 10, {})
    assert calculate_numeric_drift("x", constant, [10]).standardized_mean_shift == 0
    assert calculate_numeric_drift("x", constant, [11]).investigate


def test_categorical_frequency_comparison():
    results = calculate_categorical_drift("vendor", {"A": .5, "B": .5}, ["A"] * 3 + ["C"])
    assert next(item for item in results if item.category == "C").difference == .25


def test_summary_delayed_labels_unknowns_and_versions(baseline):
    original = (record(0), record(1, endpoint="new-endpoint"))
    insufficient = summarize_prediction_window(original, baseline, minimum_labeled=2)
    assert insufficient.performance_metrics is None
    assert insufficient.actual_positive_rate is None
    assert insufficient.unknown_endpoint_count == 1
    labeled = attach_labels(original, {"0": False, "1": True})
    summary = summarize_prediction_window(labeled, baseline, minimum_labeled=2)
    assert original[0].actual_failure is None
    assert summary.performance_metrics is not None
    assert summary.model_version == "v1" and summary.policy_version == "review-policy-v2"
    assert summary.average_probability == pytest.approx(.5)


def test_prediction_drift_latency_shadow_and_persistence():
    drift = calculate_prediction_drift([.1, .2], [.7, .8])
    assert drift.investigate and drift.current_positive_rate == 1
    assert latency_percentile([10, 20, 30, 40], 50) == 20
    assert latency_percentile([10, 20, 30, 40], 95) == 40
    shadow = compare_shadow_predictions([.1, .8], [.2, .4])
    assert shadow.average_absolute_probability_difference == pytest.approx(.25)
    assert shadow.disagreement_rate == .5
    assert compare_performance_to_baseline("recall", .8, .5).investigate
    assert not drift_persists([True]) and not drift_persists([True, True])
    assert drift_persists([False, True, True, True])


def test_simulation_is_deterministic_and_privacy_minimized():
    assert simulate_production_periods() == simulate_production_periods()
    assert set(simulate_production_periods()) == {"A", "B", "C", "D"}
    names = {field.name for field in fields(PredictionRecord)}
    assert not names.intersection({"member_id", "member_name", "account_number", "email"})
