from dataclasses import dataclass

import pytest

from harbor_ml.fairness_slices import evaluate_technical_slices


@dataclass(frozen=True)
class Observation:
    vendor: str
    endpoint: str
    request_failed: int


def test_technical_slice_confusion_metrics_and_low_support():
    rows = [Observation("A", "verify", 1), Observation("A", "verify", 1),
            Observation("A", "verify", 0), Observation("A", "verify", 0)]
    metric = evaluate_technical_slices(rows, [.9, .2, .8, .1], features=("vendor",),
                                       minimum_support=5)[0]
    assert metric.support == 4 and metric.base_rate == .5
    assert metric.precision == metric.recall == .5
    assert metric.false_positive_rate == metric.false_negative_rate == .5
    assert metric.low_support


def test_undefined_rates_are_explicit_and_unsafe_slices_are_rejected():
    rows = [Observation("A", "verify", 0), Observation("A", "verify", 0)]
    metric = evaluate_technical_slices(rows, [.1, .2], features=("endpoint",))[0]
    assert metric.precision is None and metric.recall is None
    assert metric.false_positive_rate == 0 and metric.false_negative_rate is None
    with pytest.raises(ValueError, match="unsupported technical slice"):
        evaluate_technical_slices(rows, [.1, .2], features=("member_value",))
