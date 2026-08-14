"""Technical slice evaluation for Chapter 23's responsible-ML laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TechnicalSliceMetric:
    """Classification behavior for a safe operational context, not a person."""

    feature: str
    value: str
    support: int
    base_rate: float
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    low_support: bool


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_technical_slices(
    observations: Sequence[object],
    probabilities: Sequence[float],
    *,
    features: Sequence[str] = ("vendor", "endpoint"),
    threshold: float = 0.5,
    minimum_support: int = 30,
) -> tuple[TechnicalSliceMetric, ...]:
    """Evaluate declared technical slices without introducing demographic fields."""
    if len(observations) != len(probabilities):
        raise ValueError("observations and probabilities must align")
    if not observations:
        raise ValueError("at least one observation is required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if minimum_support < 1:
        raise ValueError("minimum_support must be positive")

    metrics: list[TechnicalSliceMetric] = []
    for feature in features:
        if feature not in {"vendor", "endpoint", "channel"}:
            raise ValueError(f"unsupported technical slice: {feature}")
        if any(not hasattr(row, feature) for row in observations):
            raise ValueError(f"observations do not provide technical slice: {feature}")
        values = sorted({str(getattr(row, feature)) for row in observations})
        for value in values:
            indexes = [i for i, row in enumerate(observations)
                       if str(getattr(row, feature)) == value]
            actual = [bool(getattr(observations[i], "request_failed")) for i in indexes]
            predicted = [probabilities[i] >= threshold for i in indexes]
            tp = sum(was and forecast for was, forecast in zip(actual, predicted, strict=True))
            tn = sum(not was and not forecast
                     for was, forecast in zip(actual, predicted, strict=True))
            fp = sum(not was and forecast for was, forecast in zip(actual, predicted, strict=True))
            fn = sum(was and not forecast for was, forecast in zip(actual, predicted, strict=True))
            support = len(indexes)
            metrics.append(TechnicalSliceMetric(
                feature=feature,
                value=value,
                support=support,
                base_rate=sum(actual) / support,
                precision=_safe_ratio(tp, tp + fp),
                recall=_safe_ratio(tp, tp + fn),
                false_positive_rate=_safe_ratio(fp, fp + tn),
                false_negative_rate=_safe_ratio(fn, fn + tp),
                low_support=support < minimum_support,
            ))
    return tuple(metrics)
