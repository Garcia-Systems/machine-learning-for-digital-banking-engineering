"""Deterministic threshold detection used in Chapter 0."""

from collections.abc import Mapping
from typing import TypeAlias

Observation: TypeAlias = Mapping[str, float | int]
Thresholds: TypeAlias = Mapping[str, float | int]


def find_threshold_violations(
    observation: Observation,
    thresholds: Thresholds,
) -> list[str]:
    """Return messages for observed metrics that exceed explicit thresholds.

    Missing metrics are ignored: absence is not equivalent to a measured violation.
    Results follow threshold insertion order so command output remains deterministic.
    """
    violations: list[str] = []
    for metric, limit in thresholds.items():
        value = observation.get(metric)
        if value is not None and value > limit:
            violations.append(f"{metric}: observed {value} exceeds threshold {limit}")
    return violations

