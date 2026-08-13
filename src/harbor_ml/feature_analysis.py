"""Reusable feature investigations for Chapter 6's incident classifier."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from .incident_classifier import (
    INCIDENT_CLASSES,
    INCIDENT_FEATURES,
    INCIDENT_RANDOM_STATE,
    IncidentObservation,
    build_incident_classifier,
    build_incident_targets,
    split_incident_dataset,
)

SYNTHETIC_NOISE = "synthetic_noise"
ANALYSIS_FEATURES = (*INCIDENT_FEATURES, SYNTHETIC_NOISE)
SYNTHETIC_NOISE_SEED = 42


@dataclass(frozen=True)
class FeatureSummary:
    """Descriptive values for one feature within one known class."""

    feature: str
    incident_type: str
    mean: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ModelCoefficient:
    """A standardized logistic coefficient mapped to its class and feature."""

    incident_type: str
    feature: str
    coefficient: float


@dataclass(frozen=True)
class AblationResult:
    """Held-out result for one candidate feature set."""

    name: str
    features: tuple[str, ...]
    accuracy: float
    test_observations: int


def add_synthetic_noise(
    X: NDArray[np.float64], *, seed: int = SYNTHETIC_NOISE_SEED
) -> NDArray[np.float64]:
    """Append reproducible random noise unrelated by design to incident labels."""
    if X.ndim != 2 or X.shape[0] == 0 or not np.isfinite(X).all():
        raise ValueError("features must be a non-empty finite two-dimensional matrix")
    rng = Random(seed)
    noise = np.asarray([rng.random() for _ in range(X.shape[0])], dtype=np.float64)
    return np.column_stack((X, noise))


def _validate_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(feature_names)
    invalid = [name for name in names if name not in ANALYSIS_FEATURES]
    if invalid:
        raise ValueError("unknown feature name(s): " + ", ".join(invalid))
    if not names:
        raise ValueError("at least one feature is required")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    return names


def summarize_features_by_class(
    observations: Sequence[IncidentObservation],
    feature_names: Sequence[str] = INCIDENT_FEATURES,
) -> tuple[FeatureSummary, ...]:
    """Calculate class-grouped mean, minimum, and maximum from observations."""
    names = _validate_feature_names(feature_names)
    if SYNTHETIC_NOISE in names:
        raise ValueError("synthetic_noise has no value on IncidentObservation")
    if not observations:
        raise ValueError("observations must not be empty")
    labels = tuple(
        label
        for label in INCIDENT_CLASSES
        if any(row.incident_type == label for row in observations)
    )
    summaries: list[FeatureSummary] = []
    for feature in names:
        for label in labels:
            values = np.asarray(
                [
                    getattr(row, feature)
                    for row in observations
                    if row.incident_type == label
                ],
                dtype=np.float64,
            )
            summaries.append(
                FeatureSummary(
                    feature,
                    label,
                    float(values.mean()),
                    float(values.min()),
                    float(values.max()),
                )
            )
    return tuple(summaries)


def calculate_correlations(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a Pearson correlation matrix with one row/column per input column."""
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("correlations require at least two finite rows and one column")
    if np.any(np.ptp(X, axis=0) == 0):
        raise ValueError("correlations are undefined for a constant feature")
    return np.atleast_2d(np.asarray(np.corrcoef(X, rowvar=False), dtype=np.float64))


def extract_model_coefficients(
    model: Pipeline, feature_names: Sequence[str]
) -> tuple[ModelCoefficient, ...]:
    """Map a fitted pipeline's standardized coefficients to classes and inputs."""
    names = _validate_feature_names(feature_names)
    if "scaler" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("model must contain scaler and classifier steps")
    classifier = model.named_steps["classifier"]
    coefficients = np.asarray(classifier.coef_)
    classes = tuple(str(label) for label in classifier.classes_)
    if coefficients.shape != (len(classes), len(names)):
        raise ValueError("feature names do not match fitted coefficient columns")
    return tuple(
        ModelCoefficient(
            label, feature, float(coefficients[class_index, feature_index])
        )
        for class_index, label in enumerate(classes)
        for feature_index, feature in enumerate(names)
    )


def compare_feature_sets(
    X: NDArray[np.float64],
    y: NDArray[np.str_],
    all_feature_names: Sequence[str],
    feature_sets: Sequence[tuple[str, Sequence[str]]],
    *,
    random_state: int = INCIDENT_RANDOM_STATE,
) -> tuple[AblationResult, ...]:
    """Train candidates on identical row splits and report held-out accuracy."""
    all_names = _validate_feature_names(all_feature_names)
    if X.ndim != 2 or X.shape[1] != len(all_names):
        raise ValueError("matrix columns must match all_feature_names")
    split = split_incident_dataset(X, y, random_state=random_state)
    results: list[AblationResult] = []
    for experiment_name, supplied_names in feature_sets:
        names = _validate_feature_names(supplied_names)
        missing = [name for name in names if name not in all_names]
        if missing:
            raise ValueError("feature not present in matrix: " + ", ".join(missing))
        indices = [all_names.index(name) for name in names]
        model = build_incident_classifier()
        model.fit(split.X_train[:, indices], split.y_train)
        accuracy = float(
            np.mean(model.predict(split.X_test[:, indices]) == split.y_test)
        )
        results.append(
            AblationResult(experiment_name, names, accuracy, len(split.y_test))
        )
    return tuple(results)


def run_ablation(
    observations: Sequence[IncidentObservation],
) -> tuple[AblationResult, ...]:
    """Run Chapter 6's named experiments, including deterministic synthetic noise."""
    base = np.asarray(
        [[getattr(row, name) for name in INCIDENT_FEATURES] for row in observations],
        dtype=np.float64,
    )
    X = add_synthetic_noise(base)
    y = build_incident_targets(observations)
    experiments: list[tuple[str, Sequence[str]]] = [("all_features", ANALYSIS_FEATURES)]
    for feature in (
        "vendor_latency_ms",
        "db_connections",
        "requests_per_minute",
        SYNTHETIC_NOISE,
    ):
        experiments.append(
            (
                f"without_{feature}",
                tuple(name for name in ANALYSIS_FEATURES if name != feature),
            )
        )
    return compare_feature_sets(X, y, ANALYSIS_FEATURES, experiments)
