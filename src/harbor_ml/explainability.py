"""Transparent explanations for Harbor's fitted integration-failure model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import exp, isfinite
from typing import Iterable, Mapping

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .integration_failure_model import (
    NUMERIC_FEATURES,
    PREDICTION_FEATURES,
    IntegrationRequest,
    _request_row,
    predict_failure_probability,
)


@dataclass(frozen=True)
class FeatureContribution:
    """One transformed value's exact contribution to a binary linear score."""

    feature_name: str
    transformed_feature_name: str
    transformed_value: float
    coefficient: float
    contribution: float


@dataclass(frozen=True)
class LocalExplanation:
    """A model-versioned decomposition, not a real-world causal account."""

    model_name: str
    model_version: str
    intercept: float
    contributions: tuple[FeatureContribution, ...]
    linear_score: float
    probability: float


@dataclass(frozen=True)
class PermutationImportance:
    feature_name: str
    importance_mean: float
    importance_std: float


@dataclass(frozen=True)
class SensitivityResult:
    feature_name: str
    value: float
    probability: float
    request: IntegrationRequest


def get_transformed_feature_names(model: Pipeline) -> tuple[str, ...]:
    """Ask the fitted preprocessor for names in its actual output order."""
    names = model.named_steps["preprocessor"].get_feature_names_out(PREDICTION_FEATURES)
    return tuple(str(name) for name in names)


def readable_feature_name(transformed_name: str) -> str:
    """Display a transformed name while retaining the exact name separately."""
    _, separator, name = transformed_name.partition("__")
    name = name if separator else transformed_name
    for categorical in ("vendor", "endpoint"):
        prefix = f"{categorical}_"
        if name.startswith(prefix):
            return f"{categorical}={name[len(prefix):]}"
    return name


def extract_logistic_coefficients(model: Pipeline) -> tuple[FeatureContribution, ...]:
    """Return global coefficients paired with preprocessor-derived names."""
    classifier = model.named_steps["classifier"]
    names = get_transformed_feature_names(model)
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    if len(names) != len(coefficients):
        raise ValueError("transformed feature and coefficient counts differ")
    return tuple(
        FeatureContribution(readable_feature_name(name), name, 0.0, float(coefficient), 0.0)
        for name, coefficient in zip(names, coefficients)
    )


def explain_linear_prediction(
    model: Pipeline,
    request: IntegrationRequest | Mapping[str, object],
    *,
    model_name: str,
    model_version: str,
) -> LocalExplanation:
    """Exactly decompose one binary logistic-regression prediction."""
    row = _request_row(request)
    transformed = model.named_steps["preprocessor"].transform(row)
    values = np.asarray(transformed.toarray() if hasattr(transformed, "toarray") else transformed)[0]
    templates = extract_logistic_coefficients(model)
    contributions = tuple(
        replace(item, transformed_value=float(value), contribution=float(value * item.coefficient))
        for item, value in zip(templates, values)
    )
    intercept = float(model.named_steps["classifier"].intercept_[0])
    linear_score = intercept + sum(item.contribution for item in contributions)
    probability = 1.0 / (1.0 + exp(-linear_score))
    return LocalExplanation(
        model_name, model_version, intercept, contributions, linear_score, probability
    )


def sorted_contributions(
    explanation: LocalExplanation, *, positive: bool, limit: int | None = None
) -> tuple[FeatureContribution, ...]:
    selected = (
        [item for item in explanation.contributions if item.contribution > 0]
        if positive
        else [item for item in explanation.contributions if item.contribution < 0]
    )
    selected.sort(key=lambda item: item.contribution, reverse=positive)
    return tuple(selected[:limit])


def calculate_permutation_importance(
    model: Pipeline,
    X_test,
    y_test,
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[PermutationImportance, ...]:
    """Measure held-out reliance by shuffling original, pre-one-hot columns."""
    result = permutation_importance(
        model, X_test, y_test, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state,
    )
    items = tuple(
        PermutationImportance(name, float(mean), float(std))
        for name, mean, std in zip(PREDICTION_FEATURES, result.importances_mean, result.importances_std)
    )
    if not all(isfinite(item.importance_mean) and isfinite(item.importance_std) for item in items):
        raise ValueError("permutation importance produced a non-finite value")
    return tuple(sorted(items, key=lambda item: item.importance_mean, reverse=True))


def compare_feature_sensitivity(
    model: Pipeline,
    base_request: IntegrationRequest,
    feature_name: str,
    values: Iterable[float],
) -> tuple[SensitivityResult, ...]:
    """Vary exactly one numeric input in a model sensitivity experiment."""
    if feature_name not in NUMERIC_FEATURES:
        raise ValueError("model sensitivity requires a declared numerical feature")
    original = asdict(base_request)
    results = []
    for value in values:
        candidate = replace(base_request, **{feature_name: value})
        changed = {name for name in PREDICTION_FEATURES if asdict(candidate)[name] != original[name]}
        if changed - {feature_name}:
            raise AssertionError("model sensitivity changed an unintended feature")
        results.append(SensitivityResult(feature_name, float(value), predict_failure_probability(model, candidate), candidate))
    return tuple(results)
