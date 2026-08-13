from pathlib import Path

import numpy as np
import pytest

from harbor_ml import (
    ANALYSIS_FEATURES,
    INCIDENT_CLASSES,
    INCIDENT_FEATURES,
    add_synthetic_noise,
    build_incident_classifier,
    build_incident_features,
    build_incident_targets,
    calculate_correlations,
    compare_feature_sets,
    extract_model_coefficients,
    load_incident_dataset,
    run_ablation,
    split_incident_dataset,
    summarize_features_by_class,
)

DATASET = Path(__file__).parents[1] / "data/harbor_incident_classes.csv"


@pytest.fixture
def observations():
    return load_incident_dataset(DATASET)


def test_summaries_group_every_feature_and_class(observations) -> None:
    summaries = summarize_features_by_class(observations)
    assert len(summaries) == len(INCIDENT_FEATURES) * len(INCIDENT_CLASSES)
    assert {(item.feature, item.incident_type) for item in summaries} == {
        (feature, label) for feature in INCIDENT_FEATURES for label in INCIDENT_CLASSES
    }
    vendor_normal = next(
        item
        for item in summaries
        if item.feature == "vendor_latency_ms" and item.incident_type == "normal"
    )
    expected = [
        row.vendor_latency_ms for row in observations if row.incident_type == "normal"
    ]
    assert vendor_normal.mean == pytest.approx(np.mean(expected))
    assert vendor_normal.minimum == min(expected)
    assert vendor_normal.maximum == max(expected)


def test_correlation_matrix_is_square_symmetric_with_unit_diagonal(
    observations,
) -> None:
    matrix = calculate_correlations(build_incident_features(observations))
    assert matrix.shape == (len(INCIDENT_FEATURES), len(INCIDENT_FEATURES))
    assert np.allclose(matrix, matrix.T)
    assert np.diag(matrix) == pytest.approx(np.ones(len(INCIDENT_FEATURES)))


def test_synthetic_noise_is_reproducible_and_seed_sensitive(observations) -> None:
    X = build_incident_features(observations)
    assert np.array_equal(add_synthetic_noise(X), add_synthetic_noise(X))
    assert not np.array_equal(
        add_synthetic_noise(X)[:, -1], add_synthetic_noise(X, seed=7)[:, -1]
    )
    assert np.array_equal(add_synthetic_noise(X)[:, :-1], X)


def test_coefficients_map_every_fitted_class_and_feature(observations) -> None:
    X = add_synthetic_noise(build_incident_features(observations))
    y = build_incident_targets(observations)
    split = split_incident_dataset(X, y)
    fitted = build_incident_classifier().fit(split.X_train, split.y_train)
    coefficients = extract_model_coefficients(fitted, ANALYSIS_FEATURES)
    assert {(item.incident_type, item.feature) for item in coefficients} == {
        (label, feature) for label in fitted.classes_ for feature in ANALYSIS_FEATURES
    }
    classifier = fitted.named_steps["classifier"]
    first = coefficients[0]
    assert first.incident_type == classifier.classes_[0]
    assert first.feature == ANALYSIS_FEATURES[0]
    assert first.coefficient == classifier.coef_[0, 0]


def test_ablation_is_deterministic_and_really_removes_features(observations) -> None:
    first, second = run_ablation(observations), run_ablation(observations)
    assert first == second
    assert all(0 <= result.accuracy <= 1 for result in first)
    assert all(result.test_observations == 75 for result in first)
    removed = next(
        result for result in first if result.name == "without_vendor_latency_ms"
    )
    assert "vendor_latency_ms" not in removed.features
    assert len(removed.features) == len(ANALYSIS_FEATURES) - 1


def test_all_comparisons_use_same_deterministic_split(observations) -> None:
    X = add_synthetic_noise(build_incident_features(observations))
    y = build_incident_targets(observations)
    experiments = (("first", ANALYSIS_FEATURES), ("second", ANALYSIS_FEATURES))
    results = compare_feature_sets(X, y, ANALYSIS_FEATURES, experiments)
    assert results[0].accuracy == results[1].accuracy
    assert results[0].test_observations == results[1].test_observations


def test_invalid_feature_names_are_rejected_clearly(observations) -> None:
    with pytest.raises(ValueError, match="unknown feature.*made_up_metric"):
        summarize_features_by_class(observations, ("made_up_metric",))
    X = add_synthetic_noise(build_incident_features(observations))
    with pytest.raises(ValueError, match="unknown feature.*made_up_metric"):
        compare_feature_sets(
            X,
            build_incident_targets(observations),
            ANALYSIS_FEATURES,
            (("bad", ("made_up_metric",)),),
        )
