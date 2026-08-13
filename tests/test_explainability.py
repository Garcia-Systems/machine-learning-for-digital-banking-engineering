from dataclasses import asdict
from math import exp
from pathlib import Path

import numpy as np
import pytest

from harbor_ml.explainability import (
    calculate_permutation_importance, compare_feature_sensitivity,
    explain_linear_prediction, extract_logistic_coefficients,
    get_transformed_feature_names, sorted_contributions,
)
from harbor_ml.integration_failure_model import (
    PREDICTION_FEATURES, IntegrationRequest, build_integration_features,
    build_integration_targets, load_integration_requests, split_integration_dataset,
)
from harbor_ml.training import TrainingConfig, train_integration_failure_model

DATASET = Path(__file__).parents[1] / "data/harbor_integration_requests.csv"


@pytest.fixture(scope="module")
def fitted():
    observations = load_integration_requests(DATASET)
    split = split_integration_dataset(
        build_integration_features(observations), build_integration_targets(observations),
        test_size=.25, random_state=42,
    )
    return train_integration_failure_model(DATASET, TrainingConfig()), split


@pytest.fixture
def sample_request():
    return IntegrationRequest("ClearVerify", "identity_verify", 1500.0, .28, 90, 2, 4000, 14)


def test_names_coefficients_and_one_hot_mapping(fitted, sample_request):
    trained, _ = fitted
    row = np.asarray([[getattr(sample_request, name) for name in PREDICTION_FEATURES]], dtype=object)
    transformed = trained.pipeline.named_steps["preprocessor"].transform(row)
    names = get_transformed_feature_names(trained.pipeline)
    coefficients = extract_logistic_coefficients(trained.pipeline)
    assert len(names) == transformed.shape[1] == len(coefficients)
    active = {item.feature_name: item.transformed_value for item in explain_linear_prediction(
        trained.pipeline, sample_request, model_name=trained.metadata.model_name,
        model_version=trained.metadata.model_version).contributions}
    assert active["vendor=ClearVerify"] == 1


def test_local_explanation_exactly_reproduces_probability(fitted, sample_request):
    trained, _ = fitted
    explanation = explain_linear_prediction(
        trained.pipeline, sample_request, model_name=trained.metadata.model_name,
        model_version=trained.metadata.model_version,
    )
    assert explanation.linear_score == pytest.approx(
        explanation.intercept + sum(item.contribution for item in explanation.contributions))
    assert explanation.probability == pytest.approx(1 / (1 + exp(-explanation.linear_score)))
    assert explanation.probability == pytest.approx(trained.pipeline.predict_proba(
        [[getattr(sample_request, name) for name in PREDICTION_FEATURES]])[0, 1])
    assert explanation.model_version == trained.metadata.model_version
    assert [x.contribution for x in sorted_contributions(explanation, positive=True)] == sorted(
        (x.contribution for x in explanation.contributions if x.contribution > 0), reverse=True)
    assert [x.contribution for x in sorted_contributions(explanation, positive=False)] == sorted(
        x.contribution for x in explanation.contributions if x.contribution < 0)
    prohibited = {"member_email", "account_number", "password", "access_token"}
    assert prohibited.isdisjoint(asdict(explanation).keys())


def test_permutation_importance_uses_original_features_and_is_finite(fitted):
    trained, split = fitted
    items = calculate_permutation_importance(
        trained.pipeline, split.X_test, split.y_test, n_repeats=3, random_state=42)
    assert {item.feature_name for item in items} == set(PREDICTION_FEATURES)
    assert all(np.isfinite(item.importance_mean) and np.isfinite(item.importance_std)
               for item in items)


def test_sensitivity_changes_only_requested_feature(fitted, sample_request):
    trained, _ = fitted
    results = compare_feature_sensitivity(
        trained.pipeline, sample_request, "recent_vendor_error_rate", (.01, .1, .3))
    base = asdict(sample_request)
    for result in results:
        candidate = asdict(result.request)
        assert {key for key in base if base[key] != candidate[key]} == {
            "recent_vendor_error_rate"
        }
    assert [item.probability for item in results] == sorted(item.probability for item in results)
