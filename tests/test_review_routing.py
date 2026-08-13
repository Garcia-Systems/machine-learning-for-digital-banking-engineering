from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from harbor_ml.review_routing import (
    LEAKAGE_FIELDS, REVIEW_FEATURES, REVIEW_TARGET, ReviewObservation, ReviewPrediction,
    apply_review_threshold, build_review_features, build_review_pipeline,
    build_review_targets, evaluate_review_model, load_review_dataset,
    predict_review, predict_review_probability, predict_review_probabilities,
    split_review_dataset, train_review_model,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rows(): return load_review_dataset(ROOT / "data/harbor_review_routing.csv")


@pytest.fixture(scope="module")
def fitted(rows):
    X, y = build_review_features(rows), build_review_targets(rows)
    split = split_review_dataset(X, y)
    return train_review_model(build_review_pipeline(), split.X_train, split.y_train), split


def test_fixture_loading_binary_target_and_distribution(rows):
    targets = build_review_targets(rows)
    assert len(rows) == 1000 and set(targets) == {0, 1}
    assert np.bincount(targets).tolist() == [730, 270]


def test_feature_contract_excludes_leakage_and_identity():
    assert set(REVIEW_FEATURES).isdisjoint(LEAKAGE_FIELDS)
    assert REVIEW_TARGET not in REVIEW_FEATURES
    prohibited = {"member_name", "account_number", "card_number", "ssn", "device_id", "ip_address"}
    assert set(REVIEW_FEATURES).isdisjoint(prohibited)


def test_preprocessing_pipeline_construction():
    model = build_review_pipeline(); preprocessor = model.named_steps["preprocessor"]
    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(preprocessor.transformers[0][1], StandardScaler)
    assert isinstance(preprocessor.transformers[2][1], OneHotEncoder)
    assert preprocessor.transformers[2][1].handle_unknown == "ignore"
    assert isinstance(model.named_steps["classifier"], LogisticRegression)


def test_model_fitting_probability_range_and_scenario(fitted):
    model, split = fitted
    probabilities = predict_review_probabilities(model, split.X_test)
    assert probabilities.shape == (250,) and np.all((0 <= probabilities) & (probabilities <= 1))
    scenario = ReviewObservation("external_transfer", "web", "500_to_999", 2, 7, 2, True, .82, 3)
    prediction = predict_review(model, scenario)
    assert isinstance(prediction, ReviewPrediction) and prediction.probability > .5


def test_threshold_logic():
    assert apply_review_threshold(.5, threshold=.5)
    assert not apply_review_threshold(.49, threshold=.5)
    with pytest.raises(ValueError): apply_review_threshold(1.1)


def test_confusion_precision_and_recall_calculation():
    result = evaluate_review_model(np.array([0, 0, 1, 1, 1]), np.array([.1, .8, .9, .7, .2]))
    assert result.confusion_matrix.shape == (2, 2)
    assert (result.true_negatives, result.false_positives, result.false_negatives, result.true_positives) == (1, 1, 1, 2)
    assert result.precision == pytest.approx(2 / 3) and result.recall == pytest.approx(2 / 3)


def test_split_is_deterministic(rows):
    X, y = build_review_features(rows), build_review_targets(rows)
    first, second = split_review_dataset(X, y), split_review_dataset(X, y)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    np.testing.assert_array_equal(first.y_test, second.y_test)


def test_unseen_category_handling(fitted, rows):
    model, _ = fitted
    unknown = replace(rows[0].observation, transaction_type="future_transaction_type")
    assert 0 <= predict_review_probability(model, unknown) <= 1


def test_malformed_row_validation(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,transaction_type,channel,amount_band,hour_of_day,recent_transaction_count,minutes_since_previous_transaction,device_change,distance_from_recent_pattern,recent_failed_transaction_count,manual_review_required\nnot-time,debit_purchase,mobile,under_25,25,-1,no,maybe,2,-1,7\n")
    with pytest.raises(ValueError, match="malformed"): load_review_dataset(bad)


def test_prediction_language_does_not_label_fraud_probability():
    names = {field.name for field in fields(ReviewPrediction)}
    assert names == {"probability", "predicted_review"}
    assert "wrongdoing" in (ReviewPrediction.__doc__ or "")
