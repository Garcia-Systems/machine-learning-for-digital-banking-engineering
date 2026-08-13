from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from harbor_ml.transaction_anomaly import (
    AMOUNT_BANDS, TRANSACTION_CHANNELS, TRANSACTION_TYPES,
    TransactionAnomalyResult, build_transaction_anomaly_pipeline,
    build_transaction_features, build_transaction_preprocessor,
    load_transaction_observations, load_transaction_scenarios, score_transaction,
    train_transaction_anomaly_detector,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline():
    return load_transaction_observations(ROOT / "data/harbor_transaction_observations.csv")


@pytest.fixture(scope="module")
def scenarios():
    return load_transaction_scenarios(ROOT / "data/harbor_transaction_scenarios.csv")


@pytest.fixture(scope="module")
def model(baseline):
    return train_transaction_anomaly_detector(build_transaction_anomaly_pipeline(), baseline)


def test_fixture_loading_and_vocabulary(baseline):
    assert len(baseline) == 800
    assert {row.transaction_type for row in baseline} == set(TRANSACTION_TYPES)
    assert {row.channel for row in baseline} == set(TRANSACTION_CHANNELS)
    assert {row.amount_band for row in baseline} == set(AMOUNT_BANDS)


def test_schema_has_no_direct_identity_or_target():
    names = {field.name for field in fields(type(load_transaction_observations(
        ROOT / "data/harbor_transaction_observations.csv")[0]))}
    prohibited = {"account_number", "member_name", "card_number", "ssn", "exact_location",
                  "account_balance", "merchant_name", "device_id", "ip_address", "target",
                  "fraud", "probability"}
    assert names.isdisjoint(prohibited)


def test_preprocessor_encodes_scales_and_is_dense(baseline):
    preprocessor = build_transaction_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(preprocessor.transformers[0][1], OneHotEncoder)
    assert preprocessor.transformers[0][1].handle_unknown == "ignore"
    assert preprocessor.transformers[0][1].sparse_output is False
    assert isinstance(preprocessor.transformers[1][1], StandardScaler)
    transformed = preprocessor.fit_transform(build_transaction_features(baseline))
    assert isinstance(transformed, np.ndarray)
    numeric = transformed[:, -5:-1].astype(float)
    np.testing.assert_allclose(numeric.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(numeric.std(axis=0), 1, atol=1e-12)


def test_pipeline_fits_and_scores_finitely(model, scenarios):
    assert isinstance(model.named_steps["detector"], IsolationForest)
    results = [score_transaction(model, scenario.observation) for scenario in scenarios]
    assert all(isinstance(result, TransactionAnomalyResult) for result in results)
    assert all(np.isfinite(result.raw_score) for result in results)
    assert "probability" not in {field.name for field in fields(TransactionAnomalyResult)}


def test_teaching_scenarios_have_reproducible_expected_relationship(model, scenarios):
    results = {item.name: score_transaction(model, item.observation) for item in scenarios}
    assert not results["routine_mobile_purchase"].is_anomaly
    assert not results["large_normal_transfer"].is_anomaly
    assert not results["new_device_routine_behavior"].is_anomaly
    assert results["unusual_combination"].is_anomaly
    assert results["unusual_combination"].raw_score > results["routine_mobile_purchase"].raw_score


def test_unknown_category_is_transformed_without_claiming_understanding(model, scenarios):
    unknown = replace(scenarios[0].observation, transaction_type="experimental_payment")
    assert np.isfinite(score_transaction(model, unknown).raw_score)


def test_fixed_seed_is_deterministic(baseline, scenarios):
    first = train_transaction_anomaly_detector(build_transaction_anomaly_pipeline(), baseline)
    second = train_transaction_anomaly_detector(build_transaction_anomaly_pipeline(), baseline)
    assert score_transaction(first, scenarios[2].observation) == score_transaction(
        second, scenarios[2].observation,
    )


def test_malformed_rows_are_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "timestamp,transaction_type,channel,amount_band,hour_of_day,recent_transaction_count,"
        "minutes_since_previous_transaction,device_change,distance_from_recent_pattern\n"
        "not-a-date,debit_purchase,mobile,25_to_99,99,2,10,maybe,2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed"):
        load_transaction_observations(bad)
