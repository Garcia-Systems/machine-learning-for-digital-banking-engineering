from dataclasses import fields
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from harbor_ml.demand_forecasting import (
    DEMAND_FEATURES, DEMAND_TARGET, FORECAST_HORIZON_MINUTES, DemandExample,
    build_demand_examples, build_demand_features, build_demand_model,
    build_demand_targets, calculate_metrics, chronological_split,
    load_demand_observations, persistence_predictions, predict_future_demand,
    residuals, train_demand_model,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def observations(): return load_demand_observations(ROOT / "data/harbor_system_demand.csv")


@pytest.fixture(scope="module")
def examples(observations): return build_demand_examples(observations)


def test_fixture_loading_and_chronological_order(observations):
    assert len(observations) == 5760
    assert all(a.timestamp < b.timestamp for a, b in zip(observations, observations[1:]))
    assert all(item.requests_per_minute >= 0 for item in observations)


def test_lags_target_horizon_and_strong_leakage_contract(observations, examples):
    item = examples[100]
    now_index = observations.index(next(row for row in observations if row.timestamp == item.timestamp))
    assert item.requests_now == observations[now_index].requests_per_minute
    assert item.requests_5m_ago == observations[now_index - 1].requests_per_minute
    assert item.requests_10m_ago == observations[now_index - 2].requests_per_minute
    assert item.requests_15m_ago == observations[now_index - 3].requests_per_minute
    assert item.future_requests_per_minute == observations[now_index + 2].requests_per_minute
    assert observations[now_index + 2].timestamp - item.timestamp == timedelta(minutes=FORECAST_HORIZON_MINUTES)
    assert DEMAND_TARGET not in DEMAND_FEATURES
    assert DEMAND_TARGET not in {field.name for field in fields(DemandExample) if field.name in DEMAND_FEATURES}
    vector = build_demand_features([item])[0]
    assert len(vector) == len(DEMAND_FEATURES)
    np.testing.assert_array_equal(vector, [getattr(item, name) for name in DEMAND_FEATURES])


def test_rolling_features_use_only_present_and_past(observations, examples):
    item = examples[20]
    index = next(i for i, row in enumerate(observations) if row.timestamp == item.timestamp)
    past = observations[index - 3:index + 1]
    assert item.recent_average_requests == pytest.approx(np.mean([x.requests_per_minute for x in past]))
    assert item.recent_growth == pytest.approx(past[-1].requests_per_minute - past[0].requests_per_minute)


def test_chronological_split_has_no_overlap_inversion(examples):
    split = chronological_split(examples)
    assert len(split.train) == int(.8 * len(examples))
    assert split.train[-1].timestamp < split.test[0].timestamp
    assert max(x.timestamp for x in split.train) < min(x.timestamp for x in split.test)


def test_pipeline_fit_numeric_finite_deterministic_predictions(examples):
    split = chronological_split(examples)
    first = train_demand_model(build_demand_model(), split.train)
    second = train_demand_model(build_demand_model(), split.train)
    assert isinstance(first.named_steps["scaler"], StandardScaler)
    assert isinstance(first.named_steps["regressor"], Ridge)
    prediction = predict_future_demand(first, split.test)
    assert prediction.shape == (len(split.test),) and np.all(np.isfinite(prediction))
    np.testing.assert_array_equal(prediction, predict_future_demand(second, split.test))


def test_mae_rmse_baseline_and_residual_calculation(examples):
    metrics = calculate_metrics([800, 900], [750, 850])
    assert metrics.mae == 50 and metrics.rmse == 50
    assert residuals([900, 800], [820, 860]).tolist() == [80, -60]
    assert persistence_predictions(examples[:2]).tolist() == [x.requests_now for x in examples[:2]]


def test_baseline_and_model_metrics_use_same_test_targets(examples):
    split = chronological_split(examples)
    model = train_demand_model(build_demand_model(), split.train)
    actual = build_demand_targets(split.test)
    baseline_metrics = calculate_metrics(actual, persistence_predictions(split.test))
    model_metrics = calculate_metrics(actual, predict_future_demand(model, split.test))
    assert np.isfinite([baseline_metrics.mae, baseline_metrics.rmse,
                        model_metrics.mae, model_metrics.rmse]).all()


def test_malformed_row_and_out_of_order_handling(tmp_path):
    header = "timestamp,requests_per_minute,api_latency_ms,error_rate,queue_depth\n"
    bad = tmp_path / "bad.csv"
    bad.write_text(header + "not-time,nope,100,.01,-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"): load_demand_observations(bad)
    bad.write_text(header + "2026-01-01T00:05:00Z,100,100,.01,1\n"
                   "2026-01-01T00:00:00Z,100,100,.01,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chronological"): load_demand_observations(bad)
