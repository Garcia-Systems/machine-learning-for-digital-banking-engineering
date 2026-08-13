from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from harbor_ml.database_performance import (
    QUERY_FEATURES, QUERY_FAMILIES, QUERY_TARGET, ROW_BANDS, QueryContext,
    build_database_performance_pipeline, build_query_features, calculate_metrics,
    chronological_split, load_query_performance, predict_query_duration,
    query_family_baseline, residuals, train_database_performance_model,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def observations(): return load_query_performance(ROOT / "data/harbor_query_performance.csv")


def test_fixture_values_order_and_leakage_contract(observations):
    assert len(observations) == 1800
    assert {x.context.query_family for x in observations} == set(QUERY_FAMILIES)
    assert {x.context.rows_expected_band for x in observations} == set(ROW_BANDS)
    assert all(a.timestamp < b.timestamp for a, b in zip(observations, observations[1:]))
    assert QUERY_TARGET not in QUERY_FEATURES
    assert QUERY_TARGET not in {x.name for x in fields(QueryContext)}


def test_chronological_split(observations):
    split = chronological_split(observations)
    assert len(split.train) == 1440 and len(split.test) == 360
    assert split.train[-1].timestamp < split.test[0].timestamp


def test_pipeline_categorical_fit_predictions_and_determinism(observations):
    split = chronological_split(observations)
    first = train_database_performance_model(build_database_performance_pipeline(), split.train)
    second = train_database_performance_model(build_database_performance_pipeline(), split.train)
    assert isinstance(first.named_steps["preprocessor"], ColumnTransformer)
    assert isinstance(first.named_steps["regressor"], RandomForestRegressor)
    contexts = [x.context for x in split.test[:20]]
    prediction = predict_query_duration(first, contexts)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    np.testing.assert_array_equal(prediction, predict_query_duration(second, contexts))
    # The fitted one-hot encoder safely ignores a category not observed during fitting.
    transformed = first.named_steps["preprocessor"].transform(
        np.asarray([["new_family", "small", 1, 1, False, False, False, 20, 1, 80., 200.]],
                   dtype=object))
    assert transformed.shape[0] == 1


def test_baseline_uses_training_family_medians_and_metrics(observations):
    split = chronological_split(observations)
    contexts = [x.context for x in split.test]
    predicted = query_family_baseline(split.train, contexts)
    expected = np.median([x.query_duration_ms for x in split.train
                          if x.context.query_family == contexts[0].query_family])
    assert predicted[0] == expected
    metrics = calculate_metrics([100, 200, 500], [90, 220, 450])
    assert metrics.mae == pytest.approx(80 / 3)
    assert metrics.rmse == pytest.approx(np.sqrt(1000))
    assert metrics.median_absolute_error == 20
    assert residuals([1800, 800], [1100, 900]).tolist() == [700, -100]


def test_feature_shape_and_context_validation(observations):
    matrix = build_query_features([observations[0].context])
    assert matrix.shape == (1, len(QUERY_FEATURES))
    with pytest.raises(ValueError):
        QueryContext("bad", "small", 1, 1, False, False, False, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        QueryContext("account_summary", "huge", 1, 1, False, False, False, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        QueryContext("account_summary", "small", -1, 1, False, False, False, 1, 1, 1, 1)


def test_malformed_fixture(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,query_family\nnope,bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"): load_query_performance(path)
