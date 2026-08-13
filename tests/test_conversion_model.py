from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from harbor_ml import (
    CATEGORICAL_CONVERSION_FEATURES, CONVERSION_FEATURES, CONVERSION_TARGET,
    NUMERIC_CONVERSION_FEATURES, MemberEvent, PartialConversionState,
    build_conversion_features, build_conversion_pipeline, build_conversion_snapshots,
    build_conversion_targets, calculate_conversion_rate, calculate_majority_baseline,
    classify_conversion, evaluate_conversion_model, group_events_by_session,
    load_member_events, predict_conversion, split_conversion_dataset,
    train_conversion_model,
)

DATASET = Path(__file__).parents[1] / "data/harbor_member_events.csv"


@pytest.fixture(scope="module")
def snapshots():
    return build_conversion_snapshots(group_events_by_session(load_member_events(DATASET)))


def event(second, session, name, source="direct", device="desktop"):
    return MemberEvent(
        datetime(2026, 8, 13, 10, tzinfo=timezone.utc) + timedelta(seconds=second),
        session, name, "web", "horizon_savings", source, device,
    )


def test_fixture_detects_product_detail_sessions_and_both_outcomes(snapshots):
    assert len(snapshots) == 500
    targets = build_conversion_targets(snapshots)
    assert 0.15 < targets.mean() < 0.50
    assert all(row.snapshot_timestamp.tzinfo is not None for row in snapshots)


def test_prediction_point_snapshot_and_future_events_are_excluded():
    events = [event(0, "session-9000", "session_started"),
              event(10, "session-9000", "login_failed"),
              event(20, "session-9000", "landing_page_viewed"),
              event(30, "session-9000", "search_performed"),
              event(40, "session-9000", "help_opened"),
              event(50, "session-9000", "product_details_viewed"),
              event(60, "session-9000", "search_performed"),
              event(70, "session-9000", "help_opened"),
              event(80, "session-9000", "application_started")]
    row = build_conversion_snapshots({"session-9000": events})[0]
    assert row.snapshot_timestamp == events[5].timestamp
    assert (row.elapsed_seconds, row.events_so_far) == (50, 6)
    assert (row.search_count, row.help_events, row.prior_login_failures) == (1, 1, 1)
    assert row.application_started is True


def test_identical_prefixes_make_identical_features_but_different_targets():
    def prefix(session):
        return [event(0, session, "session_started"), event(10, session, "landing_page_viewed"),
                event(30, session, "product_details_viewed")]
    sessions = {
        "session-9001": prefix("session-9001") + [event(40, "session-9001", "application_started")],
        "session-9002": prefix("session-9002") + [event(50, "session-9002", "session_ended")],
    }
    rows = build_conversion_snapshots(sessions)
    assert np.array_equal(build_conversion_features(rows)[0], build_conversion_features(rows)[1])
    assert build_conversion_targets(rows).tolist() == [1, 0]


def test_missing_prediction_point_and_malformed_sessions():
    assert build_conversion_snapshots({"session-9003": [event(0, "session-9003", "session_started")]}) == []
    with pytest.raises(ValueError, match="begin"):
        build_conversion_snapshots({"session-9003": [event(0, "session-9003", "product_details_viewed")]})
    with pytest.raises(ValueError, match="mapping key"):
        build_conversion_snapshots({"session-9003": [event(0, "session-9004", "session_started")]})
    missing_context = [replace(event(0, "session-9005", "session_started"), landing_source=None),
                       replace(event(2, "session-9005", "product_details_viewed"), landing_source=None)]
    with pytest.raises(ValueError, match="landing source"):
        build_conversion_snapshots({"session-9005": missing_context})


def test_privacy_minimized_feature_schema_and_no_leakage(snapshots):
    assert NUMERIC_CONVERSION_FEATURES == (
        "elapsed_seconds", "events_so_far", "search_count", "help_events",
        "prior_login_failures", "hour_of_day")
    assert CATEGORICAL_CONVERSION_FEATURES == ("channel", "landing_source", "device_category")
    forbidden = {CONVERSION_TARGET, "application_start_timestamp", "final_event_name",
                 "session_duration_seconds", "application_completed", "member_name",
                 "account_number", "balance", "income", "credit_score", "age", "race"}
    assert not forbidden & set(CONVERSION_FEATURES)
    assert build_conversion_features(snapshots[:2]).shape == (2, 9)


def test_baselines_split_pipeline_fit_probability_threshold_and_evaluation(snapshots):
    X, y = build_conversion_features(snapshots), build_conversion_targets(snapshots)
    assert calculate_conversion_rate(snapshots) == pytest.approx(y.mean())
    assert calculate_majority_baseline(y) == pytest.approx(max(y.mean(), 1 - y.mean()))
    first, second = split_conversion_dataset(X, y), split_conversion_dataset(X, y)
    assert all(np.array_equal(getattr(first, name), getattr(second, name))
               for name in ("X_train", "X_test", "y_train", "y_test"))
    model = train_conversion_model(build_conversion_pipeline(), first.X_train, first.y_train)
    transformed = model.named_steps["preprocessor"].transform(first.X_test)
    assert transformed.shape[0] == len(first.y_test)
    result = evaluate_conversion_model(model, first.X_test, first.y_test)
    assert result.confusion_matrix.shape == (2, 2)
    state = PartialConversionState("web", "direct", "desktop", 60, 3, 0, 0, 0, 10)
    prediction = predict_conversion(model, state)
    assert 0 <= prediction.probability <= 1
    assert prediction.predicted_conversion == classify_conversion(prediction.probability)
    assert classify_conversion(prediction.probability, threshold=.3) >= classify_conversion(prediction.probability, threshold=.7)
    with pytest.raises(ValueError):
        classify_conversion(prediction.probability, threshold=1.1)
    with pytest.raises(ValueError):
        calculate_conversion_rate([])
