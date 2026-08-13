from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from harbor_ml import (
    JOURNEY_FEATURES, JOURNEY_TARGET, MemberEvent, PartialJourneyState,
    build_abandonment_pipeline, build_journey_features, build_journey_targets,
    build_transfer_snapshots, classify_abandonment, evaluate_abandonment_model,
    group_events_by_session, load_member_events, predict_abandonment_probability,
    split_journey_dataset, train_abandonment_model,
)

DATASET = Path(__file__).parents[1] / "data/harbor_member_events.csv"


@pytest.fixture(scope="module")
def snapshots():
    events = load_member_events(DATASET)
    return build_transfer_snapshots(group_events_by_session(events))


def event(second, session, name, channel="web"):
    return MemberEvent(datetime(2026, 8, 13, tzinfo=timezone.utc) + timedelta(seconds=second),
                       session, name, channel, "transfer")


def test_fixture_has_qualifying_balanced_transfer_journeys(snapshots):
    assert len(snapshots) == 330
    targets = build_journey_targets(snapshots)
    assert 0.25 < targets.mean() < 0.75
    assert all(row.snapshot_timestamp.tzinfo is not None for row in snapshots)


def test_prediction_point_and_feature_extraction_exclude_future_events():
    events = [event(0, "session-9000", "login_failed"), event(10, "session-9000", "transfer_started"),
              event(20, "session-9000", "help_opened"), event(30, "session-9000", "transfer_failed"),
              event(40, "session-9000", "search_performed"), event(50, "session-9000", "recipient_selected"),
              event(60, "session-9000", "help_opened"), event(70, "session-9000", "transfer_failed"),
              event(80, "session-9000", "search_performed"), event(90, "session-9000", "transfer_completed")]
    row = build_transfer_snapshots({"session-9000": events})[0]
    assert row.snapshot_timestamp == events[5].timestamp
    assert (row.elapsed_seconds, row.events_so_far) == (40, 5)
    assert (row.help_opened, row.error_count, row.search_count, row.previous_failed_login) == (True, 1, 1, True)
    assert row.journey_abandoned is False


def test_identical_histories_have_identical_features_but_different_labels():
    prefix_a = [event(0, "session-9001", "transfer_started"), event(20, "session-9001", "recipient_selected")]
    prefix_b = [event(0, "session-9002", "transfer_started"), event(20, "session-9002", "recipient_selected")]
    sessions = {"session-9001": prefix_a + [event(30, "session-9001", "transfer_completed")],
                "session-9002": prefix_b + [event(30, "session-9002", "help_opened"), event(40, "session-9002", "session_ended")]}
    rows = build_transfer_snapshots(sessions)
    assert np.array_equal(build_journey_features(rows)[0], build_journey_features(rows)[1])
    assert build_journey_targets(rows).tolist() == [0, 1]


def test_missing_prediction_point_and_malformed_journeys():
    assert build_transfer_snapshots({"session-9003": [event(0, "session-9003", "transfer_started")]}) == []
    assert build_transfer_snapshots({"session-9003": [event(0, "session-9003", "recipient_selected")]}) == []
    with pytest.raises(ValueError, match="mapping key"):
        build_transfer_snapshots({"session-9003": [event(0, "session-9004", "transfer_started")]})


def test_feature_schema_order_and_no_outcomes(snapshots):
    assert JOURNEY_FEATURES == ("elapsed_seconds", "events_so_far", "error_count", "search_count",
                                "help_opened", "previous_failed_login", "channel")
    assert JOURNEY_TARGET not in JOURNEY_FEATURES
    assert not {"transfer_completed", "session_ended", "final_event_name", "journey_duration_seconds"} & set(JOURNEY_FEATURES)
    assert build_journey_features(snapshots[:2]).shape == (2, 7)


def test_split_preprocessing_fit_evaluation_probability_and_threshold(snapshots):
    X, y = build_journey_features(snapshots), build_journey_targets(snapshots)
    first = split_journey_dataset(X, y)
    second = split_journey_dataset(X, y)
    assert all(np.array_equal(getattr(first, name), getattr(second, name)) for name in ("X_train", "X_test", "y_train", "y_test"))
    model = train_abandonment_model(build_abandonment_pipeline(), first.X_train, first.y_train)
    transformed = model.named_steps["preprocessor"].transform(first.X_test)
    assert transformed.shape[1] == 8  # four numeric, two boolean, web/mobile one-hot
    result = evaluate_abandonment_model(model, first.X_test, first.y_test)
    assert result.confusion_matrix.shape == (2, 2)
    state = PartialJourneyState("mobile", 80, 4, False, 0, 1, False)
    probability = predict_abandonment_probability(model, state)
    assert 0 <= probability <= 1
    assert classify_abandonment(probability, threshold=0.3) >= classify_abandonment(probability, threshold=0.7)
    with pytest.raises(ValueError):
        classify_abandonment(probability, threshold=1.1)
