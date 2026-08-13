from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from harbor_ml.member_behavior import group_events_by_session, load_member_events
from harbor_ml.member_segmentation import (
    SEGMENTATION_FEATURES,
    BehavioralSession,
    assign_clusters,
    build_behavioral_sessions,
    build_segmentation_features,
    build_segmentation_pipeline,
    calculate_inertia_by_k,
    inverse_transformed_centers,
    summarize_clusters,
    train_segmentation_model,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def behavioral_sessions():
    events = load_member_events(ROOT / "data/harbor_member_events.csv")
    return build_behavioral_sessions(group_events_by_session(events))


def test_behavioral_rows_are_derived_from_fixture(behavioral_sessions):
    row = next(item for item in behavioral_sessions if item.session_id == "session-1042")
    assert row.event_count == 7
    assert row.account_view_count == 1
    assert row.statement_view_count == 1
    assert row.session_duration_seconds == 116


def test_schema_is_behavioral_and_feature_order_has_no_target():
    assert SEGMENTATION_FEATURES == (
        "session_duration_seconds", "event_count", "account_view_count", "search_count",
        "transfer_count", "statement_view_count", "help_event_count",
        "verification_event_count",
    )
    names = {field.name for field in fields(BehavioralSession)}
    prohibited = {"name", "account_number", "email", "ssn", "balance", "income",
                  "credit_score", "age", "race", "ethnicity", "religion", "target",
                  "conversion", "abandonment", "incident_type"}
    assert names.isdisjoint(prohibited)
    assert set(SEGMENTATION_FEATURES) <= names


def test_pipeline_scales_before_kmeans():
    pipeline = build_segmentation_pipeline(4)
    assert list(pipeline.named_steps) == ["scaler", "kmeans"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["kmeans"], KMeans)


def test_fit_assignments_and_centers_are_structurally_correct(behavioral_sessions):
    X = build_segmentation_features(behavioral_sessions)
    model = train_segmentation_model(build_segmentation_pipeline(4), X)
    labels = assign_clusters(model, behavioral_sessions)
    assert len(labels) == len(behavioral_sessions)
    assert set(labels) <= set(range(4))
    assert np.isfinite(model.named_steps["kmeans"].inertia_)
    assert model.named_steps["kmeans"].inertia_ >= 0
    centers = inverse_transformed_centers(model)
    assert centers.shape == (4, len(SEGMENTATION_FEATURES))
    assert np.isfinite(centers).all()


def test_fixed_seed_is_deterministic(behavioral_sessions):
    X = build_segmentation_features(behavioral_sessions)
    first = train_segmentation_model(build_segmentation_pipeline(4), X).predict(X)
    second = train_segmentation_model(build_segmentation_pipeline(4), X).predict(X)
    np.testing.assert_array_equal(first, second)


def test_inertia_comparison_is_finite_and_nonincreasing(behavioral_sessions):
    values = calculate_inertia_by_k(build_segmentation_features(behavioral_sessions), (2, 3, 4, 5))
    assert list(values) == [2, 3, 4, 5]
    assert all(np.isfinite(value) and value >= 0 for value in values.values())
    assert list(values.values()) == sorted(values.values(), reverse=True)


def test_summaries_account_for_every_session(behavioral_sessions):
    X = build_segmentation_features(behavioral_sessions)
    model = train_segmentation_model(build_segmentation_pipeline(4), X)
    summaries = summarize_clusters(behavioral_sessions, model.predict(X))
    assert len(summaries) == 4
    assert sum(summary.session_count for summary in summaries) == len(behavioral_sessions)
    assert all(tuple(summary.means) == SEGMENTATION_FEATURES for summary in summaries)


def test_new_session_prediction_uses_fitted_pipeline(behavioral_sessions):
    model = train_segmentation_model(
        build_segmentation_pipeline(4), build_segmentation_features(behavioral_sessions),
    )
    new_row = BehavioralSession("session-9999", 50, 5, 4, 0, 0, 0, 0, 0)
    prediction = assign_clusters(model, [new_row])
    assert prediction.shape == (1,)
    assert int(prediction[0]) in range(4)


def test_malformed_inputs_are_rejected(behavioral_sessions):
    with pytest.raises(ValueError, match="at least one"):
        build_segmentation_features([])
    malformed = replace(behavioral_sessions[0], search_count=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        build_segmentation_features([malformed])
    with pytest.raises(ValueError, match="positive integer"):
        build_segmentation_pipeline(0)
    with pytest.raises(ValueError, match="one cluster assignment"):
        summarize_clusters(behavioral_sessions[:2], [0])
