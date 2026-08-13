from datetime import datetime, timedelta, timezone

import pytest

from harbor_ml.human_review import (
    PROHIBITED_AUDIT_FIELDS, InMemoryReviewRepository, ReviewAuditEvent,
    ReviewerReason, ReviewRoutingPolicy, ReviewStatus, RoutingSource,
    calculate_review_metrics, compare_routing_policies, create_review_case,
    reviewer_agreement, route_with_fallback, should_route, simulate_review_queue,
)

NOW = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
POLICY = ReviewRoutingPolicy("review-policy-v2", .7)


def test_routing_threshold_is_inclusive_and_validated():
    assert should_route(.7, POLICY)
    assert not should_route(.699, POLICY)
    with pytest.raises(ValueError, match="between 0 and 1"):
        should_route(1.1, POLICY)


def test_case_creation_retains_model_and_policy_versions():
    case = create_review_case("case-001", "review-routing", "model-v13", .84, POLICY, NOW)
    assert case.status is ReviewStatus.QUEUED
    assert case.model_version == "model-v13"
    assert case.policy_version == "review-policy-v2"
    assert case.routing_threshold == .7
    assert case.queued_at == NOW


def test_valid_transitions_and_override_do_not_mutate_prediction():
    repository = InMemoryReviewRepository()
    original = repository.create(create_review_case(
        "case-001", "review-routing", "model-v13", .84, POLICY, NOW))
    started = repository.transition("case-001", ReviewStatus.IN_REVIEW,
                                    NOW + timedelta(minutes=5), reviewer_id="reviewer-01")
    resolved = repository.transition(
        "case-001", ReviewStatus.RESOLVED_NO_ISSUE, NOW + timedelta(minutes=12),
        reviewer_id="reviewer-01", reason=ReviewerReason.EXPECTED_PATTERN)
    assert started.queue_wait_seconds == 300
    assert resolved.review_duration_seconds == 420
    assert resolved.model_probability == original.model_probability == .84
    assert resolved.reviewer_reason is ReviewerReason.EXPECTED_PATTERN


def test_invalid_state_transition_and_reason_are_rejected():
    repository = InMemoryReviewRepository()
    repository.create(create_review_case("case-001", "review-routing", "v1", .8, POLICY, NOW))
    with pytest.raises(ValueError, match="invalid review transition"):
        repository.transition("case-001", ReviewStatus.RESOLVED_NO_ISSUE, NOW)
    repository.transition("case-001", ReviewStatus.IN_REVIEW, NOW, reviewer_id="reviewer-01")
    with pytest.raises(ValueError, match="supported reason"):
        repository.transition("case-001", ReviewStatus.RESOLVED_NO_ISSUE, NOW,
                              reviewer_id="reviewer-01", reason="free text")  # type: ignore[arg-type]


def test_audit_events_are_created_and_privacy_allowlisted():
    repository = InMemoryReviewRepository()
    repository.create(create_review_case("case-001", "review-routing", "v1", .8, POLICY, NOW))
    event = repository.events[0]
    assert isinstance(event, ReviewAuditEvent)
    assert event.event_type == "case_created"
    assert not set(event.details) & PROHIBITED_AUDIT_FIELDS
    with pytest.raises(ValueError, match="privacy allowlist"):
        ReviewAuditEvent("case-002", NOW, "bad", {"password": "secret"})


def test_ml_unavailable_uses_only_deterministic_fallback():
    unavailable = route_with_fallback(None, POLICY)
    fallback = route_with_fallback(None, POLICY, deterministic_rule_triggered=True)
    assert unavailable == (type(unavailable))(False, RoutingSource.NOT_ROUTED)
    assert fallback.routed and fallback.source is RoutingSource.DETERMINISTIC_RULE
    case = create_review_case("case-002", "review-routing", "v1", None, POLICY, NOW,
                              deterministic_rule_triggered=True)
    assert case.model_probability is None and case.status is ReviewStatus.QUEUED


def test_queue_capacity_and_backlog_calculation():
    queue = simulate_review_queue(200, .6, reviewers=4, cases_per_reviewer_per_day=20)
    assert queue.cases_routed == 120
    assert queue.review_capacity == 80
    assert queue.backlog_change == 40


def test_policy_comparison_reuses_the_same_probabilities():
    scores = [.1, .55, .65, .75, .9]
    actual = [0, 1, 0, 1, 1]
    policies = [ReviewRoutingPolicy("review-policy-v1", .5), POLICY]
    results = compare_routing_policies(actual, scores, policies, review_capacity=2)
    assert [(item.policy_version, item.cases_routed) for item in results] == [
        ("review-policy-v1", 4), ("review-policy-v2", 2)]
    assert results[0].precision == .75 and results[1].precision == 1
    assert scores == [.1, .55, .65, .75, .9]


def test_review_metrics_and_reviewer_agreement():
    repository = InMemoryReviewRepository()
    for identifier, probability in (("case-1", .84), ("case-2", .9), ("case-3", .2)):
        repository.create(create_review_case(identifier, "review-routing", "v1",
                                             probability, POLICY, NOW))
    repository.transition("case-1", ReviewStatus.IN_REVIEW, NOW + timedelta(seconds=10),
                          reviewer_id="reviewer-01")
    repository.transition("case-1", ReviewStatus.RESOLVED_NO_ISSUE,
                          NOW + timedelta(seconds=30), reviewer_id="reviewer-01",
                          reason=ReviewerReason.EXPECTED_PATTERN)
    metrics = calculate_review_metrics(repository.cases())
    assert metrics.queue_length == 1 and metrics.resolution_count == 1
    assert metrics.average_wait_seconds == 10 and metrics.override_rate == 1
    assert reviewer_agreement([
        (ReviewStatus.RESOLVED_NO_ISSUE, ReviewStatus.RESOLVED_NO_ISSUE),
        (ReviewStatus.RESOLVED_NO_ISSUE, ReviewStatus.RESOLVED_FOLLOW_UP),
    ]) == .5
    assert reviewer_agreement([]) is None
