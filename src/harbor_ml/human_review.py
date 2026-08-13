"""Deterministic human-review workflow components for Chapter 24."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from math import ceil, isfinite
from typing import Mapping, Sequence

from .model_evaluation import evaluate_threshold


class ReviewStatus(Enum):
    NOT_ROUTED = "not_routed"
    QUEUED = "queued"
    IN_REVIEW = "in_review"
    RESOLVED_NO_ISSUE = "resolved_no_issue"
    RESOLVED_FOLLOW_UP = "resolved_follow_up"
    ESCALATED = "escalated"


class ReviewerReason(Enum):
    EXPECTED_PATTERN = "expected_pattern"
    TEMPORARY_VENDOR_ISSUE = "temporary_vendor_issue"
    DUPLICATE_OPERATIONAL_SIGNAL = "duplicate_operational_signal"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    REQUIRES_FOLLOW_UP = "requires_follow_up"


class RoutingSource(Enum):
    MODEL_POLICY = "model_policy"
    DETERMINISTIC_RULE = "deterministic_rule"
    NOT_ROUTED = "not_routed"


@dataclass(frozen=True)
class ReviewRoutingPolicy:
    version: str
    threshold: float

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("policy version cannot be empty")
        _validate_probability(self.threshold, "threshold")


@dataclass(frozen=True)
class RoutingResult:
    routed: bool
    source: RoutingSource


def _validate_probability(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite value between 0 and 1")


def should_route(probability: float, policy: ReviewRoutingPolicy) -> bool:
    """Apply the explicit policy threshold; the model does not make this decision."""
    _validate_probability(probability, "probability")
    return probability >= policy.threshold


def route_with_fallback(probability: float | None, policy: ReviewRoutingPolicy, *,
                        deterministic_rule_triggered: bool = False) -> RoutingResult:
    """Route safely when ML is unavailable without inventing a zero score."""
    if deterministic_rule_triggered:
        return RoutingResult(True, RoutingSource.DETERMINISTIC_RULE)
    if probability is None:
        return RoutingResult(False, RoutingSource.NOT_ROUTED)
    routed = should_route(probability, policy)
    return RoutingResult(routed, RoutingSource.MODEL_POLICY if routed else RoutingSource.NOT_ROUTED)


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    model_name: str
    model_version: str
    model_probability: float | None
    policy_version: str
    routing_threshold: float
    routing_source: RoutingSource
    status: ReviewStatus
    created_at: datetime
    queued_at: datetime | None = None
    review_started_at: datetime | None = None
    resolved_at: datetime | None = None
    reviewer_id: str | None = None
    reviewer_reason: ReviewerReason | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.model_name.strip() or not self.model_version.strip():
            raise ValueError("case, model name, and model version cannot be empty")
        if not self.policy_version.strip():
            raise ValueError("policy version cannot be empty")
        if self.model_probability is not None:
            _validate_probability(self.model_probability, "model probability")
        _validate_probability(self.routing_threshold, "routing threshold")

    @property
    def queue_wait_seconds(self) -> float | None:
        if self.queued_at is None or self.review_started_at is None:
            return None
        return (self.review_started_at - self.queued_at).total_seconds()

    @property
    def review_duration_seconds(self) -> float | None:
        if self.review_started_at is None or self.resolved_at is None:
            return None
        return (self.resolved_at - self.review_started_at).total_seconds()


AUDIT_DETAIL_FIELDS = frozenset({
    "model_name", "model_version", "model_probability", "policy_version",
    "routing_threshold", "routing_source", "review_status", "reviewer_id", "reason_code",
})
PROHIBITED_AUDIT_FIELDS = frozenset({
    "access_token", "password", "cookie", "authentication_cookie", "member_name",
    "account_number", "card_number", "ssn",
})
AuditValue = str | float | bool | None


@dataclass(frozen=True)
class ReviewAuditEvent:
    case_id: str
    timestamp: datetime
    event_type: str
    details: Mapping[str, AuditValue]

    def __post_init__(self) -> None:
        keys = set(self.details)
        if not keys <= AUDIT_DETAIL_FIELDS:
            raise ValueError(f"audit details contain fields outside the privacy allowlist: {sorted(keys - AUDIT_DETAIL_FIELDS)}")
        if keys & PROHIBITED_AUDIT_FIELDS:
            raise ValueError("audit details contain prohibited sensitive fields")


_TRANSITIONS = {
    ReviewStatus.NOT_ROUTED: frozenset({ReviewStatus.ESCALATED}),
    ReviewStatus.QUEUED: frozenset({ReviewStatus.IN_REVIEW, ReviewStatus.ESCALATED}),
    ReviewStatus.IN_REVIEW: frozenset({
        ReviewStatus.RESOLVED_NO_ISSUE, ReviewStatus.RESOLVED_FOLLOW_UP,
        ReviewStatus.ESCALATED,
    }),
    ReviewStatus.ESCALATED: frozenset({ReviewStatus.IN_REVIEW}),
    ReviewStatus.RESOLVED_NO_ISSUE: frozenset(),
    ReviewStatus.RESOLVED_FOLLOW_UP: frozenset(),
}


class InMemoryReviewRepository:
    """Small append-only-audit repository for the educational laboratory."""

    def __init__(self) -> None:
        self._cases: dict[str, ReviewCase] = {}
        self._events: list[ReviewAuditEvent] = []

    def create(self, case: ReviewCase) -> ReviewCase:
        if case.case_id in self._cases:
            raise ValueError(f"review case already exists: {case.case_id}")
        self._cases[case.case_id] = case
        self._append(case, "case_created")
        return case

    def get(self, case_id: str) -> ReviewCase:
        try:
            return self._cases[case_id]
        except KeyError as error:
            raise KeyError(f"unknown review case: {case_id}") from error

    def transition(self, case_id: str, status: ReviewStatus, timestamp: datetime, *,
                   reviewer_id: str | None = None,
                   reason: ReviewerReason | None = None) -> ReviewCase:
        case = self.get(case_id)
        if status not in _TRANSITIONS[case.status]:
            raise ValueError(f"invalid review transition: {case.status.value} -> {status.value}")
        resolved = status in {ReviewStatus.RESOLVED_NO_ISSUE, ReviewStatus.RESOLVED_FOLLOW_UP}
        if resolved and (reviewer_id is None or reason is None):
            raise ValueError("resolved reviews require a reviewer identifier and reason code")
        if reason is not None and not isinstance(reason, ReviewerReason):
            raise ValueError("reviewer reason must be a supported reason code")
        updates: dict[str, object] = {"status": status}
        if status is ReviewStatus.IN_REVIEW:
            updates.update(review_started_at=timestamp, reviewer_id=reviewer_id)
        if resolved:
            updates.update(resolved_at=timestamp, reviewer_id=reviewer_id,
                           reviewer_reason=reason)
        updated = replace(case, **updates)
        self._cases[case_id] = updated
        self._append(updated, f"status_{status.value}", timestamp)
        return updated

    def events_for(self, case_id: str) -> tuple[ReviewAuditEvent, ...]:
        return tuple(event for event in self._events if event.case_id == case_id)

    @property
    def events(self) -> tuple[ReviewAuditEvent, ...]:
        return tuple(self._events)

    def cases(self) -> tuple[ReviewCase, ...]:
        return tuple(self._cases.values())

    def _append(self, case: ReviewCase, event_type: str,
                timestamp: datetime | None = None) -> None:
        details: dict[str, AuditValue] = {
            "model_name": case.model_name,
            "model_version": case.model_version,
            "model_probability": case.model_probability,
            "policy_version": case.policy_version,
            "routing_threshold": case.routing_threshold,
            "routing_source": case.routing_source.value,
            "review_status": case.status.value,
            "reviewer_id": case.reviewer_id,
            "reason_code": case.reviewer_reason.value if case.reviewer_reason else None,
        }
        self._events.append(ReviewAuditEvent(case.case_id, timestamp or case.created_at,
                                             event_type, details))


def create_review_case(case_id: str, model_name: str, model_version: str,
                       probability: float | None, policy: ReviewRoutingPolicy,
                       timestamp: datetime, *,
                       deterministic_rule_triggered: bool = False) -> ReviewCase:
    result = route_with_fallback(probability, policy,
                                 deterministic_rule_triggered=deterministic_rule_triggered)
    status = ReviewStatus.QUEUED if result.routed else ReviewStatus.NOT_ROUTED
    return ReviewCase(case_id, model_name, model_version, probability, policy.version,
                      policy.threshold, result.source, status, timestamp,
                      queued_at=timestamp if result.routed else None)


@dataclass(frozen=True)
class QueueCapacity:
    observations: int
    cases_routed: int
    review_capacity: int
    backlog_change: int


def simulate_review_queue(observations_per_day: int, predicted_positive_rate: float,
                          reviewers: int, cases_per_reviewer_per_day: int) -> QueueCapacity:
    if observations_per_day < 0 or reviewers < 0 or cases_per_reviewer_per_day < 0:
        raise ValueError("queue inputs cannot be negative")
    _validate_probability(predicted_positive_rate, "predicted positive rate")
    routed = ceil(observations_per_day * predicted_positive_rate)
    capacity = reviewers * cases_per_reviewer_per_day
    return QueueCapacity(observations_per_day, routed, capacity, routed - capacity)


@dataclass(frozen=True)
class PolicyComparison:
    policy_version: str
    threshold: float
    cases_routed: int
    precision: float
    recall: float
    review_capacity: int
    backlog_change: int


def compare_routing_policies(actual_labels: Sequence[int], probabilities: Sequence[float],
                             policies: Sequence[ReviewRoutingPolicy], *,
                             review_capacity: int) -> tuple[PolicyComparison, ...]:
    if review_capacity < 0:
        raise ValueError("review capacity cannot be negative")
    # Every policy is deliberately evaluated against this same immutable score sequence.
    scores = tuple(probabilities)
    output = []
    for policy in policies:
        result = evaluate_threshold(actual_labels, scores, policy.threshold)
        routed = result.predicted_failures
        output.append(PolicyComparison(policy.version, policy.threshold, routed,
            result.metrics.precision, result.metrics.recall, review_capacity,
            routed - review_capacity))
    return tuple(output)


@dataclass(frozen=True)
class ReviewMetrics:
    queue_length: int
    resolution_count: int
    average_wait_seconds: float | None
    override_rate: float | None


def calculate_review_metrics(cases: Sequence[ReviewCase]) -> ReviewMetrics:
    queued = sum(case.status in {ReviewStatus.QUEUED, ReviewStatus.IN_REVIEW,
                                 ReviewStatus.ESCALATED} for case in cases)
    resolved = [case for case in cases if case.status in {
        ReviewStatus.RESOLVED_NO_ISSUE, ReviewStatus.RESOLVED_FOLLOW_UP}]
    waits = [case.queue_wait_seconds for case in cases if case.queue_wait_seconds is not None]
    routed_resolved = [case for case in resolved if case.routing_source is RoutingSource.MODEL_POLICY]
    overrides = sum(case.status is ReviewStatus.RESOLVED_NO_ISSUE for case in routed_resolved)
    return ReviewMetrics(queued, len(resolved),
                         sum(waits) / len(waits) if waits else None,
                         overrides / len(routed_resolved) if routed_resolved else None)


def reviewer_agreement(review_pairs: Sequence[tuple[ReviewStatus, ReviewStatus]]) -> float | None:
    """Return simple agreement, which does not establish reviewer correctness."""
    if not review_pairs:
        return None
    return sum(first is second for first, second in review_pairs) / len(review_pairs)
