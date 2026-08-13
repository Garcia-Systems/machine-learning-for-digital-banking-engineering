"""Executable Chapter 24 human-in-the-loop laboratory."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.human_review import (  # noqa: E402
    InMemoryReviewRepository, ReviewerReason, ReviewRoutingPolicy, ReviewStatus,
    compare_routing_policies, create_review_case, route_with_fallback,
    simulate_review_queue,
)
from harbor_ml.review_routing import (  # noqa: E402
    build_review_features, build_review_pipeline, build_review_targets,
    load_review_dataset, predict_review_probabilities, split_review_dataset,
    train_review_model,
)


def main() -> None:
    rows = load_review_dataset(ROOT / "data/harbor_review_routing.csv")
    split = split_review_dataset(build_review_features(rows), build_review_targets(rows))
    model = train_review_model(build_review_pipeline(), split.X_train, split.y_train)
    probabilities = predict_review_probabilities(model, split.X_test)
    policies = (
        ReviewRoutingPolicy("review-policy-v1", .5),
        ReviewRoutingPolicy("review-policy-v2", .7),
    )
    capacity = 30
    comparisons = compare_routing_policies(split.y_test, probabilities, policies,
                                           review_capacity=capacity)

    print("Harbor Federal Credit Union\nHuman-in-the-Loop Laboratory")
    print(f"\nHeld-out observations: {len(probabilities)}")
    print("Same held-out model probabilities are reused for both policies:")
    for result in comparisons:
        print(f"  {result.policy_version}: threshold={result.threshold:.2f}, "
              f"routed={result.cases_routed}, precision={result.precision:.3f}, "
              f"recall={result.recall:.3f}, capacity={result.review_capacity}, "
              f"backlog change={result.backlog_change:+d}")

    selected = comparisons[1]
    queue = simulate_review_queue(len(probabilities),
                                  selected.cases_routed / len(probabilities),
                                  reviewers=3, cases_per_reviewer_per_day=10)
    print(f"\nPolicy:\nreview threshold = {policies[1].threshold:.2f}")
    print(f"Observed cases: {queue.observations}")
    print(f"Routed cases: {queue.cases_routed}")
    print(f"Daily review capacity: {queue.review_capacity}")
    print(f"Projected backlog change: {queue.backlog_change:+d}")

    repository = InMemoryReviewRepository()
    now = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    case = repository.create(create_review_case(
        "case-001", "manual-review-routing", "review-model-v1", .84,
        policies[1], now))
    repository.transition(case.case_id, ReviewStatus.IN_REVIEW,
                          now + timedelta(minutes=4), reviewer_id="reviewer-01")
    resolved = repository.transition(
        case.case_id, ReviewStatus.RESOLVED_NO_ISSUE, now + timedelta(minutes=11),
        reviewer_id="reviewer-01", reason=ReviewerReason.EXPECTED_PATTERN)
    print(f"\nCase: {resolved.case_id}\nModel probability: {resolved.model_probability:.2f}")
    print("Routing result: queued")
    print(f"Reviewer outcome: {resolved.status.value}")
    print(f"Reviewer reason: {resolved.reviewer_reason.value}")
    print("Important: The model prediction remains 0.84.")
    print("The human outcome is recorded separately.")

    unavailable = route_with_fallback(None, policies[1])
    deterministic = route_with_fallback(None, policies[1],
                                        deterministic_rule_triggered=True)
    print("\nModel unavailable (no invented probability):")
    print(f"  model-only routing: {unavailable.routed}")
    print(f"  deterministic-rule routing: {deterministic.routed}")
    print("\nAppend-only audit trail:")
    for event in repository.events:
        print(f"  {event.timestamp.isoformat()} {event.event_type} "
              f"status={event.details['review_status']} policy={event.details['policy_version']}")


if __name__ == "__main__":
    main()
