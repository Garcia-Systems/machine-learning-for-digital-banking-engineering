"""Run Chapter 13's deterministic operational review-routing laboratory."""

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.review_routing import (  # noqa: E402
    REVIEW_FEATURES, ReviewObservation, build_review_features, build_review_pipeline,
    build_review_targets, calculate_majority_baseline, evaluate_review_model,
    load_review_dataset, predict_review_probabilities, predict_review_probability,
    split_review_dataset, train_review_model,
)
from harbor_ml.transaction_anomaly import (  # noqa: E402
    TransactionObservation, build_transaction_anomaly_pipeline,
    load_transaction_observations, score_transaction, train_transaction_anomaly_detector,
)


SCENARIOS = {
    "routine_observation": ReviewObservation("debit_purchase", "mobile", "25_to_99", 14, 1, 55, False, .10, 0),
    "large_but_ordinary_pattern": ReviewObservation("internal_transfer", "branch_assisted", "2500_plus", 11, 1, 90, False, .12, 0),
    "repeated_failures_plus_behavior_shift": ReviewObservation("external_transfer", "web", "500_to_999", 2, 7, 2, True, .82, 3),
    "new_device_only": ReviewObservation("debit_purchase", "mobile", "100_to_499", 15, 2, 35, True, .14, 0),
    "mixed_ambiguous_scenario": ReviewObservation("external_transfer", "mobile", "100_to_499", 22, 4, 12, False, .48, 1),
}


def main() -> None:
    rows = load_review_dataset(ROOT / "data/harbor_review_routing.csv")
    X, y = build_review_features(rows), build_review_targets(rows)
    split = split_review_dataset(X, y)
    model = train_review_model(build_review_pipeline(), split.X_train, split.y_train)
    probabilities = predict_review_probabilities(model, split.X_test)
    print("Harbor Federal Credit Union\nClassification and Risk Signals Laboratory")
    print(f"\nRows: {len(y)}\nnot reviewed: {(y == 0).sum()}\nreviewed: {(y == 1).sum()}")
    print(f"majority-class baseline accuracy: {calculate_majority_baseline(y):.3f}")
    print("\nRouting-time features:", ", ".join(REVIEW_FEATURES))
    print("\nOne fitted model and one held-out probability vector reused at every threshold:")
    for threshold in (.30, .50, .70):
        result = evaluate_review_model(split.y_test, probabilities, threshold=threshold)
        print(f"threshold={threshold:.2f} review_count={result.predicted_review_count} "
              f"accuracy={result.accuracy:.3f} precision={result.precision:.3f} "
              f"recall={result.recall:.3f} false_positives={result.false_positives} "
              f"false_negatives={result.false_negatives}")
    print("\nFitted-model scenarios:")
    for name, observation in SCENARIOS.items():
        print(f"{name}: manual_review_probability={predict_review_probability(model, observation):.3f}")

    anomaly_rows = load_transaction_observations(ROOT / "data/harbor_transaction_observations.csv")
    anomaly_model = train_transaction_anomaly_detector(build_transaction_anomaly_pipeline(), anomaly_rows)
    print("\nDifferent models, different questions:")
    for name in ("large_but_ordinary_pattern", "repeated_failures_plus_behavior_shift"):
        item = SCENARIOS[name]
        anomaly_item = TransactionObservation(datetime(2025, 1, 1, tzinfo=timezone.utc),
            item.transaction_type, item.channel, item.amount_band, item.hour_of_day,
            item.recent_transaction_count, item.minutes_since_previous_transaction,
            item.device_change, item.distance_from_recent_pattern)
        print(f"{name}: anomaly_score={score_transaction(anomaly_model, anomaly_item).raw_score:.3f} "
              f"review_probability={predict_review_probability(model, item):.3f}")
    print("\nThese are synthetic historical-routing estimates—not fraud, criminality, credit,")
    print("eligibility, member-quality, or wrongdoing probabilities. Policy controls routing.")


if __name__ == "__main__":
    main()
