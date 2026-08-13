"""Run Chapter 10's deterministic digital conversion laboratory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml import (  # noqa: E402
    CONVERSION_FEATURES, PartialConversionState, build_conversion_features,
    build_conversion_pipeline, build_conversion_snapshots, build_conversion_targets,
    calculate_conversion_rate, calculate_majority_baseline, classify_conversion,
    evaluate_conversion_model, group_events_by_session, load_member_events,
    predict_conversion_probability, split_conversion_dataset, train_conversion_model,
)


def main() -> None:
    sessions = group_events_by_session(load_member_events(ROOT / "data/harbor_member_events.csv"))
    snapshots = build_conversion_snapshots(sessions)
    X, y = build_conversion_features(snapshots), build_conversion_targets(snapshots)
    split = split_conversion_dataset(X, y)

    print("Harbor Federal Credit Union\nDigital Conversion Prediction Laboratory")
    print("\nProduct:\nHarbor Horizon Savings")
    print("\nPrediction point:\nproduct_details_viewed")
    print(f"\nEligible sessions: {len(snapshots)}")
    print(f"Application starts: {sum(y)}")
    print(f"Baseline conversion rate: {calculate_conversion_rate(snapshots):.1%}")
    print(f"Naive majority-class accuracy: {calculate_majority_baseline(y):.1%}")
    print("\nFeatures:")
    for feature in CONVERSION_FEATURES:
        print(f"- {feature}")
    print(f"\nTraining observations: {len(split.y_train)}\nTest observations: {len(split.y_test)}")
    print("\nTraining conversion model...")
    model = train_conversion_model(build_conversion_pipeline(), split.X_train, split.y_train)
    print("Model trained.")
    evaluation = evaluate_conversion_model(model, split.X_test, split.y_test)
    print(f"\nModel accuracy: {evaluation.accuracy:.3f}")
    print("Confusion matrix (rows=actual 0/1, columns=predicted 0/1):")
    print(evaluation.confusion_matrix)
    print(f"False positives: {evaluation.false_positives}\nFalse negatives: {evaluation.false_negatives}")

    scenarios = {
        "Smooth product exploration": PartialConversionState(
            "web", "internal_navigation", "desktop", 45, 3, 0, 0, 0, 10),
        "Friction-heavy session": PartialConversionState(
            "mobile", "search", "phone", 180, 7, 3, 2, 1, 18),
        "Ambiguous session": PartialConversionState(
            "web", "direct", "tablet", 90, 5, 1, 0, 0, 14),
    }
    print("\nFictional early-session scenarios:")
    probabilities = {}
    for name, state in scenarios.items():
        probability = predict_conversion_probability(model, state)
        probabilities[name] = probability
        print(f"{name}: probability={probability:.3f}, class_at_0.50={int(classify_conversion(probability))}")
    probability = probabilities["Ambiguous session"]
    print(f"\nOne fitted-model probability reused across thresholds: {probability:.3f}")
    for threshold in (0.30, 0.50, 0.70):
        print(f"threshold {threshold:.2f}: predicted_conversion={classify_conversion(probability, threshold=threshold)}")

    print("\nAggregate observed rates (associations, not causal effects):")
    for channel in ("mobile", "web"):
        rows = [row for row in snapshots if row.channel == channel]
        print(f"{channel}: sessions={len(rows)}, actual_conversion_rate={calculate_conversion_rate(rows):.1%}")
    print("\nAll data and relationships are fictional. This product-analytics signal must not")
    print("control eligibility, approval, pricing, interest rates, or credit decisions.")


if __name__ == "__main__":
    main()
