"""Run Chapter 9's deterministic digital journey abandonment laboratory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml import (  # noqa: E402
    JOURNEY_FEATURES, PartialJourneyState, build_abandonment_pipeline,
    build_journey_features, build_journey_targets, build_transfer_snapshots,
    classify_abandonment, evaluate_abandonment_model, group_events_by_session,
    load_member_events, predict_abandonment_probability, split_journey_dataset,
    train_abandonment_model,
)


def main() -> None:
    sessions = group_events_by_session(load_member_events(ROOT / "data/harbor_member_events.csv"))
    snapshots = build_transfer_snapshots(sessions)
    X, y = build_journey_features(snapshots), build_journey_targets(snapshots)
    split = split_journey_dataset(X, y)

    print("Harbor Federal Credit Union\nDigital Journey Abandonment Laboratory")
    print("\nPrediction point:\nrecipient_selected")
    print(f"\nEligible transfer journeys: {len(snapshots)}")
    print(f"Completed: {sum(y == 0)}\nAbandoned: {sum(y == 1)}")
    print("\nFeatures:")
    for feature in JOURNEY_FEATURES:
        print(f"- {feature}")
    print(f"\nTraining observations: {len(split.y_train)}\nTest observations: {len(split.y_test)}")
    print("\nTraining model...")
    model = train_abandonment_model(build_abandonment_pipeline(), split.X_train, split.y_train)
    print("Model trained.")
    evaluation = evaluate_abandonment_model(model, split.X_test, split.y_test)
    print(f"\nAccuracy: {evaluation.accuracy:.3f}")
    print("Confusion matrix (rows=actual 0/1, columns=predicted 0/1):")
    print(evaluation.confusion_matrix)
    print(f"False positives: {evaluation.false_positives}\nFalse negatives: {evaluation.false_negatives}")

    scenarios = {
        "Smooth journey": PartialJourneyState("web", 35, 3, False, 0, 0, False),
        "Journey with friction": PartialJourneyState("mobile", 160, 7, True, 2, 2, False),
        "Ambiguous journey": PartialJourneyState("web", 80, 4, False, 0, 1, False),
    }
    print("\nFictional partial-journey scenarios:")
    probabilities = {}
    for name, state in scenarios.items():
        probability = predict_abandonment_probability(model, state)
        probabilities[name] = probability
        print(f"{name}: probability={probability:.3f}, class_at_0.50={int(classify_abandonment(probability))}")
    probability = probabilities["Ambiguous journey"]
    print(f"\nOne probability reused across thresholds: {probability:.3f}")
    for threshold in (0.30, 0.50, 0.70):
        print(f"threshold {threshold:.2f}: at_risk={classify_abandonment(probability, threshold=threshold)}")
    print("\nSynthetic behavioral associations are not claims about real members or causes.")
    print("The model must not control eligibility, authorization, pricing, or financial decisions.")


if __name__ == "__main__":
    main()
