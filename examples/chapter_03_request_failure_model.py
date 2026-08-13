"""Train and evaluate Chapter 3's fictional request-failure model."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from harbor_ml import (  # noqa: E402
    FEATURE_NAMES,
    build_feature_matrix,
    build_model,
    build_target_vector,
    evaluate_model,
    load_request_outcomes,
    predict_request_failure,
    split_dataset,
    train_model,
)

NEW_REQUEST = {
    "vendor_latency_ms": 1650,
    "queue_depth": 103,
    "db_connections": 74,
    "retry_count": 3,
}


def main() -> None:
    """Run the complete deterministic training, evaluation, and inference path."""
    outcomes = load_request_outcomes(
        REPOSITORY_ROOT / "data/harbor_request_outcomes.csv"
    )
    X = build_feature_matrix(outcomes)
    y = build_target_vector(outcomes)
    split = split_dataset(X, y)

    print("Harbor Federal Credit Union")
    print("Request Failure Model Laboratory\n")
    print("Engineering question:\nWill this vendor-backed request fail?\n")
    print("Features:")
    for feature in FEATURE_NAMES:
        print(f"- {feature}")
    print("\nTarget:\nrequest_failed\n")
    print(f"Historical observations: {len(outcomes)}\n")
    print(f"Training observations: {len(split.y_train)}")
    print(f"Test observations: {len(split.y_test)}\n")
    print("Training model...")
    model = train_model(build_model(), split.X_train, split.y_train)
    print("Model trained.\n")

    evaluation = evaluate_model(model, split.X_test, split.y_test)
    print("Test predictions:")
    for actual, predicted in zip(split.y_test, evaluation.predictions):
        print(f"actual={actual} predicted={predicted}")
    print(f"\nAccuracy: {evaluation.accuracy:.1%}")
    print(f"Correct predictions: {evaluation.correct}")
    print(f"Incorrect predictions: {evaluation.incorrect}")
    print(
        "Confusion counts: "
        f"TN={evaluation.true_negatives} FP={evaluation.false_positives} "
        f"FN={evaluation.false_negatives} TP={evaluation.true_positives}"
    )

    prediction = predict_request_failure(model, NEW_REQUEST)
    print("\nNew Harbor request\n")
    print(f"Vendor latency: {NEW_REQUEST['vendor_latency_ms']} ms")
    print(f"Queue depth: {NEW_REQUEST['queue_depth']}")
    print(f"DB connections: {NEW_REQUEST['db_connections']}")
    print(f"Retry count: {NEW_REQUEST['retry_count']}\n")
    print(f"Predicted failure probability: {prediction.failure_probability:.2f}")
    label = "failure" if prediction.predicted_class == 1 else "success"
    print(f"Predicted class: {label}\n")
    print("This prediction is evidence for investigation, not a root-cause diagnosis")
    print("or authorization to change the vendor integration.")


if __name__ == "__main__":
    main()
