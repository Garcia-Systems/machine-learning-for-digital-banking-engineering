"""Train and explore Chapter 5's fictional incident classifier."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from harbor_ml import (  # noqa: E402
    INCIDENT_CLASSES,
    build_incident_classifier,
    build_incident_features,
    build_incident_targets,
    evaluate_incident_classifier,
    load_incident_dataset,
    predict_incident,
    split_incident_dataset,
    train_incident_classifier,
)

SCENARIOS = {
    "vendor slowdown": {
        "api_latency_ms": 1840, "error_rate": 0.061, "db_connections": 58,
        "queue_depth": 112, "vendor_latency_ms": 1720,
        "requests_per_minute": 710,
    },
    "database pressure": {
        "api_latency_ms": 1120, "error_rate": 0.043, "db_connections": 101,
        "queue_depth": 142, "vendor_latency_ms": 310,
        "requests_per_minute": 760,
    },
    "traffic spike": {
        "api_latency_ms": 720, "error_rate": 0.014, "db_connections": 55,
        "queue_depth": 52, "vendor_latency_ms": 300,
        "requests_per_minute": 2110,
    },
    "application regression": {
        "api_latency_ms": 1510, "error_rate": 0.072, "db_connections": 49,
        "queue_depth": 60, "vendor_latency_ms": 290,
        "requests_per_minute": 650,
    },
    "ambiguous dependency pressure": {
        "api_latency_ms": 1050, "error_rate": 0.039, "db_connections": 72,
        "queue_depth": 91, "vendor_latency_ms": 710,
        "requests_per_minute": 760,
    },
}


def _print_matrix(labels, matrix) -> None:
    width = max(12, max(map(len, labels)) + 2)
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(" " * width + "".join(f"{label[:10]:>12}" for label in labels))
    for label, row in zip(labels, matrix, strict=True):
        print(f"{label:<{width}}" + "".join(f"{value:>12}" for value in row))


def main() -> None:
    observations = load_incident_dataset(
        REPOSITORY_ROOT / "data/harbor_incident_classes.csv"
    )
    split = split_incident_dataset(
        build_incident_features(observations), build_incident_targets(observations)
    )
    print("Harbor Federal Credit Union")
    print("Incident Classification Laboratory\n")
    print("Known classes:")
    for label in INCIDENT_CLASSES:
        print(f"- {label}")
    print(f"\nHistorical observations: {len(observations)}")
    print(f"Training observations: {len(split.y_train)}")
    print(f"Test observations: {len(split.y_test)}\n")
    print("Training classifier...")
    model = train_incident_classifier(
        build_incident_classifier(), split.X_train, split.y_train
    )
    print("Model trained.")
    evaluation = evaluate_incident_classifier(model, split.X_test, split.y_test)
    print(f"\nTest accuracy: {evaluation.accuracy:.3f}")
    _print_matrix(evaluation.labels, evaluation.confusion_matrix)

    for name, observation in SCENARIOS.items():
        result = predict_incident(model, observation)
        print(f"\nScenario: {name}\n\nClass probabilities:")
        for label in INCIDENT_CLASSES:
            print(f"{label:<24}{result.probabilities[label]:.3f}")
        print(f"\nPredicted class: {result.predicted_class}")
        ordered = sorted(result.probabilities.values(), reverse=True)
        print(f"Top-two probability gap: {ordered[0] - ordered[1]:.3f}")
    print("\nPredictions identify resemblance to known history, not proven root cause.")


if __name__ == "__main__":
    main()
