"""Train Chapter 4's detector and score fictional application scenarios."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from harbor_ml import (  # noqa: E402
    ANOMALY_FEATURE_NAMES,
    build_anomaly_detector,
    build_anomaly_features,
    load_anomaly_scenarios,
    load_normal_telemetry,
    observation_features,
    score_observation,
    train_anomaly_detector,
)


def main() -> None:
    """Run deterministic baseline training and scenario inference."""
    baseline = load_normal_telemetry(REPOSITORY_ROOT / "data/harbor_normal_telemetry.csv")
    scenarios = load_anomaly_scenarios(REPOSITORY_ROOT / "data/harbor_anomaly_scenarios.csv")

    print("Harbor Federal Credit Union")
    print("Application Anomaly Detection Laboratory\n")
    print(f"Training baseline observations: {len(baseline)}\n")
    print("Features:")
    for feature in ANOMALY_FEATURE_NAMES:
        print(f"- {feature}")
    print("\nTraining Isolation Forest...")
    detector = train_anomaly_detector(
        build_anomaly_detector(), build_anomaly_features(baseline)
    )
    print("Model trained.\n")
    print("Evaluating scenarios:")
    for scenario in scenarios:
        result = score_observation(detector, observation_features(scenario.observation))
        classification = "anomaly" if result.is_anomaly else "normal"
        print(f"\n{scenario.name}")
        print(f"score: {result.score:.4f}")
        print(f"classification: {classification}")
    print("\nScores are transformed model values, not probabilities.")
    print("Flags invite investigation; they do not identify root cause or authorize remediation.")


if __name__ == "__main__":
    main()
