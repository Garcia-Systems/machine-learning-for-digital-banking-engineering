"""Train Chapter 12's detector and score held-out fictional scenarios."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.transaction_anomaly import (  # noqa: E402
    TRANSACTION_FEATURES,
    build_transaction_anomaly_pipeline,
    load_transaction_observations,
    load_transaction_scenarios,
    score_transaction,
    train_transaction_anomaly_detector,
)


def main() -> None:
    baseline = load_transaction_observations(ROOT / "data/harbor_transaction_observations.csv")
    scenarios = load_transaction_scenarios(ROOT / "data/harbor_transaction_scenarios.csv")
    print("Harbor Federal Credit Union")
    print("Transaction Anomaly Detection Laboratory\n")
    print(f"Training baseline observations: {len(baseline)}")
    print("Evaluation scenarios in training: 0\n")
    print("Features:")
    for feature in TRANSACTION_FEATURES:
        print(f"- {feature}")
    print("\nTraining Isolation Forest...")
    model = train_transaction_anomaly_detector(
        build_transaction_anomaly_pipeline(), baseline,
    )
    print("Model trained.")
    for scenario in scenarios:
        result = score_transaction(model, scenario.observation)
        classification = "anomaly" if result.is_anomaly else "normal"
        print(f"\nScenario: {scenario.name}")
        print(f"anomaly score: {result.raw_score:.4f}")
        print(f"classification: {classification}")
    print("\nHigher scores mean more unusual relative to this fitted baseline.")
    print("Scores are not probabilities and anomalies are not findings of wrongdoing.")
    print("Use a flag only as a controlled review or observability signal.")


if __name__ == "__main__":
    main()
