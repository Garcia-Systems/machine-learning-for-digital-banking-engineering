"""Run Chapter 7's deterministic request-failure prediction laboratory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml import (  # noqa: E402
    IntegrationRequest,
    build_integration_features,
    build_integration_pipeline,
    build_integration_targets,
    evaluate_integration_model,
    load_integration_requests,
    predict_failure,
    predict_failure_probability,
    split_integration_dataset,
    train_integration_model,
)


def request(vendor: str, endpoint: str, **changes: int | float) -> IntegrationRequest:
    values: dict[str, str | int | float] = {
        "vendor": vendor,
        "endpoint": endpoint,
        "recent_vendor_latency_ms": 180,
        "recent_vendor_error_rate": 0.015,
        "queue_depth": 10,
        "retry_count": 0,
        "request_size_bytes": 1200,
        "hour_of_day": 10,
    }
    values.update(changes)
    return IntegrationRequest(**values)  # type: ignore[arg-type]


def main() -> None:
    observations = load_integration_requests(ROOT / "data/harbor_integration_requests.csv")
    split = split_integration_dataset(
        build_integration_features(observations), build_integration_targets(observations)
    )
    model = train_integration_model(
        build_integration_pipeline(), split.X_train, split.y_train
    )
    result = evaluate_integration_model(model, split.X_test, split.y_test)

    print("Harbor Federal Credit Union")
    print("Integration Failure Prediction Laboratory")
    print(f"\nObservations: {len(observations)} (train={len(split.y_train)}, test={len(split.y_test)})")
    print(f"Accuracy: {result.accuracy:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(result.confusion_matrix)
    print(f"False positives: {result.false_positives}")
    print(f"False negatives: {result.false_negatives}")

    scenarios = {
        "healthy request": request("ClearVerify", "identity_verify"),
        "degraded vendor": request(
            "Northstar Payments", "transfer_status",
            recent_vendor_latency_ms=720, recent_vendor_error_rate=0.15, queue_depth=48,
        ),
        "heavy request during pressure": request(
            "BlueCurrent Documents", "statement_fetch",
            recent_vendor_latency_ms=480, recent_vendor_error_rate=0.08,
            queue_depth=88, request_size_bytes=7800,
        ),
        "retrying request": request(
            "HarborLink Core Gateway", "account_summary",
            recent_vendor_latency_ms=510, recent_vendor_error_rate=0.12,
            queue_depth=45, retry_count=2,
        ),
    }
    print("\nScenarios (threshold 0.50)")
    for name, scenario in scenarios.items():
        prediction = predict_failure(model, scenario)
        print(f"\n{name}")
        print(f"  vendor: {scenario.vendor}; endpoint: {scenario.endpoint}")
        print(
            f"  latency={scenario.recent_vendor_latency_ms} ms; "
            f"error_rate={scenario.recent_vendor_error_rate:.3f}; "
            f"queue={scenario.queue_depth}; retries={scenario.retry_count}; "
            f"size={scenario.request_size_bytes} bytes"
        )
        print(f"  predicted failure probability: {prediction.probability:.3f}")
        print(f"  class: {'failure' if prediction.predicted_failure else 'success'}")

    comparison = scenarios["heavy request during pressure"]
    probability = predict_failure_probability(model, comparison)
    print(f"\nSame model probability across thresholds: {probability:.3f}")
    for threshold in (0.30, 0.50, 0.70):
        label = "failure" if predict_failure(model, comparison, threshold=threshold).predicted_failure else "success"
        print(f"  threshold {threshold:.2f} -> {label}")

    unknown = request(
        "Harbor Experimental Sandbox", "identity_verify",
        recent_vendor_latency_ms=330, recent_vendor_error_rate=0.05,
    )
    print("\nUnknown vendor demonstration")
    print(f"  vendor: {unknown.vendor}")
    print(f"  inference completed; probability: {predict_failure_probability(model, unknown):.3f}")
    print("  Execution does not mean the model understands this vendor.")
    print("\nPredictions are uncertain engineering signals for observability, not proof")
    print("and never permission to bypass Harbor's deterministic controls.")


if __name__ == "__main__":
    main()
