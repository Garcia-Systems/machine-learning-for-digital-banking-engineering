"""Run Chapter 15's deterministic database performance laboratory."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.database_performance import (  # noqa: E402
    QueryContext, build_database_performance_pipeline, calculate_metrics,
    chronological_split, load_query_performance, predict_query_duration,
    query_family_baseline, residuals, train_database_performance_model,
)


def main() -> None:
    observations = load_query_performance(ROOT / "data/harbor_query_performance.csv")
    split = chronological_split(observations)
    model = train_database_performance_model(build_database_performance_pipeline(), split.train)
    contexts = [x.context for x in split.test]
    actual = np.asarray([x.query_duration_ms for x in split.test])
    baseline = query_family_baseline(split.train, contexts)
    predicted = predict_query_duration(model, contexts)
    baseline_metrics = calculate_metrics(actual, baseline)
    model_metrics = calculate_metrics(actual, predicted)
    errors = residuals(actual, predicted)

    print("Harbor Federal Credit Union\nDatabase Performance Prediction Laboratory")
    print(f"\nHistorical query observations: {len(observations)}")
    print(f"Training observations: {len(split.train)}\nTest observations: {len(split.test)}")
    print("\nQuery families:")
    for family in sorted({x.context.query_family for x in observations}): print(f"- {family}")
    print(f"\nQuery-family median baseline\nMAE: {baseline_metrics.mae:.2f} ms"
          f"\nRMSE: {baseline_metrics.rmse:.2f} ms"
          f"\nMedian absolute error: {baseline_metrics.median_absolute_error:.2f} ms")
    print(f"\nRandom forest model\nMAE: {model_metrics.mae:.2f} ms"
          f"\nRMSE: {model_metrics.rmse:.2f} ms"
          f"\nMedian absolute error: {model_metrics.median_absolute_error:.2f} ms")
    print("\nSample predictions\nquery_family          actual_ms predicted_ms residual_ms")
    for index in np.linspace(0, len(contexts) - 1, 6, dtype=int):
        print(f"{contexts[index].query_family:22} {actual[index]:9.2f} "
              f"{predicted[index]:12.2f} {errors[index]:11.2f}")
    print("\nLargest prediction misses")
    for index in np.argsort(np.abs(errors))[-3:][::-1]:
        print(f"query family: {contexts[index].query_family}; actual: {actual[index]:.2f} ms; "
              f"predicted: {predicted[index]:.2f} ms; absolute error: {abs(errors[index]):.2f} ms")

    scenarios = {
        "Simple account summary": QueryContext("account_summary", "small", 1, 2, False,
            True, False, 35, 2, 85, 350),
        "Large transaction history": QueryContext("transaction_history", "large", 2, 4, True,
            False, False, 65, 10, 175, 650),
        "Query under connection pressure": QueryContext("statement_lookup", "small", 1, 3,
            False, False, False, 105, 34, 390, 900),
        "Complex query during high load": QueryContext("verification_audit", "very_large", 4,
            6, True, True, True, 120, 45, 480, 1050),
    }
    print("\nFictional pre-execution scenarios")
    for name, value in scenarios.items():
        estimate = predict_query_duration(model, [value])[0]
        warning = " — annotate request" if estimate > 1000 else ""
        print(f"{name}: {estimate:.2f} ms{warning}")
    print("\nPredictions are observability signals, not EXPLAIN output or query-optimizer actions.")


if __name__ == "__main__":
    main()
