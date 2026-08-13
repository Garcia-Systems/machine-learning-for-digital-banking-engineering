"""Run Chapter 14's deterministic system-demand forecasting laboratory."""

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.demand_forecasting import (  # noqa: E402
    DEMAND_FEATURES, FORECAST_HORIZON_MINUTES, build_demand_examples,
    build_demand_model, build_demand_targets, calculate_metrics, chronological_split,
    load_demand_observations, persistence_predictions, predict_future_demand,
    residuals, train_demand_model,
)


def main() -> None:
    observations = load_demand_observations(ROOT / "data/harbor_system_demand.csv")
    examples = build_demand_examples(observations)
    split = chronological_split(examples)
    model = train_demand_model(build_demand_model(), split.train)
    actual = build_demand_targets(split.test)
    baseline = persistence_predictions(split.test)
    predictions = predict_future_demand(model, split.test)
    baseline_metrics = calculate_metrics(actual, baseline)
    model_metrics = calculate_metrics(actual, predictions)

    print("Harbor Federal Credit Union\nSystem Demand Forecasting Laboratory")
    print(f"\nForecast horizon: {FORECAST_HORIZON_MINUTES} minutes")
    print(f"Raw observations: {len(observations)}\nSupervised examples: {len(examples)}")
    print(f"\nTraining period: {split.train[0].timestamp.isoformat()} to "
          f"{split.train[-1].timestamp.isoformat()}")
    print(f"Test period: {split.test[0].timestamp.isoformat()} to "
          f"{split.test[-1].timestamp.isoformat()}")
    print("\nFeatures:")
    for feature in DEMAND_FEATURES:
        print(f"- {feature}")
    print(f"\nPersistence baseline\nMAE: {baseline_metrics.mae:.2f}\nRMSE: {baseline_metrics.rmse:.2f}")
    print(f"\nRegression model\nMAE: {model_metrics.mae:.2f}\nRMSE: {model_metrics.rmse:.2f}")
    print("\nSample forecasts\ntimestamp                  actual  predicted  residual")
    sample_residuals = residuals(actual, predictions)
    for index in range(0, min(5 * 24, len(split.test)), 24):
        print(f"{split.test[index].timestamp.isoformat():25} {actual[index]:7.2f} "
              f"{predictions[index]:10.2f} {sample_residuals[index]:9.2f}")

    latest = split.test[-1]
    scenario = replace(latest, requests_now=820, requests_5m_ago=790,
        requests_10m_ago=755, requests_15m_ago=730,
        recent_average_requests=(820 + 790 + 755 + 730) / 4,
        recent_growth=90, api_latency_ms=310, error_rate=.008, queue_depth=28,
        hour_of_day=10, day_of_week=2)
    forecast = predict_future_demand(model, [scenario])[0]
    print("\nFictional current scenario:")
    print("requests now=820, 5m ago=790, 10m ago=755, 15m ago=730")
    print("api latency=310 ms, error rate=0.008, queue depth=28, hour=10, weekday=2")
    print(f"Predicted demand {FORECAST_HORIZON_MINUTES} minutes ahead: "
          f"{forecast:.2f} requests/min")
    print("\nThe fitted-model forecast is an estimate for capacity context, not scaling policy.")


if __name__ == "__main__":
    main()
