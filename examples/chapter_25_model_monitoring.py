"""Executable Chapter 25 laboratory: monitor a deployed model without auto-retraining."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.integration_failure_model import (  # noqa: E402
    NUMERIC_FEATURES, build_integration_features, build_integration_targets,
    load_integration_requests,
)
from harbor_ml.model_monitoring import (  # noqa: E402
    PredictionRecord, ServiceHealth, attach_labels, build_training_monitoring_baseline,
    calculate_categorical_drift, calculate_numeric_drift, calculate_prediction_drift,
    compare_performance_to_baseline, compare_shadow_predictions, latency_percentile, simulate_production_periods,
    summarize_prediction_window,
)
from harbor_ml.training import TrainingConfig, train_integration_failure_model  # noqa: E402


DATA = ROOT / "data" / "harbor_integration_requests.csv"


def probabilities(model, rows) -> list[float]:
    matrix = build_integration_features(rows)
    positive = list(model.classes_).index(1)
    return [float(value) for value in model.predict_proba(matrix)[:, positive]]


def main() -> None:
    config = TrainingConfig()
    trained = train_integration_failure_model(
        DATA, config, now=lambda: datetime(2025, 1, 15, tzinfo=timezone.utc))
    observations = load_integration_requests(DATA)
    # Use the identical deterministic split as Chapter 16: the reference describes
    # only the rows on which this artifact was fitted, never synthetic production.
    train_rows, _ = train_test_split(
        observations, test_size=config.test_size, random_state=config.random_state,
        stratify=build_integration_targets(observations))
    baseline = build_training_monitoring_baseline(
        train_rows, model_name=trained.metadata.model_name,
        model_version=trained.metadata.model_version, dataset_sha256=trained.dataset_sha256,
        created_at=trained.metadata.trained_at)
    baseline_scores = probabilities(trained.pipeline, train_rows)

    print("Harbor Federal Credit Union\nProduction Model Monitoring Laboratory")
    print(f"\nModel: {baseline.model_name}\nVersion: {baseline.model_version}")
    print(f"Training baseline rows: {len(train_rows)}; dataset SHA-256: {baseline.dataset_sha256[:12]}…")

    for period_name, rows in simulate_production_periods().items():
        scores = probabilities(trained.pipeline, rows)
        records = tuple(PredictionRecord(
            f"{period_name}-{index:03}", row.timestamp, baseline.model_name,
            baseline.model_version, "review-policy-v2", row.vendor, row.endpoint,
            score, score >= config.classification_threshold)
            for index, (row, score) in enumerate(zip(rows, scores)))
        # Labels arrive later as separate facts; the immutable prediction records survive.
        labeled = attach_labels(records, {record.prediction_id: bool(row.request_failed)
                                          for record, row in zip(records, rows)})
        numeric = [calculate_numeric_drift(
            feature, baseline.numeric_summaries[feature],
            [float(getattr(row, feature)) for row in rows]) for feature in NUMERIC_FEATURES]
        category = (*calculate_categorical_drift(
            "vendor", baseline.categorical_frequencies["vendor"], [row.vendor for row in rows]),
                    *calculate_categorical_drift(
            "endpoint", baseline.categorical_frequencies["endpoint"],
            [row.endpoint for row in rows]))
        output_drift = calculate_prediction_drift(baseline_scores, scores)
        drift = any(item.investigate for item in numeric) or any(
            item.investigate for item in category) or output_drift.investigate
        preliminary = summarize_prediction_window(labeled, baseline, minimum_labeled=20)
        if preliminary.performance_metrics:
            performance = compare_performance_to_baseline(
                "recall", trained.metrics.recall, preliminary.performance_metrics.recall)
            # Performance evidence is strongest in the deliberately degraded period;
            # other periods still print the comparison without forcing every signal.
            drift = drift or (period_name == "D" and performance.investigate)
        summary = summarize_prediction_window(labeled, baseline, minimum_labeled=20,
                                              drift_detected=drift)
        # Deterministic service simulation; these numbers are operational observations.
        errors = 3 if period_name == "D" else 0
        latencies = tuple(12 + ((index * 7 + ord(period_name)) % 19) for index in range(40))
        service = ServiceHealth(40 + errors, 40, errors, latencies, True)
        latency_shift = next(item for item in numeric
                             if item.feature == "recent_vendor_latency_ms")

        print(f"\nPeriod {period_name} — " + {
            "A": "baseline-like traffic", "B": "vendor shift",
            "C": "new endpoint", "D": "performance degradation"}[period_name])
        print(f"Predictions: {summary.prediction_count}; service errors: {service.prediction_error_total}")
        print(f"p50/p95 latency: {latency_percentile(latencies, 50):.0f}/"
              f"{latency_percentile(latencies, 95):.0f} ms")
        print(f"Unknown categories: {summary.unknown_category_count}")
        print(f"Average probability: {summary.average_probability:.3f}; "
              f"predicted failure rate: {summary.predicted_positive_rate:.1%}")
        print(f"Vendor latency standardized mean shift: "
              f"{latency_shift.standardized_mean_shift:+.2f}")
        print(f"Prediction mean change: {output_drift.average_probability_difference:+.3f}")
        if summary.performance_metrics:
            metrics = summary.performance_metrics
            print(f"Labeled performance ({summary.labeled_count}): accuracy={metrics.accuracy:.3f}, "
                  f"precision={metrics.precision:.3f}, recall={metrics.recall:.3f}, F1={metrics.f1:.3f}, "
                  f"FPR={1 - metrics.specificity:.3f}, FNR={1 - metrics.recall:.3f}")
        print(f"Status: {summary.status}")

    # Same observations, two score streams: the candidate has no production effect.
    production = probabilities(trained.pipeline, simulate_production_periods()["B"])
    candidate = np.clip(np.asarray(production) * .92 + .03, 0, 1).tolist()
    shadow = compare_shadow_predictions(production, candidate)
    print("\nShadow comparison (same Period B observations)")
    print(f"Average absolute probability difference: "
          f"{shadow.average_absolute_probability_difference:.3f}")
    print(f"Threshold disagreement rate: {shadow.disagreement_rate:.1%}")
    print("Candidate disagreement is evidence to investigate, not proof it is better.")
    print("No retraining or deployment was performed.")


if __name__ == "__main__":
    main()
