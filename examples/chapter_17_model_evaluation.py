"""Evaluate Harbor's fictional integration-failure observability signal."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.integration_failure_model import (  # noqa: E402
    PREDICTION_FEATURES, build_integration_features, build_integration_targets,
    load_integration_requests, split_integration_dataset,
)
from harbor_ml.model_evaluation import (  # noqa: E402
    build_evaluation_report, collect_error_examples, select_threshold_for_minimum_recall,
)
from harbor_ml.training import TrainingConfig, train_integration_failure_model  # noqa: E402


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def build_output() -> str:
    """Build deterministic text from one held-out split and one probability vector."""
    dataset = ROOT / "data/harbor_integration_requests.csv"
    config = TrainingConfig()
    observations = load_integration_requests(dataset)
    features = build_integration_features(observations)
    targets = build_integration_targets(observations)
    split = split_integration_dataset(features, targets, test_size=config.test_size,
                                      random_state=config.random_state)
    trained = train_integration_failure_model(dataset, config)
    failure_index = list(trained.pipeline.classes_).index(1)
    probabilities = trained.pipeline.predict_proba(split.X_test)[:, failure_index]
    rows = [dict(zip(PREDICTION_FEATURES, row)) for row in split.X_test]
    report = build_evaluation_report(
        split.y_test,
        probabilities,
        {"vendor": [row["vendor"] for row in rows],
         "endpoint": [row["endpoint"] for row in rows]},
    )
    metrics = report.default_threshold.metrics
    errors = collect_error_examples(rows, split.y_test, probabilities)
    selected = select_threshold_for_minimum_recall(report.thresholds, minimum_recall=.70)

    lines = [
        "Harbor Federal Credit Union", "Model Evaluation Laboratory", "",
        "Target distribution", f"Test observations: {report.observations}",
        f"Success: {report.successes}", f"Failure: {report.failures}",
        f"Failure rate: {report.failures / report.observations:.3f}", "",
        f"Majority baseline accuracy: {report.baseline_accuracy:.3f}", "",
        "Confusion matrix", "                        PREDICTED",
        "                    success   failure",
        f"ACTUAL success       {metrics.true_negatives:7d}   {metrics.false_positives:7d}",
        f"ACTUAL failure       {metrics.false_negatives:7d}   {metrics.true_positives:7d}", "",
        "Model @ threshold 0.50", f"Accuracy: {metrics.accuracy:.3f}",
        f"Precision: {metrics.precision:.3f}", f"Recall: {metrics.recall:.3f}",
        f"Specificity: {metrics.specificity:.3f}", f"F1: {metrics.f1:.3f}",
        f"ROC-AUC: {report.roc_auc:.3f}",
        f"Average precision: {report.average_precision:.3f}", "",
        "Threshold Evaluation",
        "threshold predicted accuracy precision recall specificity F1    FP  FN",
    ]
    for item in report.thresholds:
        m = item.metrics
        lines.append(f"{item.threshold:9.2f} {item.predicted_failures:9d} {m.accuracy:8.3f} "
                     f"{m.precision:9.3f} {m.recall:6.3f} {m.specificity:11.3f} "
                     f"{m.f1:5.3f} {m.false_positives:3d} {m.false_negatives:3d}")
    lines.extend(["", "Example policy: highest precision with recall >= 0.70",
                  f"Selected threshold: {selected.threshold:.2f}",
                  f"Precision: {selected.metrics.precision:.3f}; recall: {selected.metrics.recall:.3f}",
                  "This is an example policy rule, not a universally correct threshold.", "",
                  "Largest-confidence errors"])
    for kind in ("false_positives", "false_negatives"):
        lines.append(kind.replace("_", " ").upper())
        examples = errors[kind]
        if not examples:
            lines.append("none")
        for example in examples:
            lines.append(" | ".join(f"{name}={example[name]}" for name in (
                "vendor", "endpoint", "recent_vendor_latency_ms", "recent_vendor_error_rate",
                "queue_depth", "retry_count", "probability", "actual")))
    for dimension, slices in report.slices.items():
        lines.extend(["", f"Slice metrics: {dimension}",
                      "value                    n failure accuracy precision recall"])
        for item in slices:
            lines.append(f"{item.value:22s} {item.count:3d} {item.failure_rate:7.3f} "
                         f"{item.accuracy:8.3f} {_number(item.precision):9s} {_number(item.recall):6s}")
    lines.extend(["", "Probability bins", "bin       count average predicted actual failure rate"])
    for item in report.probability_bins:
        lines.append(f"{item.lower:.1f}-{item.upper:.1f} {item.count:7d} "
                     f"{_number(item.average_probability):17s} {_number(item.actual_failure_rate)}")

    baseline_delta = metrics.accuracy - report.baseline_accuracy
    vendor_recalls = [item.recall for item in report.slices["vendor"] if item.recall is not None]
    nonempty_bins = [item for item in report.probability_bins if item.count]
    calibration_gap = max(abs(item.average_probability - item.actual_failure_rate)
                          for item in nonempty_bins
                          if item.average_probability is not None and item.actual_failure_rate is not None)
    lines.extend(["", "Engineering interpretation",
                  f"- Model accuracy differs from the actual majority baseline by {baseline_delta:+.3f}.",
                  f"- At threshold 0.50 it catches {metrics.recall:.1%} of held-out failures.",
                  f"- It generates {metrics.false_positives} false positives and misses {metrics.false_negatives} failures.",
                  "- The threshold table shows the recall/false-alarm tradeoff using the same probabilities.",
                  f"- Vendor recall spans {min(vendor_recalls):.3f} to {max(vendor_recalls):.3f}; counts provide essential context.",
                  f"- The largest probability-bin calibration gap is {calibration_gap:.3f}; raw probabilities should not be read too literally.",
                  "- This synthetic held-out evaluation cannot establish real production performance."])
    return "\n".join(lines) + "\n"


def main() -> None:
    print(build_output(), end="")


if __name__ == "__main__":
    main()
