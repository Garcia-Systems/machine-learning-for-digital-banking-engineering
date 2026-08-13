import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from harbor_ml.integration_failure_model import (
    PREDICTION_FEATURES, build_integration_features, build_integration_targets,
    load_integration_requests, split_integration_dataset,
)
from harbor_ml.model_evaluation import (
    apply_threshold, build_evaluation_report, build_probability_bins,
    calculate_classification_metrics, calculate_majority_baseline,
    calculate_ranking_metrics, collect_error_examples, evaluate_slices,
    evaluate_threshold, evaluate_thresholds, select_threshold_for_minimum_recall,
)
from harbor_ml.training import TrainingConfig, train_integration_failure_model

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data/harbor_integration_requests.csv"


def test_confusion_matrix_and_metric_formulas_use_predictions():
    metrics = calculate_classification_metrics(
        [1] * 40 + [0] * 170,
        [1] * 30 + [0] * 10 + [1] * 20 + [0] * 150,
    )
    assert (metrics.true_positives, metrics.true_negatives,
            metrics.false_positives, metrics.false_negatives) == (30, 150, 20, 10)
    assert metrics.observations == 210
    assert metrics.accuracy == pytest.approx(180 / 210)
    assert metrics.precision == pytest.approx(30 / 50)
    assert metrics.recall == pytest.approx(30 / 40)
    assert metrics.specificity == pytest.approx(150 / 170)
    assert metrics.f1 == pytest.approx(2 * .6 * .75 / (.6 + .75))


def test_zero_division_is_explicit_and_baseline_uses_actual_distribution():
    metrics = calculate_classification_metrics([0, 0, 0], [0, 0, 0])
    assert metrics.precision == metrics.recall == metrics.f1 == 0
    assert metrics.specificity == metrics.accuracy == 1
    baseline = calculate_majority_baseline([0] * 8 + [1] * 2)
    assert baseline.accuracy == .8
    assert baseline.true_negatives == 8


def test_threshold_application_sweep_order_and_selection():
    actual = [0, 1, 1, 0]
    probabilities = [.1, .45, .8, .7]
    assert apply_threshold(probabilities, .5).tolist() == [0, 0, 1, 1]
    result = evaluate_threshold(actual, probabilities, .5)
    assert result.predicted_failures == 2
    assert result.metrics.true_positives == 1
    sweep = evaluate_thresholds(actual, probabilities, [.8, .2, .5])
    assert [item.threshold for item in sweep] == [.2, .5, .8]
    selected = select_threshold_for_minimum_recall(sweep, .5)
    assert selected.threshold == .8
    with pytest.raises(ValueError, match="no evaluated"):
        select_threshold_for_minimum_recall([evaluate_threshold(actual, probabilities, .9)], .5)


def test_roc_auc_uses_probability_ranking():
    auc, average_precision = calculate_ranking_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
    assert auc == pytest.approx(.75)
    assert average_precision == pytest.approx(5 / 6)
    # At .5 these two score vectors classify identically but rank differently.
    worse_auc, _ = calculate_ranking_metrics([0, 0, 1, 1], [.49, .48, .2, .8])
    assert worse_auc != auc


def test_probability_bins_cover_every_row_and_include_one():
    bins = build_probability_bins([0, 0, 1, 1, 1], [0, .2, .4, .8, 1])
    assert sum(item.count for item in bins) == 5
    assert [item.count for item in bins] == [1, 1, 1, 0, 2]
    assert bins[-1].actual_failure_rate == 1
    assert bins[3].average_probability is None


def test_confident_error_examples_are_real_errors_and_ordered():
    rows = [{"id": value} for value in "abcde"]
    errors = collect_error_examples(rows, [0, 0, 1, 1, 0], [.9, .7, .1, .4, .2], limit=2)
    assert [item["id"] for item in errors["false_positives"]] == ["a", "b"]
    assert [item["id"] for item in errors["false_negatives"]] == ["c", "d"]
    assert all(item["actual"] == "success" for item in errors["false_positives"])
    assert all(item["actual"] == "failure" for item in errors["false_negatives"])


def test_slice_grouping_preserves_counts_and_undefined_ratios():
    slices = evaluate_slices(["B", "A", "A", "B"], [0, 0, 1, 0], [.1, .2, .3, .9])
    assert [item.value for item in slices] == ["A", "B"]
    assert sum(item.count for item in slices) == 4
    assert slices[0].precision is None
    assert slices[1].recall is None


@pytest.fixture(scope="module")
def harbor_evaluation():
    observations = load_integration_requests(DATASET)
    X = build_integration_features(observations)
    y = build_integration_targets(observations)
    config = TrainingConfig()
    split = split_integration_dataset(X, y, test_size=config.test_size,
                                      random_state=config.random_state)
    trained = train_integration_failure_model(DATASET, config)
    probabilities = trained.pipeline.predict_proba(split.X_test)[:, list(trained.pipeline.classes_).index(1)]
    rows = [dict(zip(PREDICTION_FEATURES, row)) for row in split.X_test]
    report = build_evaluation_report(split.y_test, probabilities, {
        "vendor": [row["vendor"] for row in rows],
        "endpoint": [row["endpoint"] for row in rows],
    })
    return split, probabilities, rows, report


def test_report_is_computed_from_held_out_predictions_and_serializable(harbor_evaluation):
    split, probabilities, _, report = harbor_evaluation
    expected = calculate_classification_metrics(split.y_test, apply_threshold(probabilities, .5))
    assert report.default_threshold.metrics == expected
    assert expected.observations == len(split.y_test) == report.observations
    assert report.baseline_accuracy == calculate_majority_baseline(split.y_test).accuracy
    assert sum(item.count for item in report.probability_bins) == len(split.y_test)
    assert all(sum(item.count for item in slices) == len(split.y_test)
               for slices in report.slices.values())
    assert json.loads(report.to_json())["observations"] == len(split.y_test)


def test_chapter_17_laboratory_output_is_deterministic():
    command = [sys.executable, str(ROOT / "examples/chapter_17_model_evaluation.py")]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    assert first == second
    assert "Test observations: 150" in first
    assert "Engineering interpretation" in first
