"""Run Harbor's deterministic, internal model-explainability laboratory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.explainability import (  # noqa: E402
    calculate_permutation_importance, compare_feature_sensitivity,
    explain_linear_prediction, extract_logistic_coefficients, sorted_contributions,
)
from harbor_ml.integration_failure_model import (  # noqa: E402
    IntegrationRequest, build_integration_features, build_integration_targets,
    load_integration_requests, split_integration_dataset,
)
from harbor_ml.training import TrainingConfig, train_integration_failure_model  # noqa: E402


def build_output() -> str:
    dataset = ROOT / "data/harbor_integration_requests.csv"
    config = TrainingConfig()
    observations = load_integration_requests(dataset)
    split = split_integration_dataset(
        build_integration_features(observations), build_integration_targets(observations),
        test_size=config.test_size, random_state=config.random_state,
    )
    trained = train_integration_failure_model(dataset, config)
    lines = ["Harbor Federal Credit Union", "Model Explainability Laboratory", "",
             f"Model: {trained.metadata.model_name}",
             f"Version: {trained.metadata.model_version}", "", "Global coefficient summary"]
    coefficients = sorted(extract_logistic_coefficients(trained.pipeline),
                          key=lambda item: abs(item.coefficient), reverse=True)
    lines.extend(f"{item.feature_name:40s} {item.coefficient:+.4f}" for item in coefficients)
    lines.extend(["", "Permutation importance (held-out ROC-AUC)"])
    importance = calculate_permutation_importance(
        trained.pipeline, split.X_test, split.y_test, random_state=config.random_state
    )
    lines.extend(f"{item.feature_name:40s} {item.importance_mean:+.4f} ± {item.importance_std:.4f}"
                 for item in importance)

    scenarios = {
        "degraded_vendor_request": IntegrationRequest(
            "ClearVerify", "identity_verify", 1600.0, .32, 85, 2, 4200, 14),
        "queue_pressure_request": IntegrationRequest(
            "Northstar", "identity_verify", 280.0, .02, 190, 0, 4200, 14),
    }
    for name, request in scenarios.items():
        explanation = explain_linear_prediction(
            trained.pipeline, request, model_name=trained.metadata.model_name,
            model_version=trained.metadata.model_version,
        )
        expected = trained.pipeline.predict_proba(
            [[getattr(request, feature) for feature in trained.metadata.numerical_features + trained.metadata.categorical_features]]
        )[0, list(trained.pipeline.classes_).index(1)]
        if abs(explanation.probability - expected) > 1e-12:
            raise AssertionError("contributions did not reproduce predict_proba")
        lines.extend(["", f"Scenario: {name}",
                      f"Failure probability: {explanation.probability:.3f}",
                      "Top contributions toward failure:"])
        lines.extend(f"+ {item.feature_name}: {item.contribution:+.4f}"
                     for item in sorted_contributions(explanation, positive=True, limit=4))
        lines.append("Top contributions away from failure:")
        lines.extend(f"- {item.feature_name}: {item.contribution:+.4f}"
                     for item in sorted_contributions(explanation, positive=False, limit=4))
        lines.extend(["Interpretation: The fitted model relied on the transformed values above.",
                      "This does not establish the request's actual root cause."])

    lines.extend(["", "Model sensitivity experiment (not a causal effect)"])
    for result in compare_feature_sensitivity(
        trained.pipeline, scenarios["degraded_vendor_request"],
        "recent_vendor_latency_ms", (300.0, 900.0, 1600.0),
    ):
        lines.append(f"recent_vendor_latency_ms={result.value:.0f}: probability={result.probability:.3f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(build_output(), end="")
