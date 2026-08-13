"""Run Chapter 6's deterministic, text-based feature analysis laboratory."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from harbor_ml import (  # noqa: E402
    ANALYSIS_FEATURES,
    INCIDENT_CLASSES,
    INCIDENT_FEATURES,
    add_synthetic_noise,
    build_incident_classifier,
    build_incident_features,
    build_incident_targets,
    calculate_correlations,
    extract_model_coefficients,
    load_incident_dataset,
    run_ablation,
    split_incident_dataset,
    summarize_features_by_class,
)


def main() -> None:
    observations = load_incident_dataset(
        REPOSITORY_ROOT / "data/harbor_incident_classes.csv"
    )
    base = build_incident_features(observations)
    X = add_synthetic_noise(base)
    y = build_incident_targets(observations)

    print("Harbor Federal Credit Union")
    print("Feature Analysis Laboratory\n\nFeatures:")
    for feature in ANALYSIS_FEATURES:
        print(f"- {feature}")

    print("\nFeature summaries by incident type")
    summaries = summarize_features_by_class(observations)
    for feature in INCIDENT_FEATURES:
        print(f"\nFeature: {feature}")
        for item in (summary for summary in summaries if summary.feature == feature):
            print(
                f"  {item.incident_type:<24} mean={item.mean:8.2f} "
                f"min={item.minimum:8.2f} max={item.maximum:8.2f}"
            )

    correlations = calculate_correlations(X)
    short = ("api_lat", "err_rate", "db_conn", "queue", "vendor_lat", "rpm", "noise")
    print("\nSelected correlations (Pearson)")
    print(f"{'':>12}" + "".join(f"{name:>11}" for name in short))
    for name, row in zip(short, correlations, strict=True):
        print(f"{name:>12}" + "".join(f"{value:>11.2f}" for value in row))

    split = split_incident_dataset(X, y)
    model = build_incident_classifier().fit(split.X_train, split.y_train)
    coefficients = extract_model_coefficients(model, ANALYSIS_FEATURES)
    print("\nModel coefficients (standardized inputs)")
    for label in INCIDENT_CLASSES:
        print(f"\nClass: {label}")
        for item in (
            coefficient
            for coefficient in coefficients
            if coefficient.incident_type == label
        ):
            print(f"  {item.feature:<24}{item.coefficient:+.3f}")

    results = run_ablation(observations)
    print("\nAblation study")
    for result in results:
        print(f"{result.name:<30} accuracy: {result.accuracy:.3f}")
    baseline = results[0]
    largest_drop = max(
        results[1:], key=lambda result: baseline.accuracy - result.accuracy
    )
    noise_result = next(
        result for result in results if result.name == "without_synthetic_noise"
    )
    print("\nEngineering interpretation:")
    print(
        f"- On this split, {largest_drop.name.removeprefix('without_')} had the largest "
        "accuracy loss when removed."
    )
    relation = (
        "higher" if noise_result.accuracy > baseline.accuracy else "lower or equal"
    )
    print(
        f"- Accuracy without synthetic_noise was {relation} than the all-feature result."
    )
    print("- These fixture-specific associations do not prove incident causes.")
    print(
        f"- One test mistake changes accuracy by {100 / baseline.test_observations:.2f} points."
    )


if __name__ == "__main__":
    main()
