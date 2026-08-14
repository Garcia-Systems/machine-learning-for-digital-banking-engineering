"""Run Chapter 29's reproducible incident-classifier laboratory."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.capstone_incident_classifier import (  # noqa: E402
    CAPSTONE_INCIDENT_CLASSES, CAPSTONE_INCIDENT_FEATURES,
    create_capstone_incident_metadata, evaluate_capstone_incident_classifier,
    load_capstone_classification_data, load_capstone_incident_artifact,
    load_chapter_26_timeline, predict_incident_probabilities,
    save_capstone_incident_artifact, score_capstone_incident_timeline,
    split_capstone_incident_data, train_capstone_incident_classifier,
)


def run(output_dir: Path) -> None:
    data = load_capstone_classification_data(ROOT / "data/harbor_incident_classes.csv")
    split = split_capstone_incident_data(data)
    model = train_capstone_incident_classifier(split)
    evaluation = evaluate_capstone_incident_classifier(model, split)
    metadata = create_capstone_incident_metadata(data, split, evaluation)
    print("Harbor Federal Credit Union\nCapstone Incident Classifier Training Laboratory")
    print("\nClasses (training observations):")
    for label in CAPSTONE_INCIDENT_CLASSES:
        print(f"{label:24} {metadata.training_class_counts[label]}")
    print("\nFeature contract:\n" + "\n".join(f"- {name}" for name in CAPSTONE_INCIDENT_FEATURES))
    print(f"\nTraining observations: {len(split.y_train)}\nTest observations: {len(split.y_test)}")
    print(f"\nMajority baseline accuracy: {evaluation.baseline_accuracy:.3f}")
    print(f"Model accuracy: {evaluation.accuracy:.3f}\nMacro F1: {evaluation.macro_f1:.3f}")
    print("\nConfusion matrix (rows=actual, columns=predicted)")
    print(" " * 25 + " ".join(f"{label[:5]:>5}" for label in CAPSTONE_INCIDENT_CLASSES))
    for label, values in zip(CAPSTONE_INCIDENT_CLASSES, evaluation.confusion_matrix, strict=True):
        print(f"{label:24} " + " ".join(f"{value:5d}" for value in values))
    print("\nClass                    precision recall    F1 support")
    for label in CAPSTONE_INCIDENT_CLASSES:
        metric = evaluation.per_class[label]
        print(f"{label:24} {metric.precision:9.3f} {metric.recall:6.3f} "
              f"{metric.f1:5.3f} {metric.support:7d}")
    print("\nCapstone incident timeline")
    print("time  top class                probability gap   ambiguous editorial phase")
    timeline = score_capstone_incident_timeline(model,
        load_chapter_26_timeline(ROOT / "data/harbor_capstone_incident.csv"))
    for item in timeline:
        p = item.prediction
        print(f"{item.timestamp:%H:%M} {p.predicted_class:24} {p.top_probability:11.3f} "
              f"{p.probability_gap:5.3f} {'yes' if p.ambiguous else 'no ':9} "
              f"{item.editorial_class}/{item.editorial_phase}")
    errors = np.flatnonzero(evaluation.predictions != split.y_test)
    print("\nHeld-out error analysis")
    if len(errors) == 0:
        print("No held-out errors occurred; do not infer production perfection from synthetic data.")
    for index in errors[:3]:
        prediction = predict_incident_probabilities(model, split.X_test[index])
        values = ", ".join(f"{name}={split.X_test[index, column]:.3g}"
                           for column, name in enumerate(CAPSTONE_INCIDENT_FEATURES))
        print(f"actual={split.y_test[index]} favored={prediction.predicted_class} "
              f"p={prediction.top_probability:.3f}\n  {values}")
    model_path, metadata_path = save_capstone_incident_artifact(model, metadata, output_dir)
    reloaded = load_capstone_incident_artifact(model_path)
    before = model.predict_proba(split.X_test[:5])
    after = reloaded.predict_proba(split.X_test[:5])
    passed = np.allclose(before, after)
    print(f"\nArtifact: {model_path}\nMetadata: {metadata_path}")
    print(f"Round-trip probability verification: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise RuntimeError("artifact round-trip changed class probabilities")


if __name__ == "__main__":
    run(ROOT / "artifacts/capstone-incident-classifier")
