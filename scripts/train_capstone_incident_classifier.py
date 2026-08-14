"""Train and persist Harbor's Chapter 29 capstone incident classifier."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.capstone_incident_classifier import (  # noqa: E402
    CAPSTONE_INCIDENT_CLASSES, create_capstone_incident_metadata,
    evaluate_capstone_incident_classifier, load_capstone_classification_data,
    load_chapter_26_timeline, save_capstone_incident_artifact,
    score_capstone_incident_timeline, split_capstone_incident_data,
    train_capstone_incident_classifier,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/harbor_incident_classes.csv")
    parser.add_argument("--timeline", type=Path, default=ROOT / "data/harbor_capstone_incident.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/capstone-incident-classifier")
    parser.add_argument("--ambiguity-gap", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_capstone_classification_data(args.dataset)
    split = split_capstone_incident_data(data, random_state=args.random_state)
    model = train_capstone_incident_classifier(split, random_state=args.random_state)
    evaluation = evaluate_capstone_incident_classifier(model, split)
    timeline = score_capstone_incident_timeline(model, load_chapter_26_timeline(args.timeline),
                                                ambiguity_gap=args.ambiguity_gap)
    metadata = create_capstone_incident_metadata(data, split, evaluation,
        ambiguity_gap=args.ambiguity_gap, random_state=args.random_state)
    model_path, metadata_path = save_capstone_incident_artifact(model, metadata, args.output_dir)
    print("Harbor Federal Credit Union\nCapstone Incident Classifier Training")
    for label in CAPSTONE_INCIDENT_CLASSES:
        print(f"{label:24} {metadata.training_class_counts[label]}")
    print(f"Training observations: {metadata.train_rows}\nTest observations: {metadata.test_rows}")
    print(f"Majority baseline accuracy: {evaluation.baseline_accuracy:.3f}")
    print(f"Model accuracy: {evaluation.accuracy:.3f}\nMacro F1: {evaluation.macro_f1:.3f}")
    print("\nCapstone timeline")
    for result in timeline:
        prediction = result.prediction
        print(f"{result.timestamp:%H:%M} {prediction.predicted_class:24} "
              f"p={prediction.top_probability:.3f} gap={prediction.probability_gap:.3f} "
              f"ambiguous={'yes' if prediction.ambiguous else 'no'} "
              f"editorial={result.editorial_class}")
    print(f"\nModel: {model_path}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
