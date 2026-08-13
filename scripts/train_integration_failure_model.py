"""Train and write Harbor's trusted local integration-failure artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.training import (  # noqa: E402
    TrainingConfig, save_model_artifact, save_training_metadata,
    train_integration_failure_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/harbor_integration_requests.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/integration-failure")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        random_state=args.random_state, classification_threshold=args.threshold
    )
    print("Harbor Federal Credit Union\nIntegration Failure Model Training")
    print(f"\nDataset:\n{args.data}\n\nValidating, fingerprinting, and training...")
    result = train_integration_failure_model(args.data, config)
    model_path = save_model_artifact(result.pipeline, args.output_dir)
    metadata_path = save_training_metadata(result.metadata, args.output_dir)
    print(f"Model trained.\n\nDataset SHA-256:\n{result.dataset_sha256}")
    print(f"\nRows: {result.metadata.dataset_rows}\nTraining rows: {result.training_rows}"
          f"\nTest rows: {result.test_rows}")
    print("\nFeatures:\n" + "\n".join(f"- {x}" for x in (
        *result.metadata.numerical_features, *result.metadata.categorical_features)))
    print("\nEvaluation:")
    for name, value in result.metadata.metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"\nWriting artifact:\n{model_path}\n\nWriting metadata:\n{metadata_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
