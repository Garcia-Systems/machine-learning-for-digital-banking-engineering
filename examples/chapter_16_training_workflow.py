"""Run Chapter 16's repeatable training and artifact round-trip laboratory."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.integration_failure_model import IntegrationRequest, predict_failure_probability  # noqa: E402
from harbor_ml.training import (  # noqa: E402
    calculate_file_sha256, load_trusted_model_artifact, save_model_artifact,
    save_training_metadata, train_integration_failure_model, validate_training_dataset,
)


def main() -> None:
    dataset = ROOT / "data/harbor_integration_requests.csv"
    print("Harbor Federal Credit Union\nRepeatable Training Workflow Laboratory")
    print("\nValidating dataset...")
    validate_training_dataset(dataset)
    print(f"OK\n\nDataset SHA-256:\n{calculate_file_sha256(dataset)}\n\nTraining...")
    result = train_integration_failure_model(dataset)
    print("Complete\n\nEvaluation:")
    for name, value in result.metadata.metrics.items():
        print(f"{name}: {value:.4f}")
    request = IntegrationRequest("ClearVerify", "identity_verify", 410, 0.08, 38, 1, 1800, 14)
    before = predict_failure_probability(result.pipeline, request)
    with tempfile.TemporaryDirectory(prefix="harbor-ch16-") as temporary:
        model_path = save_model_artifact(result.pipeline, temporary)
        metadata_path = save_training_metadata(result.metadata, temporary)
        print(f"\nArtifact written:\n{model_path}\n\nMetadata written:\n{metadata_path}")
        print("\nReloading trusted artifact...")
        loaded = load_trusted_model_artifact(model_path)
        after = predict_failure_probability(loaded, request)
        print("OK\n\nRound-trip prediction:")
        print(f"before save: {before:.8f}\nafter load: {after:.8f}\n\nMatch: {'yes' if before == after else 'no'}")
        if before != after:
            raise RuntimeError("artifact round-trip changed the prediction")
    print("\nTemporary laboratory artifacts removed.")


if __name__ == "__main__":
    main()
