"""Run Chapter 21's privacy-minimized data-contract laboratory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.data_security import (  # noqa: E402
    APPROVED_INTEGRATION_FEATURES,
    SensitiveFieldError,
    build_safe_log_context,
    calculate_file_sha256,
    validate_dataset_columns,
    validate_prediction_payload_fields,
)


SAFE_PAYLOAD = {
    "vendor": "ClearVerify",
    "endpoint": "identity_verify",
    "recent_vendor_latency_ms": 1200.0,
    "recent_vendor_error_rate": 0.04,
    "queue_depth": 63,
    "retry_count": 2,
    "request_size_bytes": 2200,
    "hour_of_day": 14,
}


def main() -> None:
    print("Harbor Federal Credit Union\nML Data Security Laboratory")
    print("\nApproved prediction fields:")
    print("\n".join(f"- {name}" for name in APPROVED_INTEGRATION_FEATURES))
    print("\nValidating safe payload...")
    validate_prediction_payload_fields(SAFE_PAYLOAD)
    print("OK")

    unsafe = {**SAFE_PAYLOAD, "access_token": "fictional-do-not-use", "account_number": "000"}
    print("\nValidating unsafe payload...")
    try:
        validate_prediction_payload_fields(unsafe)
    except SensitiveFieldError as error:
        print(f"Rejected:\n{error}")

    context = build_safe_log_context(unsafe, model_version="harbor-integration-failure-demo")
    print("\nSafe log context:")
    print(json.dumps(context, indent=2))

    dataset = ROOT / "data/harbor_integration_requests.csv"
    with dataset.open(newline="", encoding="utf-8") as source:
        validate_dataset_columns(next(csv.reader(source)))
    print(f"\nDataset schema:\n{dataset.name}: OK")

    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "fictional-model.joblib"
        artifact.write_bytes(b"Harbor fictional teaching artifact")
        fingerprint = calculate_file_sha256(artifact)
    print(f"\nArtifact SHA-256:\n{fingerprint}")

    print("\nLeast-privilege component summary:")
    print("- training job: dataset + artifact write directory")
    print("- prediction service: trusted artifact + narrow request")
    print("- dashboard: summarized technical telemetry")
    print("\nNo real member data or credentials were used.")


if __name__ == "__main__":
    main()
