import csv
from pathlib import Path

import pytest

from harbor_ml.data_security import (
    APPROVED_INTEGRATION_FEATURES,
    INTEGRATION_DATASET_COLUMNS,
    PROHIBITED_SENSITIVE_FIELDS,
    SensitiveFieldError,
    UnexpectedFieldError,
    build_integration_feature_payload,
    build_safe_log_context,
    calculate_file_sha256,
    find_prohibited_fields,
    validate_dataset_columns,
    validate_feature_contract,
    validate_prediction_payload_fields,
)

ROOT = Path(__file__).resolve().parents[1]
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


def test_exact_feature_contract_excludes_target_identifiers_and_prohibited_fields():
    assert APPROVED_INTEGRATION_FEATURES == (
        "vendor", "endpoint", "recent_vendor_latency_ms", "recent_vendor_error_rate",
        "queue_depth", "retry_count", "request_size_bytes", "hour_of_day",
    )
    validate_feature_contract(APPROVED_INTEGRATION_FEATURES)
    assert "request_failed" not in APPROVED_INTEGRATION_FEATURES
    assert not {"request_id", "trace_id", "session_id"} & set(APPROVED_INTEGRATION_FEATURES)
    assert not PROHIBITED_SENSITIVE_FIELDS & set(APPROVED_INTEGRATION_FEATURES)


def test_prohibited_fields_are_detected_and_rejected():
    assert find_prohibited_fields(["vendor", "access_token", "account_number"]) == (
        "access_token", "account_number"
    )
    for field in ("account_number", "access_token"):
        with pytest.raises(SensitiveFieldError, match=field):
            validate_prediction_payload_fields({**SAFE_PAYLOAD, field: "fictional"})


def test_prediction_payload_is_strict_and_feature_builder_is_allowlist_first():
    validate_prediction_payload_fields(SAFE_PAYLOAD)
    assert build_integration_feature_payload(SAFE_PAYLOAD) == SAFE_PAYLOAD
    with pytest.raises(UnexpectedFieldError, match="Unexpected prediction field"):
        validate_prediction_payload_fields({**SAFE_PAYLOAD, "debug_note": "not approved"})


def test_safe_log_context_copies_only_deliberately_safe_fields():
    context = build_safe_log_context(
        {**SAFE_PAYLOAD, "member_email": "fictional@example.invalid"},
        model_version="harbor-integration-failure-demo",
    )
    assert context == {
        "vendor": "ClearVerify", "endpoint": "identity_verify",
        "model_version": "harbor-integration-failure-demo",
    }
    assert "member_email" not in context


def test_dataset_contract_accepts_known_schema_and_rejects_sensitive_drift():
    validate_dataset_columns(INTEGRATION_DATASET_COLUMNS)
    with pytest.raises(SensitiveFieldError, match="member_email"):
        validate_dataset_columns((*INTEGRATION_DATASET_COLUMNS, "member_email"))


def test_hash_is_real_deterministic_and_changes_with_bytes(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"fictional artifact bytes")
    first = calculate_file_sha256(artifact)
    assert first == calculate_file_sha256(artifact)
    assert len(first) == 64
    artifact.write_bytes(b"different fictional artifact bytes")
    assert calculate_file_sha256(artifact) != first


def test_committed_csv_headers_have_no_known_prohibited_direct_fields():
    for path in sorted((ROOT / "data").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as source:
            header = next(csv.reader(source))
        assert find_prohibited_fields(header) == (), path
