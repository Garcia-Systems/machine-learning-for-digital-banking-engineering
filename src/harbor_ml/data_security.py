"""Small, explicit data-security contracts for Harbor's teaching model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping


APPROVED_INTEGRATION_FEATURES = (
    "vendor",
    "endpoint",
    "recent_vendor_latency_ms",
    "recent_vendor_error_rate",
    "queue_depth",
    "retry_count",
    "request_size_bytes",
    "hour_of_day",
)
INTEGRATION_DATASET_COLUMNS = (
    "timestamp",
    *APPROVED_INTEGRATION_FEATURES,
    "request_failed",
)
PROHIBITED_SENSITIVE_FIELDS = frozenset(
    {
        "ssn",
        "social_security_number",
        "account_number",
        "routing_number",
        "card_number",
        "cvv",
        "password",
        "access_token",
        "refresh_token",
        "api_key",
        "private_key",
        "authentication_cookie",
        "full_name",
        "member_name",
        "member_email",
        "email_address",
    }
)


class SensitiveFieldError(ValueError):
    """A declared sensitive field crossed a teaching contract boundary."""


class UnexpectedFieldError(ValueError):
    """A field was not deliberately approved for a strict contract."""


def find_prohibited_fields(fields: Iterable[str]) -> tuple[str, ...]:
    """Find exact prohibited names; this is intentionally not content inspection."""
    return tuple(sorted(set(fields).intersection(PROHIBITED_SENSITIVE_FIELDS)))


def validate_feature_contract(features: Iterable[str]) -> None:
    """Require the integration model's exact, ordered feature contract."""
    actual = tuple(features)
    prohibited = find_prohibited_fields(actual)
    if prohibited:
        raise SensitiveFieldError("prohibited field: " + ", ".join(prohibited))
    if actual != APPROVED_INTEGRATION_FEATURES:
        raise UnexpectedFieldError("integration feature contract does not match the allowlist")


def _validate_fields(fields: Iterable[str], allowed: Iterable[str], *, label: str) -> None:
    actual = set(fields)
    prohibited = find_prohibited_fields(actual)
    if prohibited:
        raise SensitiveFieldError("prohibited field: " + ", ".join(prohibited))
    approved = set(allowed)
    missing = sorted(approved - actual)
    unexpected = sorted(actual - approved)
    if missing:
        raise UnexpectedFieldError(f"Missing {label} field: " + ", ".join(missing))
    if unexpected:
        raise UnexpectedFieldError(f"Unexpected {label} field: " + ", ".join(unexpected))


def validate_prediction_payload_fields(payload: Mapping[str, object]) -> None:
    """Fail closed unless a prediction payload has exactly the approved keys."""
    _validate_fields(payload, APPROVED_INTEGRATION_FEATURES, label="prediction")


def build_integration_feature_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate, then construct a new object containing only approved features."""
    validate_prediction_payload_fields(payload)
    return {name: payload[name] for name in APPROVED_INTEGRATION_FEATURES}


def validate_dataset_columns(columns: Iterable[str]) -> None:
    """Fail closed on integration-training schema drift."""
    _validate_fields(columns, INTEGRATION_DATASET_COLUMNS, label="dataset")


def build_safe_log_context(
    payload: Mapping[str, object], *, model_version: str
) -> dict[str, object]:
    """Build log context explicitly; never copy the supplied mapping wholesale."""
    return {
        "vendor": payload.get("vendor"),
        "endpoint": payload.get("endpoint"),
        "model_version": model_version,
    }


def calculate_file_sha256(path: str | Path) -> str:
    """Return a deterministic fingerprint of bytes, not a provenance guarantee."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
