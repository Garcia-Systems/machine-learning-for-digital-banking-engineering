"""Explicit, validated machine-learning problem definitions for Chapter 2."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

REQUEST_OUTCOME_FIELDS = (
    "timestamp",
    "vendor_latency_ms",
    "queue_depth",
    "db_connections",
    "retry_count",
    "request_failed",
)


class ProblemType(Enum):
    """The analytical output an ML problem asks a future model to produce."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"

    @property
    def is_supervised(self) -> bool:
        """Return whether examples for this type require known targets."""
        return self in {
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
            ProblemType.REGRESSION,
        }


@dataclass(frozen=True)
class MLProblem:
    """A measurable ML question, its prediction-time inputs, and its target."""

    name: str
    engineering_question: str
    problem_type: ProblemType
    features: tuple[str, ...]
    target: str | None

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("Invalid ML problem: at least one feature is required.")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Invalid ML problem: feature names must be unique.")
        if self.target is not None and self.target in self.features:
            raise ValueError(
                f"Invalid ML problem: target '{self.target}' cannot also appear "
                "in the feature set."
            )
        if self.problem_type.is_supervised and self.target is None:
            raise ValueError(
                "Invalid ML problem: supervised problems require a target."
            )
        if not self.problem_type.is_supervised and self.target is not None:
            raise ValueError(
                "Invalid ML problem: unsupervised problems must not define a target."
            )

    def validate(self) -> None:
        """Document that construction performs deterministic validation.

        A successfully constructed immutable instance is valid. This method gives
        examples an explicit validation step without duplicating the rules.
        """


@dataclass(frozen=True)
class RequestOutcome:
    """One fictional, labeled vendor-backed request observation."""

    timestamp: datetime
    vendor_latency_ms: int
    queue_depth: int
    db_connections: int
    retry_count: int
    request_failed: int

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        for field_name in (
            "vendor_latency_ms",
            "queue_depth",
            "db_connections",
            "retry_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.request_failed not in (0, 1):
            raise ValueError("request_failed must be 0 or 1")


def load_request_outcomes(path: str | Path) -> list[RequestOutcome]:
    """Load Chapter 2's deterministic educational labeled observations."""
    outcomes: list[RequestOutcome] = []
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != list(REQUEST_OUTCOME_FIELDS):
            raise ValueError(
                "CSV header must be exactly: " + ",".join(REQUEST_OUTCOME_FIELDS)
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                outcome = RequestOutcome(
                    timestamp=datetime.fromisoformat(
                        row["timestamp"].replace("Z", "+00:00")
                    ),
                    vendor_latency_ms=int(row["vendor_latency_ms"]),
                    queue_depth=int(row["queue_depth"]),
                    db_connections=int(row["db_connections"]),
                    retry_count=int(row["retry_count"]),
                    request_failed=int(row["request_failed"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid request outcome row {line_number}: {error}"
                ) from error
            if outcomes and outcome.timestamp <= outcomes[-1].timestamp:
                raise ValueError(
                    f"invalid request outcome row {line_number}: timestamps must be "
                    "strictly chronological"
                )
            outcomes.append(outcome)
    return outcomes


REQUEST_FAILURE = MLProblem(
    name="Request failure prediction",
    engineering_question="Will this vendor-backed request fail?",
    problem_type=ProblemType.BINARY_CLASSIFICATION,
    features=("vendor_latency_ms", "queue_depth", "db_connections", "retry_count"),
    target="request_failed",
)

INCIDENT_CLASSIFICATION = MLProblem(
    name="Incident classification",
    engineering_question="Which known incident category matches this telemetry?",
    problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
    features=(
        "api_latency_ms",
        "error_rate",
        "db_connections",
        "queue_depth",
        "vendor_latency_ms",
    ),
    target="incident_type",
)

FUTURE_LATENCY = MLProblem(
    name="Future latency prediction",
    engineering_question="What will Harbor's API latency be 10 minutes from now?",
    problem_type=ProblemType.REGRESSION,
    features=(
        "api_latency_ms",
        "error_rate",
        "queue_depth",
        "vendor_latency_ms",
        "requests_per_minute",
    ),
    target="future_api_latency_ms",
)

TELEMETRY_ANOMALY = MLProblem(
    name="Telemetry anomaly detection",
    engineering_question=(
        "Does the current telemetry window look abnormal compared with normal "
        "Harbor operation?"
    ),
    problem_type=ProblemType.ANOMALY_DETECTION,
    features=(
        "api_latency_ms",
        "error_rate",
        "db_connections",
        "queue_depth",
        "vendor_latency_ms",
        "requests_per_minute",
    ),
    target=None,
)

HARBOR_PROBLEMS = (
    REQUEST_FAILURE,
    INCIDENT_CLASSIFICATION,
    FUTURE_LATENCY,
    TELEMETRY_ANOMALY,
)
