"""Leakage-resistant, time-aligned dataset construction for Chapter 27."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Sequence, TypeVar

from .capstone_incident import TraceSpan, load_capstone_incident, load_capstone_traces
from .incident_classifier import INCIDENT_CLASSES

DATASET_NAME = "harbor-capstone-training"
DATASET_VERSION = "harbor-capstone-dataset-v1"
GENERATOR_VERSION = "chapter-27-v1"
LOOKBACK = timedelta(minutes=5)
DEFAULT_FRESHNESS = {
    "application_metrics": timedelta(seconds=60),
    "database_metrics": timedelta(seconds=60),
    "vendor_metrics": timedelta(seconds=120),
}
MODEL_FEATURES = (
    "api_latency_ms", "error_rate", "queue_depth", "requests_per_minute",
    "db_connections", "recent_db_latency_ms", "vendor_latency_ms",
    "vendor_error_rate", "retry_count", "vendor_latency_mean_5m",
    "vendor_latency_max_5m", "error_rate_mean_5m", "queue_growth_5m",
    "requests_mean_5m", "retry_count_5m", "db_connections_mean_5m",
)
OUTPUT_FIELDS = ("observation_time", *MODEL_FEATURES, "incident_type")
PROHIBITED_FIELDS = frozenset({
    "member_id", "account_number", "email", "name", "social_security_number",
    "current_request_vendor_duration_ms", "final_status", "failure_reason",
})


@dataclass(frozen=True)
class ApplicationMetric:
    timestamp: datetime
    api_latency_ms: float
    error_rate: float
    queue_depth: int
    requests_per_minute: float


@dataclass(frozen=True)
class DatabaseMetric:
    timestamp: datetime
    db_connections: int
    recent_db_latency_ms: float


@dataclass(frozen=True)
class VendorMetric:
    timestamp: datetime
    vendor: str
    vendor_latency_ms: float
    vendor_error_rate: float
    retry_count: int


@dataclass(frozen=True)
class RequestOutcome:
    request_id: str
    created_at: datetime
    completed_at: datetime
    request_failed: int


@dataclass(frozen=True)
class CapstoneSources:
    application: tuple[ApplicationMetric, ...]
    database: tuple[DatabaseMetric, ...]
    vendor: tuple[VendorMetric, ...]
    traces: tuple[TraceSpan, ...]
    outcomes: tuple[RequestOutcome, ...]
    source_paths: Mapping[str, Path]


@dataclass(frozen=True)
class SourceSelection:
    source_name: str
    source_timestamp: datetime
    age_seconds: float


@dataclass(frozen=True)
class CapstoneTrainingExample:
    observation_time: datetime
    features: Mapping[str, float]
    incident_type: str
    source_selections: tuple[SourceSelection, ...]


T = TypeVar("T")


def parse_timestamp(value: str, context: str = "timestamp") -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"invalid {context}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{context} must be timezone-aware")
    return timestamp


def _validate_and_sort(records: Sequence[T], source_name: str) -> tuple[T, ...]:
    ordered = sorted(records, key=lambda item: item.timestamp)  # type: ignore[attr-defined]
    seen: dict[datetime, T] = {}
    result = []
    for record in ordered:
        timestamp = record.timestamp  # type: ignore[attr-defined]
        prior = seen.get(timestamp)
        if prior is not None:
            if prior != record:
                raise ValueError(f"{source_name}: conflicting records at {timestamp.isoformat()}")
            continue
        seen[timestamp] = record
        result.append(record)
    return tuple(result)


def load_request_outcomes(path: str | Path) -> tuple[RequestOutcome, ...]:
    """Load the existing Chapter 2 outcome fixture with explicit completion semantics."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    required = {"timestamp", "request_failed"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("invalid request outcome header")
    outcomes = []
    for number, row in enumerate(rows, 2):
        completed = parse_timestamp(row["timestamp"], f"outcome row {number} timestamp")
        try:
            failed = int(row["request_failed"])
        except ValueError as error:
            raise ValueError(f"outcome row {number}: invalid request_failed") from error
        if failed not in {0, 1}:
            raise ValueError(f"outcome row {number}: request_failed must be zero or one")
        outcomes.append(RequestOutcome(f"historical-request-{number - 1:03d}",
                                       completed - timedelta(seconds=10), completed, failed))
    return tuple(sorted(outcomes, key=lambda item: item.completed_at))


def load_capstone_sources(data_directory: str | Path) -> CapstoneSources:
    """Load existing fictional fixtures and expose their distinct source semantics."""
    data = Path(data_directory)
    incident_path = data / "harbor_capstone_incident.csv"
    trace_path = data / "harbor_capstone_traces.csv"
    outcome_path = data / "harbor_request_outcomes.csv"
    timeline = load_capstone_incident(incident_path)
    application = [ApplicationMetric(row.timestamp, row.api_latency_ms, row.error_rate,
                                     row.queue_depth, row.requests_per_minute) for row in timeline]
    # Chapter 26 has no separate DB-latency instrument. This transparent teaching proxy
    # is kept semantically named rather than merged with API or vendor latency.
    database = [DatabaseMetric(row.timestamp, row.db_connections,
                               12.0 + row.db_connections * 0.8) for row in timeline]
    vendor = [VendorMetric(row.timestamp, "ClearVerify", row.vendor_latency_ms,
                           row.vendor_timeout_rate, row.retry_count) for row in timeline]
    return CapstoneSources(
        _validate_and_sort(application, "application_metrics"),
        _validate_and_sort(database, "database_metrics"),
        _validate_and_sort(vendor, "vendor_metrics"),
        tuple(sorted(load_capstone_traces(trace_path), key=lambda item: item.timestamp)),
        load_request_outcomes(outcome_path),
        {"capstone_telemetry": incident_path, "traces": trace_path, "request_outcomes": outcome_path},
    )


def select_latest_before(records: Sequence[T], observation_time: datetime, *,
                         source_name: str, maximum_age: timedelta,
                         timestamp: Callable[[T], datetime] = lambda item: item.timestamp) -> T:
    """Return the newest eligible as-of value, rejecting absent or stale data."""
    eligible = [record for record in records if timestamp(record) <= observation_time]
    if not eligible:
        raise ValueError(f"{source_name} unavailable at observation time")
    selected = max(eligible, key=timestamp)
    age = observation_time - timestamp(selected)
    if age > maximum_age:
        raise ValueError(f"{source_name} too stale: age={age.total_seconds():.0f}s")
    return selected


def select_window(records: Sequence[T], observation_time: datetime,
                  lookback: timedelta = LOOKBACK,
                  timestamp: Callable[[T], datetime] = lambda item: item.timestamp) -> tuple[T, ...]:
    """Select an inclusive event-time window that can never contain future records."""
    start = observation_time - lookback
    return tuple(record for record in records if start <= timestamp(record) <= observation_time)


def _queue_growth(application: Sequence[ApplicationMetric], observation_time: datetime) -> float:
    now = select_latest_before(application, observation_time, source_name="application_metrics",
                               maximum_age=DEFAULT_FRESHNESS["application_metrics"])
    cutoff = observation_time - LOOKBACK
    older = [row for row in application if row.timestamp <= cutoff]
    if not older:
        raise ValueError("queue_growth_5m unavailable: no sufficiently old observation")
    anchor = max(older, key=lambda item: item.timestamp)
    if cutoff - anchor.timestamp > timedelta(minutes=2):
        raise ValueError("queue_growth_5m unavailable: historical observation too stale")
    return float(now.queue_depth - anchor.queue_depth)


def build_rolling_features(sources: CapstoneSources, observation_time: datetime) -> dict[str, float]:
    app = select_window(sources.application, observation_time)
    database = select_window(sources.database, observation_time)
    vendor = select_window(sources.vendor, observation_time)
    if not app or not database or not vendor:
        raise ValueError("rolling feature window is empty")
    return {
        "vendor_latency_mean_5m": fmean(row.vendor_latency_ms for row in vendor),
        "vendor_latency_max_5m": max(row.vendor_latency_ms for row in vendor),
        "error_rate_mean_5m": fmean(row.error_rate for row in app),
        "queue_growth_5m": _queue_growth(sources.application, observation_time),
        "requests_mean_5m": fmean(row.requests_per_minute for row in app),
        "retry_count_5m": float(sum(row.retry_count for row in vendor)),
        "db_connections_mean_5m": fmean(row.db_connections for row in database),
    }


def _historical_incident_type(timestamp: datetime) -> str:
    minute = timestamp.minute
    if minute <= 4 or minute >= 32:
        return "normal"
    if minute <= 18:
        return "vendor_degradation"
    return "database_pressure"


def build_capstone_observation(
    sources: CapstoneSources, observation_time: datetime, *,
    freshness: Mapping[str, timedelta] = DEFAULT_FRESHNESS,
    incident_type: str | None = None,
) -> CapstoneTrainingExample:
    """Reconstruct only what was knowable at ``observation_time``; never read outcomes/traces."""
    app = select_latest_before(sources.application, observation_time,
                               source_name="application_metrics",
                               maximum_age=freshness["application_metrics"])
    database = select_latest_before(sources.database, observation_time,
                                    source_name="database_metrics",
                                    maximum_age=freshness["database_metrics"])
    vendor = select_latest_before(sources.vendor, observation_time,
                                  source_name="vendor_metrics",
                                  maximum_age=freshness["vendor_metrics"])
    features = {
        "api_latency_ms": app.api_latency_ms, "error_rate": app.error_rate,
        "queue_depth": float(app.queue_depth), "requests_per_minute": app.requests_per_minute,
        "db_connections": float(database.db_connections),
        "recent_db_latency_ms": database.recent_db_latency_ms,
        "vendor_latency_ms": vendor.vendor_latency_ms,
        "vendor_error_rate": vendor.vendor_error_rate, "retry_count": float(vendor.retry_count),
        **build_rolling_features(sources, observation_time),
    }
    label = incident_type or _historical_incident_type(observation_time)
    if label not in INCIDENT_CLASSES:
        raise ValueError(f"invalid incident_type: {label}")
    selections = tuple(SourceSelection(name, record.timestamp,
                                       (observation_time - record.timestamp).total_seconds())
                       for name, record in (("application_metrics", app),
                                            ("database_metrics", database),
                                            ("vendor_metrics", vendor)))
    example = CapstoneTrainingExample(observation_time, features, label, selections)
    validate_temporal_integrity(example)
    return example


def build_label(outcome: RequestOutcome, observation_time: datetime) -> int:
    """Build a historical target separately, and only from a later completion."""
    if outcome.created_at > observation_time or outcome.completed_at <= observation_time:
        raise ValueError("outcome must describe a request known by, but completed after, observation time")
    return outcome.request_failed


def validate_temporal_integrity(example: CapstoneTrainingExample) -> None:
    if set(example.features) != set(MODEL_FEATURES):
        raise ValueError("feature contract mismatch")
    if any(name in PROHIBITED_FIELDS for name in example.features):
        raise ValueError("prohibited field in model features")
    if any(not isfinite(float(value)) for value in example.features.values()):
        raise ValueError("model features must be finite")
    if any(selection.source_timestamp > example.observation_time for selection in example.source_selections):
        raise ValueError("future source timestamp selected")
    if example.incident_type in example.features:
        raise ValueError("incident label must not be a model feature")


def build_capstone_dataset(sources: CapstoneSources) -> list[CapstoneTrainingExample]:
    examples = []
    for row in sources.application:
        try:
            examples.append(build_capstone_observation(sources, row.timestamp))
        except ValueError as error:
            if "queue_growth_5m unavailable" not in str(error):
                raise
    return examples


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_capstone_dataset(examples: Sequence[CapstoneTrainingExample], output_path: str | Path,
                           metadata_path: str | Path, source_paths: Mapping[str, Path]) -> dict[str, object]:
    """Write deterministic CSV/metadata artifacts and record every fixture fingerprint."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for example in examples:
            writer.writerow({"observation_time": example.observation_time.isoformat(),
                             **example.features, "incident_type": example.incident_type})
    metadata: dict[str, object] = {
        "dataset_name": DATASET_NAME, "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "sources": {name: sha256_file(path) for name, path in sorted(source_paths.items())},
        "output_sha256": sha256_file(output), "row_count": len(examples),
        "feature_names": list(MODEL_FEATURES),
    }
    metadata_output = Path(metadata_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
