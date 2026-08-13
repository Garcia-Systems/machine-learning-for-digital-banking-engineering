"""Structured telemetry observations and descriptive summaries for Chapter 1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean

CSV_FIELDS = (
    "timestamp",
    "api_latency_ms",
    "error_rate",
    "db_connections",
    "queue_depth",
    "vendor_latency_ms",
)


@dataclass(frozen=True)
class TelemetryObservation:
    """One time-aligned view of Harbor's fictional operational telemetry."""

    timestamp: datetime
    api_latency_ms: int
    error_rate: float
    db_connections: int
    queue_depth: int
    vendor_latency_ms: int

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        for field_name in (
            "api_latency_ms",
            "db_connections",
            "queue_depth",
            "vendor_latency_ms",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")


@dataclass(frozen=True)
class TelemetrySummary:
    """Deterministic descriptive statistics, not ML predictions."""

    minimum_api_latency_ms: int
    maximum_api_latency_ms: int
    average_api_latency_ms: float
    minimum_vendor_latency_ms: int
    maximum_vendor_latency_ms: int
    queue_growth: int
    error_rate_change: float


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp, including the common ``Z`` UTC suffix."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_telemetry(path: str | Path) -> list[TelemetryObservation]:
    """Load and validate chronological observations from a Chapter 1 CSV file."""
    observations: list[TelemetryObservation] = []
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(
                "CSV header must be exactly: " + ",".join(CSV_FIELDS)
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                observation = TelemetryObservation(
                    timestamp=_parse_timestamp(row["timestamp"]),
                    api_latency_ms=int(row["api_latency_ms"]),
                    error_rate=float(row["error_rate"]),
                    db_connections=int(row["db_connections"]),
                    queue_depth=int(row["queue_depth"]),
                    vendor_latency_ms=int(row["vendor_latency_ms"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid telemetry row {line_number}: {error}") from error

            if observations and observation.timestamp <= observations[-1].timestamp:
                raise ValueError(
                    f"invalid telemetry row {line_number}: timestamps must be "
                    "strictly chronological"
                )
            observations.append(observation)
    return observations


def summarize_telemetry(
    observations: list[TelemetryObservation],
) -> TelemetrySummary:
    """Calculate simple descriptions of a non-empty chronological timeline."""
    if not observations:
        raise ValueError("cannot summarize an empty telemetry timeline")

    api_latencies = [item.api_latency_ms for item in observations]
    vendor_latencies = [item.vendor_latency_ms for item in observations]
    return TelemetrySummary(
        minimum_api_latency_ms=min(api_latencies),
        maximum_api_latency_ms=max(api_latencies),
        average_api_latency_ms=fmean(api_latencies),
        minimum_vendor_latency_ms=min(vendor_latencies),
        maximum_vendor_latency_ms=max(vendor_latencies),
        queue_growth=observations[-1].queue_depth - observations[0].queue_depth,
        error_rate_change=observations[-1].error_rate - observations[0].error_rate,
    )
