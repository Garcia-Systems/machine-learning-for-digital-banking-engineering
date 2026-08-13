from datetime import timezone
from pathlib import Path

import pytest

from harbor_ml import load_telemetry, summarize_telemetry

FIXTURE = Path(__file__).parents[1] / "data/harbor_incident_telemetry.csv"


def test_loads_fixture_as_typed_observations() -> None:
    observations = load_telemetry(FIXTURE)

    assert len(observations) == 8
    assert observations[0].timestamp.isoformat() == "2026-08-12T10:10:00+00:00"
    assert observations[0].timestamp.tzinfo is timezone.utc
    assert observations[0].api_latency_ms == 180
    assert isinstance(observations[0].api_latency_ms, int)
    assert observations[0].error_rate == pytest.approx(0.004)
    assert isinstance(observations[0].error_rate, float)


def test_fixture_is_chronologically_ordered() -> None:
    observations = load_telemetry(FIXTURE)

    assert all(
        earlier.timestamp < later.timestamp
        for earlier, later in zip(observations, observations[1:])
    )


def test_summary_calculations_are_descriptive_and_deterministic() -> None:
    summary = summarize_telemetry(load_telemetry(FIXTURE))

    assert summary.minimum_api_latency_ms == 180
    assert summary.maximum_api_latency_ms == 2380
    assert summary.average_api_latency_ms == pytest.approx(794.5)
    assert summary.minimum_vendor_latency_ms == 220
    assert summary.maximum_vendor_latency_ms == 1900
    assert summary.queue_growth == 135
    assert summary.error_rate_change == pytest.approx(0.083)


@pytest.mark.parametrize(
    "bad_row, expected_message",
    [
        ("not-a-time,180,0.004,31,12,220", "invalid telemetry row 2"),
        ("2026-08-12T10:10:00Z,slow,0.004,31,12,220", "invalid telemetry row 2"),
        ("2026-08-12T10:10:00Z,180,1.5,31,12,220", "error_rate"),
    ],
)
def test_malformed_rows_have_clear_errors(
    tmp_path: Path, bad_row: str, expected_message: str
) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "timestamp,api_latency_ms,error_rate,db_connections,queue_depth,"
        f"vendor_latency_ms\n{bad_row}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_message):
        load_telemetry(csv_path)


def test_out_of_order_rows_are_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "unordered.csv"
    csv_path.write_text(
        "timestamp,api_latency_ms,error_rate,db_connections,queue_depth,"
        "vendor_latency_ms\n"
        "2026-08-12T10:11:00Z,191,0.005,32,14,235\n"
        "2026-08-12T10:10:00Z,180,0.004,31,12,220\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly chronological"):
        load_telemetry(csv_path)
