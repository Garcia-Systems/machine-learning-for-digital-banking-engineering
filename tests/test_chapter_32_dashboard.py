"""Chapter 32 evidence-boundary and presentation tests."""

import asyncio
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from harbor_ml.dashboard.app import create_dashboard_app
from harbor_ml.dashboard.service import CONFIRMED_TRACE_TEXT, build_capstone_dashboard

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def capstone():
    service, rows = build_capstone_dashboard(ROOT)
    snapshots = [service.build_snapshot(row) for row in rows]
    return service, rows, snapshots


def _get(app, path):
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get(path)
    return asyncio.run(request())


def test_snapshot_assembles_actual_independent_model_results(capstone):
    _, _, snapshots = capstone
    item = next(x for x in snapshots if x.telemetry.timestamp.minute == 18)
    assert item.severity == "warning"
    assert item.anomaly.score is not None
    assert sum(item.incident.probabilities.values()) == pytest.approx(1)
    assert item.integration_failure.failure_probability is not None
    assert len({item.anomaly.metadata.model_version, item.incident.metadata.model_version,
                item.integration_failure.metadata.model_version}) == 3


def test_partial_and_complete_unavailability_are_not_normal(capstone):
    service, rows, _ = capstone
    partial = service.build_snapshot(rows[12])
    assert partial.ml_capability_status == "partially_available"
    assert partial.incident.predicted_class is None
    assert partial.investigation_guidance  # deterministic queue guidance remains
    missing = service.build_snapshot(rows[9], all_ml_available=False)
    assert missing.ml_capability_status == "unavailable"
    assert missing.anomaly.score is None and missing.integration_failure.failure_probability is None
    assert missing.telemetry.queue_depth == 108


def test_each_signal_has_independent_staleness(capstone):
    service, rows, _ = capstone
    row = rows[9]
    times = {"anomaly": row.timestamp - timedelta(minutes=6),
             "incident": row.timestamp - timedelta(minutes=8),
             "integration": row.timestamp - timedelta(minutes=10)}
    item = service.build_snapshot(row, prediction_timestamps=times)
    assert [item.anomaly.metadata.prediction_age_seconds,
            item.incident.metadata.prediction_age_seconds,
            item.integration_failure.metadata.prediction_age_seconds] == [360, 480, 600]
    assert all(signal.metadata.availability.value == "STALE" for signal in
               (item.anomaly, item.incident, item.integration_failure))


def test_ambiguous_guidance_combines_top_class_paths(capstone):
    _, _, snapshots = capstone
    ambiguous = next(x for x in snapshots if x.incident.ambiguous)
    guidance = " ".join(ambiguous.investigation_guidance).lower()
    top = {name for name, _ in ambiguous.incident.ordered_probabilities[:2]}
    if top == {"vendor_degradation", "database_pressure"}:
        assert "clearverify" in guidance and "database" in guidance


def test_routes_render_semantics_accessibility_and_no_sensitive_content(capstone):
    service, _, snapshots = capstone
    app = create_dashboard_app(service, snapshots[9], snapshots)
    for path in ("/dashboard", "/dashboard/incident?time=10:18"):
        response = _get(app, path)
        assert response.status_code == 200
        text = response.text
        for label in ("OBSERVED", "MODEL SUGGESTS", "CONFIRMED", "Observed system state",
                      "ML capability", "Model explanation", "scope=\"col\""):
            assert label in text
        assert "Measures unusualness relative to the learned healthy baseline." in text
        assert "request-level prediction, not an incident diagnosis" in text
        assert all(term not in text.lower() for term in
                   ("account number", "authentication token", "api secret", "stack trace", "artifact path"))


def test_historical_playback_does_not_leak_future_trace_evidence(capstone):
    service, _, snapshots = capstone
    app = create_dashboard_app(service, snapshots[-1], snapshots)
    early = _get(app, "/dashboard/incident?time=10:18").text
    final = _get(app, "/dashboard/incident?time=10:30").text
    assert "No root cause confirmed yet." in early
    assert CONFIRMED_TRACE_TEXT not in early
    assert CONFIRMED_TRACE_TEXT in final


def test_stale_and_unavailable_text_is_explicit(capstone):
    service, rows, snapshots = capstone
    row = rows[9]
    stale = service.build_snapshot(row, prediction_timestamps={key: row.timestamp-timedelta(minutes=i)
        for key, i in (("anomaly", 6), ("incident", 8), ("integration", 10))})
    unavailable = service.build_snapshot(row, all_ml_available=False)
    assert "STALE" in _get(create_dashboard_app(service, stale), "/dashboard").text
    assert _get(create_dashboard_app(service, unavailable), "/dashboard").text.count("Prediction unavailable") >= 2
