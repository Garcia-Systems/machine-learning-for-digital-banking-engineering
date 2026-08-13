"""Run Chapter 20's in-process ML-assisted monitoring dashboard laboratory."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from harbor_ml.dashboard import build_teaching_service  # noqa: E402
from harbor_ml.dashboard.app import create_dashboard_app  # noqa: E402
from harbor_ml.dashboard.service import teaching_scenarios  # noqa: E402


def main() -> None:
    print("Harbor Federal Credit Union\nML-Assisted Monitoring Dashboard Laboratory")
    now = datetime(2025, 1, 15, 14, 35, tzinfo=timezone.utc)
    service = build_teaching_service(ROOT)
    snapshots = {}
    for name, telemetry, request in teaching_scenarios(now):
        unavailable = name == "ml_unavailable"
        prediction_time = now - timedelta(minutes=20) if name == "stale_prediction" else now
        snapshots[name] = service.assemble(
            telemetry, request, now=now, prediction_timestamp=prediction_time,
            ml_available=not unavailable,
        )

    current = snapshots["vendor_degradation"]
    response = TestClient(create_dashboard_app(service, current)).get("/dashboard")
    assert response.status_code == 200
    assert "Observed — system health" in response.text
    assert "Model suggests — ML signals" in response.text
    assert "ML prediction unavailable" in response.text  # recent-history state

    print("\nDeterministic scenarios")
    for name, snapshot in snapshots.items():
        probability = (
            f"{snapshot.integration_failure_probability:.3f}"
            if snapshot.integration_failure_probability is not None else "unavailable"
        )
        print(
            f"- {name}: severity={snapshot.severity}, ML={snapshot.ml_status.value}, "
            f"pattern={snapshot.predicted_incident_class or 'unavailable'}, failure probability={probability}"
        )

    print(f"""
Current system health

API latency:       {current.telemetry.api_latency_ms:.0f} ms
Error rate:        {current.telemetry.error_rate:.1%}
Queue depth:       {current.telemetry.queue_depth}
DB connections:    {current.telemetry.db_connections}
Vendor latency:    {current.telemetry.vendor_latency_ms:.0f} ms

Telemetry anomaly: {'YES' if current.telemetry_anomaly else 'NO'}
Anomaly score:     {current.anomaly_score:.4f} (not a probability)
Incident model:    {current.predicted_incident_class}
Integration failure probability: {current.integration_failure_probability:.3f}
Model version:     {current.model_version}

Interpretation:
The ML signals suggest behavior resembling a historical incident pattern.
This is not a confirmed diagnosis.

Suggested investigation:""")
    for item in current.investigation_guidance:
        print(f"- {item}")
    print("\nGET /dashboard returned semantic HTML in process; no server or external infrastructure was needed.")


if __name__ == "__main__":
    main()
