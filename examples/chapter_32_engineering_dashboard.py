"""Run Chapter 32's deterministic capstone dashboard laboratory."""

from datetime import timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.dashboard.service import build_capstone_dashboard  # noqa: E402


def summarize(snapshot) -> None:
    telemetry = snapshot.telemetry
    print(f"\nSnapshot: {telemetry.timestamp:%H:%M}\n\nOBSERVED\n--------")
    print(f"API latency: {telemetry.api_latency_ms:.0f} ms\nVendor latency: {telemetry.vendor_latency_ms:.0f} ms")
    print(f"Queue depth: {telemetry.queue_depth}\nDB connections: {telemetry.db_connections}\nError rate: {telemetry.error_rate:.1%}")
    print(f"\nDETERMINISTIC STATUS\n--------------------\n{snapshot.severity}")
    print(f"\nML CAPABILITY\n-------------\n{snapshot.ml_capability_status}")
    print("\nANOMALY DETECTOR\n----------------")
    print("Prediction unavailable" if snapshot.anomaly.score is None else
          f"Anomaly: {'yes' if snapshot.anomaly.is_anomaly else 'no'}\nScore: {snapshot.anomaly.score:.3f}\nModel: {snapshot.anomaly.metadata.model_name}\nVersion: {snapshot.anomaly.metadata.model_version}")
    print("\nINCIDENT MODEL\n--------------")
    print("Prediction unavailable" if not snapshot.incident.predicted_class else
          f"Top pattern: {snapshot.incident.predicted_class}\nTop probability: {snapshot.incident.ordered_probabilities[0][1]:.3f}\nAmbiguous: {'yes' if snapshot.incident.ambiguous else 'no'}")
    print("\nINTEGRATION FAILURE\n-------------------")
    print("Prediction unavailable" if snapshot.integration_failure.failure_probability is None else
          f"Failure probability: {snapshot.integration_failure.failure_probability:.3f}")
    print("\nMODEL SUGGESTS\n--------------\nModel signals prioritize investigation; they do not diagnose the incident.")
    print("\nCONFIRMED\n---------\n" + ("\n".join(snapshot.confirmed_evidence) or "No root cause confirmed yet."))
    print("\nINVESTIGATE\n-----------")
    for item in snapshot.investigation_guidance: print(f"- {item}")


def main() -> None:
    service, rows = build_capstone_dashboard(ROOT)
    healthy, early, ambiguous, classifier_down, final = (rows[i] for i in (0, 6, 17, 12, 15))
    scenarios = [service.build_snapshot(healthy), service.build_snapshot(early),
                 service.build_snapshot(ambiguous), service.build_snapshot(classifier_down),
                 service.build_snapshot(ambiguous, all_ml_available=False),
                 service.build_snapshot(final, retrospective=True),
                 service.build_snapshot(early, prediction_timestamps={
                     "anomaly": early.timestamp-timedelta(minutes=6),
                     "incident": early.timestamp-timedelta(minutes=8),
                     "integration": early.timestamp-timedelta(minutes=10)})]
    print("Harbor Federal Credit Union\nCapstone Engineering Dashboard Laboratory")
    for snapshot in scenarios: summarize(snapshot)
    print("\nRendered HTML is available from: PYTHONPATH=src uvicorn harbor_ml.dashboard.run:app")


if __name__ == "__main__": main()
