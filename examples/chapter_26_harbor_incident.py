"""Run Chapter 26's evidence-ordered Harbor incident laboratory."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.anomaly_detection import (  # noqa: E402
    build_anomaly_detector, build_anomaly_features, load_normal_telemetry,
    train_anomaly_detector,
)
from harbor_ml.capstone_incident import (  # noqa: E402
    CONFIRMING_EVIDENCE_TIME, dominant_trace_component, evaluate_incident_timeline,
    format_incident_observation, load_capstone_incident, load_capstone_traces,
)
from harbor_ml.incident_classifier import (  # noqa: E402
    build_incident_classifier, build_incident_features, build_incident_targets,
    load_incident_dataset, train_incident_classifier,
)
from harbor_ml.training import TrainingConfig, train_integration_failure_model  # noqa: E402


def main() -> None:
    timeline = load_capstone_incident(ROOT / "data/harbor_capstone_incident.csv")
    baseline = load_normal_telemetry(ROOT / "data/harbor_normal_telemetry.csv")
    anomaly = train_anomaly_detector(
        build_anomaly_detector(), build_anomaly_features(baseline))
    history = load_incident_dataset(ROOT / "data/harbor_incident_classes.csv")
    classifier = train_incident_classifier(
        build_incident_classifier(), build_incident_features(history),
        build_incident_targets(history))
    integration = train_integration_failure_model(
        ROOT / "data/harbor_integration_requests.csv", TrainingConfig(),
        now=lambda: datetime(2026, 8, 12, 9, 55, tzinfo=timezone.utc))
    observations = evaluate_incident_timeline(
        timeline, anomaly, classifier, integration.pipeline,
        integration_model_version=integration.metadata.model_version)

    print("Harbor Federal Credit Union\nCapstone Incident Laboratory")
    print("\nOBSERVATION → MODEL SIGNAL → HYPOTHESIS → INVESTIGATION → EVIDENCE → DIAGNOSIS → ACTION")
    for item in observations:
        print("\n" + "=" * 72)
        print(format_incident_observation(item))
        if item.timestamp == CONFIRMING_EVIDENCE_TIME:
            spans = load_capstone_traces(ROOT / "data/harbor_capstone_traces.csv")
            request = "capstone-request-001"
            selected = [span for span in spans if span.request_id == request]
            print("\nCONFIRMING TRACE EVIDENCE")
            print(f"{request} total: {sum(span.duration_ms for span in selected)} ms")
            for span in selected:
                print(f"  {span.component:<20} {span.duration_ms:>4} ms  {span.status}")
            dominant = dominant_trace_component(spans, request)
            print(f"Dominant component: {dominant.component} ({dominant.duration_ms} ms)")

    print("\n" + "=" * 72)
    print("RETROSPECTIVE")
    print("- Vendor latency and request-level failure probability provided localized early signals.")
    print("- The anomaly detector surfaced unusual combined telemetry; it did not name a cause.")
    print("- The classifier prioritized known patterns and became ambiguous as secondary pressure grew.")
    print("- At 10:24 the classifier was unavailable; telemetry and investigation remained available.")
    print("- Computed trace durations, timeout status, retries, and operational telemetry confirmed the diagnosis.")
    print("\nML helps prioritize hypotheses. Evidence establishes diagnosis.")


if __name__ == "__main__":
    main()
