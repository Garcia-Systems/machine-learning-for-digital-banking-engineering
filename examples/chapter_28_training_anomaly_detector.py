"""Train and inspect Chapter 28's healthy-baseline anomaly detector."""

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.capstone_anomaly import (  # noqa: E402
    CAPSTONE_ANOMALY_FEATURES, build_capstone_anomaly_timeline, create_anomaly_metadata,
    evaluate_detection_behavior, load_capstone_anomaly_artifact, save_capstone_anomaly_artifact,
    score_capstone_timeline, select_anomaly_baseline, train_capstone_anomaly_detector,
    validate_anomaly_baseline,
)
from harbor_ml.capstone_dataset import load_capstone_sources  # noqa: E402


def run(output_dir: Path) -> None:
    timeline = build_capstone_anomaly_timeline(load_capstone_sources(ROOT / "data"))
    baseline = select_anomaly_baseline(timeline)
    incident_evaluation = tuple(row for row in timeline if row.timestamp > baseline[-1].timestamp)
    validate_anomaly_baseline(baseline, evaluation=incident_evaluation)
    print("Harbor Federal Credit Union\nCapstone Anomaly Detector Training Laboratory")
    print(f"\nCapstone observations: {len(timeline)}")
    print(f"Full range: {timeline[0].timestamp.isoformat()} .. {timeline[-1].timestamp.isoformat()}")
    print("\nTraining baseline")
    print(f"Start: {baseline[0].timestamp.isoformat()}\nEnd: {baseline[-1].timestamp.isoformat()}\nRows: {len(baseline)}")
    print("\nFeatures:\n" + "\n".join(f"- {name}" for name in CAPSTONE_ANOMALY_FEATURES))
    print("\nTraining Isolation Forest...")
    model = train_capstone_anomaly_detector(baseline)
    print("Complete.\n\ntime    anomaly_score  anomaly?  phase")
    results = score_capstone_timeline(model, timeline)
    for row in results:
        print(f"{row.timestamp:%H:%M}   {row.anomaly_score: .4f}       "
              f"{'yes' if row.is_anomaly else 'no ':3}       {row.phase}")
    # Chapter 26's narrative first describes representative request failures at 10:20.
    first_request_failure = datetime(2026, 8, 12, 10, 20, tzinfo=timezone.utc)
    behavior = evaluate_detection_behavior(results,
        reference_incident_timestamp=first_request_failure)
    print(f"\nRecovery-proxy anomaly rate: {behavior.healthy_flagged_rows}/"
          f"{behavior.healthy_eval_rows} = {behavior.healthy_anomaly_rate:.1%}")
    print(f"First anomaly: {behavior.first_anomaly_timestamp.isoformat() if behavior.first_anomaly_timestamp else 'none'}")
    print(f"First request failure milestone: {first_request_failure.isoformat()}")
    print(f"Detection lead time: {behavior.lead_time_seconds / 60:.0f} minutes"
          if behavior.lead_time_seconds is not None else "Detection lead time: unavailable")
    metadata = create_anomaly_metadata(baseline, timeline, behavior)
    model_path, metadata_path = save_capstone_anomaly_artifact(model, metadata, output_dir)
    reloaded = load_capstone_anomaly_artifact(model_path)
    reloaded_results = score_capstone_timeline(reloaded, timeline)
    passed = np.allclose([row.anomaly_score for row in results[:5]],
                         [row.anomaly_score for row in reloaded_results[:5]])
    print(f"\nArtifact: {model_path}\nMetadata: {metadata_path}")
    print(f"Round-trip verification: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise RuntimeError("artifact round-trip changed scores")


if __name__ == "__main__":
    run(ROOT / "artifacts/capstone-anomaly")
