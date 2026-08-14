"""Train and persist Harbor's Chapter 28 capstone anomaly detector."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.capstone_anomaly import (  # noqa: E402
    build_capstone_anomaly_timeline, create_anomaly_metadata, evaluate_detection_behavior,
    save_capstone_anomaly_artifact, score_capstone_timeline, select_anomaly_baseline,
    train_capstone_anomaly_detector, validate_anomaly_baseline,
)
from harbor_ml.capstone_dataset import load_capstone_sources  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/capstone-anomaly")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeline = build_capstone_anomaly_timeline(load_capstone_sources(ROOT / "data"))
    baseline = select_anomaly_baseline(timeline)
    validate_anomaly_baseline(baseline, evaluation=timeline[len(baseline):])
    model = train_capstone_anomaly_detector(baseline, contamination=args.contamination,
                                             random_state=args.random_state)
    results = score_capstone_timeline(model, timeline)
    behavior = evaluate_detection_behavior(results)
    metadata = create_anomaly_metadata(baseline, timeline, behavior,
        contamination=args.contamination, random_state=args.random_state)
    model_path, metadata_path = save_capstone_anomaly_artifact(model, metadata, args.output_dir)
    print("Harbor Federal Credit Union\nCapstone Anomaly Detector Training")
    print(f"Baseline: {baseline[0].timestamp.isoformat()} .. {baseline[-1].timestamp.isoformat()} ({len(baseline)} rows)")
    print(f"Dataset SHA-256: {metadata.dataset_sha256}\nModel version: {metadata.model_version}")
    print(f"First anomaly: {behavior.first_anomaly_timestamp.isoformat() if behavior.first_anomaly_timestamp else 'none'}")
    print(f"Recovery-proxy anomaly rate: {behavior.healthy_anomaly_rate:.1%}")
    print(f"Model: {model_path}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
