"""Build Chapter 27's deterministic, leakage-resistant capstone dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.capstone_dataset import (  # noqa: E402
    DATASET_VERSION, MODEL_FEATURES, build_capstone_dataset, build_capstone_observation,
    build_label, load_capstone_sources, write_capstone_dataset,
)


def _range(records, attribute="timestamp"):
    return f"{getattr(records[0], attribute).isoformat()} .. {getattr(records[-1], attribute).isoformat()}"


def main() -> None:
    sources = load_capstone_sources(ROOT / "data")
    print("Harbor Federal Credit Union\nCapstone Telemetry Dataset Laboratory")
    print("\nSources loaded")
    inventories = (
        ("Application telemetry", sources.application), ("Database telemetry", sources.database),
        ("Vendor telemetry", sources.vendor), ("Trace records", sources.traces),
    )
    for name, records in inventories:
        fields = ", ".join(records[0].__dataclass_fields__)
        print(f"{name}: {len(records)} | {fields}\n  {_range(records)}")
    print(f"Request outcomes: {len(sources.outcomes)} | request_id, created_at, completed_at, request_failed")
    print(f"  {_range(sources.outcomes, 'completed_at')}")

    prediction_time = next(row.timestamp for row in sources.application if row.timestamp.minute == 16)
    worked = build_capstone_observation(sources, prediction_time)
    print(f"\nBuilding observation at {prediction_time.isoformat()}")
    for selection in worked.source_selections:
        print(f"{selection.source_name}: {selection.source_timestamp.isoformat()} age={selection.age_seconds:.0f}s")
    print("\nFeatures (outcomes and traces are absent):")
    for name in MODEL_FEATURES:
        print(f"  {name}: {worked.features[name]:.4g}")

    outcome = sources.outcomes[0]
    print("\nFeature/label separation:")
    print(f"  historical feature time: {outcome.created_at.isoformat()}")
    print(f"  later completion: {outcome.completed_at.isoformat()}")
    print(f"  request_failed target: {build_label(outcome, outcome.created_at)}")

    print("\nMissing versus zero:")
    zero_vendor = replace(sources.vendor[-1], retry_count=0)
    zero_sources = replace(sources, vendor=(*sources.vendor[:-1], zero_vendor))
    zero_example = build_capstone_observation(zero_sources, zero_vendor.timestamp)
    print(f"  observed retry_count = {zero_example.features['retry_count']:.0f} (valid observation)")
    stale_sources = replace(sources, vendor=tuple(
        row for row in sources.vendor if row.timestamp <= prediction_time - timedelta(minutes=5)))
    try:
        build_capstone_observation(stale_sources, prediction_time)
    except ValueError as error:
        print(f"  observation rejected: {error}; missing was not converted to zero")

    changed_future = replace(outcome, request_failed=1 - outcome.request_failed)
    features_a = build_capstone_observation(replace(sources, outcomes=(outcome,)), prediction_time).features
    features_b = build_capstone_observation(replace(sources, outcomes=(changed_future,)), prediction_time).features
    print(f"\nLeakage validation: {'PASS' if features_a == features_b else 'FAIL'}")

    examples = build_capstone_dataset(sources)
    output = ROOT / "artifacts/capstone-dataset/harbor_capstone_training_examples.csv"
    metadata = ROOT / "artifacts/capstone-dataset/metadata.json"
    manifest = write_capstone_dataset(examples, output, metadata, sources.source_paths)
    counts = Counter(row.incident_type for row in examples)
    print("\nDataset summary")
    print(f"  version: {DATASET_VERSION}")
    print(f"  rows: {len(examples)}")
    print(f"  time range: {examples[0].observation_time.isoformat()} .. {examples[-1].observation_time.isoformat()}")
    print(f"  feature count: {len(MODEL_FEATURES)}")
    for label in ("normal", "vendor_degradation", "database_pressure", "traffic_spike",
                  "application_regression"):
        print(f"  {label.replace('_', '-')} rows: {counts[label]}")
    print(f"  dataset SHA-256: {manifest['output_sha256']}")
    print(f"  CSV: {output.relative_to(ROOT)}\n  metadata: {metadata.relative_to(ROOT)}")
    print("\nTemporal validation: PASS")


if __name__ == "__main__":
    main()
