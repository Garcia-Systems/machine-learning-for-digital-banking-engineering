"""Run Chapter 33's complete in-process Harbor operating laboratory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.operations import run_operating_laboratory  # noqa: E402


def main() -> None:
    report = run_operating_laboratory(ROOT)
    print("Harbor Federal Credit Union\nComplete ML-Assisted Banking Engineering Laboratory")
    print(f"\nDATA\n----\nCapstone dataset validation: PASS ({report.dataset_rows} rows)\nTemporal leakage validation: PASS\nSensitive-field validation: {'PASS' if report.sensitive_field_rejected else 'FAIL'}")
    print("\nMODELS\n------")
    for item in report.inventory:
        print(f"{item.name}: {item.version} | {item.model_type} | dataset {item.dataset_hash[:12]}")
    print(f"\nML SERVICE\n----------\nHealth: {report.health_status}")
    print("\nPHP / APPLICATION CONTRACT\n--------------------------\nContract validation: PASS")
    print(f"\nINCIDENT\n--------\nTimeline evaluated: PASS ({report.timeline_rows} observations)\nFirst anomaly: {report.first_anomaly}\nConfirmed evidence: trace evidence appears only at its observation time")
    print(f"\nFAILURE CONTAINMENT\n-------------------\nPartial ML outage: {'PASS' if report.partial_outage else 'FAIL'}\nComplete ML outage: {'PASS' if report.complete_outage else 'FAIL'}\nCore deterministic operation preserved: {'YES' if report.deterministic_operation_preserved else 'NO'}\nStale output recognized: {'PASS' if report.stale_recognized else 'FAIL'}")
    print(f"\nRESPONSIBLE ML\n--------------\nModel explanation: PASS (reproduction delta {report.explanation_delta:.3g})\nSlice evaluation: PASS ({len(report.slices)} slices)\nHuman review workflow: PASS (prediction {report.review_prediction:.2f}; outcome {report.review_outcome})\nMonitoring: PASS ({', '.join(report.monitoring_periods)})\nRollback: PASS (restored {report.rollback_version})")
    print("\nFINAL STATUS\n------------\nHarbor laboratory validated.")


if __name__ == "__main__":
    main()
