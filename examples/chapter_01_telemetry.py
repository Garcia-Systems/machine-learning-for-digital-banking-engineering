"""Print Chapter 1's fictional telemetry timeline and descriptive statistics."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from harbor_ml import load_telemetry, summarize_telemetry  # noqa: E402


def main() -> None:
    """Display observations and deterministic summaries; no model is trained."""
    observations = load_telemetry(REPOSITORY_ROOT / "data/harbor_incident_telemetry.csv")

    print("Harbor Federal Credit Union — Telemetry Timeline\n")
    for observation in observations:
        print(
            f"{observation.timestamp:%H:%M}  "
            f"API {observation.api_latency_ms:<4} ms  "
            f"Vendor {observation.vendor_latency_ms:<4} ms  "
            f"Queue {observation.queue_depth:<3}  "
            f"Errors {observation.error_rate:.2%}"
        )

    # These are deterministic descriptions of the fixture, not ML predictions.
    summary = summarize_telemetry(observations)
    print("\nDescriptive statistics (not ML predictions)")
    print(f"Minimum API latency: {summary.minimum_api_latency_ms} ms")
    print(f"Maximum API latency: {summary.maximum_api_latency_ms} ms")
    print(f"Average API latency: {summary.average_api_latency_ms:.1f} ms")
    print(f"Minimum vendor latency: {summary.minimum_vendor_latency_ms} ms")
    print(f"Maximum vendor latency: {summary.maximum_vendor_latency_ms} ms")
    print(f"Queue growth: {summary.queue_growth}")
    print(f"Error-rate change: {summary.error_rate_change:.2%}")


if __name__ == "__main__":
    main()
