"""Run Chapter 0's deterministic system-health check."""

from pathlib import Path
import sys

# Keep the first example runnable from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harbor_ml import Observation, Thresholds, find_threshold_violations

NORMAL: Observation = {
    "api_latency_ms": 180,
    "error_rate": 0.004,
    "db_connections": 31,
    "vendor_latency_ms": 220,
}

INCIDENT: Observation = {
    "api_latency_ms": 2400,
    "error_rate": 0.087,
    "db_connections": 96,
    "vendor_latency_ms": 1900,
}

THRESHOLDS: Thresholds = {
    "api_latency_ms": 500,
    "error_rate": 0.05,
    "db_connections": 80,
    "vendor_latency_ms": 750,
}


def report(name: str, observation: Observation) -> None:
    """Print a readable threshold report for one observation."""
    violations = find_threshold_violations(observation, THRESHOLDS)
    print(f"{name}: {len(violations)} threshold violation(s)")
    for violation in violations:
        print(f"  - {violation}")


def main() -> None:
    """Compare a normal period and an incident period."""
    report("normal", NORMAL)
    report("incident", INCIDENT)


if __name__ == "__main__":
    main()

