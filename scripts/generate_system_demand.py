"""Generate Chapter 14's deterministic fictional system-demand fixture."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SEED = 1414
OUTPUT = Path(__file__).resolve().parents[1] / "data/harbor_system_demand.csv"


def main() -> None:
    rng = np.random.default_rng(SEED)
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    rows = ["timestamp,requests_per_minute,api_latency_ms,error_rate,queue_depth"]
    for index in range(20 * 24 * 12):
        timestamp = start + timedelta(minutes=5 * index)
        hour = timestamp.hour + timestamp.minute / 60
        # Smooth daily peaks plus a smaller evening shoulder.
        daytime = 500 * np.exp(-((hour - 11.0) / 4.0) ** 2)
        evening = 180 * np.exp(-((hour - 19.0) / 3.0) ** 2)
        weekday = 1.0 if timestamp.weekday() < 5 else .72
        event = 0.0
        if timestamp.day in {9, 15, 20} and 9.5 <= hour <= 11:
            event = 120 * np.sin(np.pi * (hour - 9.5) / 1.5)
        requests = max(90, (230 + daytime + evening) * weekday + event + rng.normal(0, 24))
        latency = max(65, 92 + .105 * requests + .025 * max(requests - 700, 0) + rng.normal(0, 11))
        error_rate = max(0, .0015 + max(requests - 650, 0) / 115_000 + rng.normal(0, .0007))
        queue = max(0, round((requests - 500) / 30 + rng.normal(2, 2.2)))
        rows.append(f"{timestamp.isoformat().replace('+00:00', 'Z')},{requests:.2f},"
                    f"{latency:.2f},{error_rate:.5f},{queue}")
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
