"""Generate Chapter 15's deterministic, entirely fictional query observations."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/harbor_query_performance.csv"
SEED = 1515

FAMILIES = {
    "account_summary": 70, "transaction_history": 180, "member_search": 120,
    "statement_lookup": 95, "transfer_history": 150, "verification_audit": 210,
}
BANDS = {"tiny": 10, "small": 35, "medium": 120, "large": 330, "very_large": 720}


def main() -> None:
    rng = np.random.default_rng(SEED)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(1800):
        timestamp = start + timedelta(minutes=5 * index)
        family = rng.choice(list(FAMILIES), p=[.18, .25, .17, .14, .16, .10])
        band = rng.choice(list(BANDS), p=[.16, .27, .28, .20, .09])
        joins = int(rng.integers(0, 5)); filters = int(rng.integers(1, 7))
        sort = bool(rng.random() < .42); aggregation = bool(rng.random() < .30)
        grouping = bool(aggregation and rng.random() < .58)
        daily = max(0, np.sin((timestamp.hour - 6) * np.pi / 12))
        requests = max(80, 260 + 520 * daily + .045 * index + rng.normal(0, 55))
        connections = max(8, int(22 + requests / 18 + rng.normal(0, 8)))
        queue = max(0, int((connections - 50) * .45 + rng.normal(3, 5)))
        recent = max(25, 48 + connections * 1.7 + queue * 4.2 + rng.normal(0, 18))
        duration = (FAMILIES[family] + BANDS[band] + joins * 42 + filters * 7
                    + sort * 65 + aggregation * 75 + grouping * 55
                    + max(0, connections - 55) * 4 + queue * 10 + recent * .55
                    + max(0, requests - 600) * .22)
        if sort and band in ("large", "very_large"):
            duration += 150 + max(0, connections - 65) * 7
        if joins >= 3 and queue >= 15:
            duration += (joins - 2) * queue * 8
        duration = max(15, duration + rng.normal(0, 55 + .06 * duration))
        rows.append([timestamp.isoformat().replace("+00:00", "Z"), family, band, joins,
                     filters, str(sort).lower(), str(aggregation).lower(), str(grouping).lower(),
                     connections, queue, f"{recent:.2f}", f"{requests:.2f}", f"{duration:.2f}"])
    header = ["timestamp", "query_family", "rows_expected_band", "join_count", "filter_count",
              "uses_sort", "uses_aggregation", "uses_grouping", "current_db_connections",
              "current_queue_depth", "recent_db_latency_ms", "requests_per_minute",
              "query_duration_ms"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target); writer.writerow(header); writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic observations to {OUTPUT}")


if __name__ == "__main__":
    main()
