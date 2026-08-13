"""Generate Chapter 7's deterministic fictional integration-request fixture.

The rules and seed are educational only. They use no real financial, member,
vendor, or production data and are not estimates of real failure behavior.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from math import exp
from pathlib import Path
from random import Random

SEED = 707
ROWS = 600
OUTPUT = Path(__file__).resolve().parents[1] / "data/harbor_integration_requests.csv"
INTEGRATIONS = (
    ("ClearVerify", "identity_verify"),
    ("ClearVerify", "identity_document_upload"),
    ("Northstar Payments", "transfer_submit"),
    ("Northstar Payments", "transfer_status"),
    ("HarborLink Core Gateway", "account_summary"),
    ("HarborLink Core Gateway", "transaction_history"),
    ("BlueCurrent Documents", "statement_fetch"),
    ("BlueCurrent Documents", "notice_fetch"),
)


def generate(path: Path = OUTPUT) -> None:
    """Write stable observations whose overlapping labels include random noise."""
    rng = Random(SEED)
    start = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)
    rows = []
    for index in range(ROWS):
        vendor, endpoint = INTEGRATIONS[index % len(INTEGRATIONS)]
        timestamp = start + timedelta(minutes=3 * index)
        hour = timestamp.hour
        latency = max(45, round(rng.gauss(310, 170)))
        error_rate = min(0.28, max(0.001, rng.betavariate(1.4, 14)))
        queue = max(0, round(rng.gauss(30, 27)))
        retry = rng.choices((0, 1, 2, 3), weights=(74, 17, 7, 2))[0]
        size = max(250, round(rng.lognormvariate(7.4, 0.72)))

        # Risk comes mostly from combinations of request-time conditions. Small
        # interaction offsets prevent vendors from becoming deterministic labels.
        interaction = {
            "identity_document_upload": 0.18 if size > 3000 else -0.06,
            "transfer_submit": 0.14 if retry else -0.04,
            "transaction_history": 0.12 if queue > 45 else -0.03,
            "statement_fetch": 0.16 if size > 3500 else -0.05,
        }.get(endpoint, 0.0)
        log_odds = (
            -3.45
            + 0.0032 * (latency - 200)
            + 8.5 * error_rate
            + 0.018 * queue
            + 0.55 * retry
            + 0.00010 * max(0, size - 1800)
            + interaction
            + rng.gauss(0, 0.38)
        )
        probability = 1 / (1 + exp(-log_odds))
        failed = int(rng.random() < probability)
        rows.append(
            {
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "vendor": vendor,
                "endpoint": endpoint,
                "recent_vendor_latency_ms": latency,
                "recent_vendor_error_rate": f"{error_rate:.4f}",
                "queue_depth": queue,
                "retry_count": retry,
                "request_size_bytes": size,
                "hour_of_day": hour,
                "request_failed": failed,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate()
