"""Generate Chapter 13's deterministic, fictional review-routing fixture."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from math import exp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 1313


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    types = ("debit_purchase", "bill_payment", "internal_transfer", "external_transfer",
             "atm_withdrawal", "deposit")
    channels = ("web", "mobile", "atm", "branch_assisted")
    bands = ("under_25", "25_to_99", "100_to_499", "500_to_999",
             "1000_to_2499", "2500_plus")
    rows = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for index in range(1_000):
        transaction_type = str(rng.choice(types, p=(.31, .18, .18, .13, .12, .08)))
        channel = str(rng.choice(channels, p=(.25, .52, .12, .11)))
        amount_band = str(rng.choice(bands, p=(.20, .28, .27, .12, .08, .05)))
        hour = int(rng.integers(0, 24))
        count = int(min(rng.poisson(2.4), 12))
        minutes = round(float(min(rng.exponential(42), 360)), 2)
        changed = bool(rng.random() < .13)
        distance = round(float(rng.beta(2.0, 5.0)), 3)
        failures = int(min(rng.poisson(.35), 5))
        # Overlapping educational tendencies plus irreducible noise. No single value
        # determines the process label.
        logit = (-2.45 + 2.35 * distance + .17 * count + .62 * failures
                 + .48 * changed * (distance > .35)
                 + .52 * (transaction_type == "external_transfer" and channel == "web")
                 + .42 * (hour < 5 and count >= 3)
                 - .28 * (channel == "branch_assisted")
                 + rng.normal(0, .45))
        label = int(rng.random() < 1 / (1 + exp(-logit)))
        rows.append({
            "timestamp": (start + timedelta(minutes=17 * index)).isoformat().replace("+00:00", "Z"),
            "transaction_type": transaction_type, "channel": channel,
            "amount_band": amount_band, "hour_of_day": hour,
            "recent_transaction_count": count,
            "minutes_since_previous_transaction": f"{minutes:.2f}",
            "device_change": str(changed).lower(),
            "distance_from_recent_pattern": f"{distance:.3f}",
            "recent_failed_transaction_count": failures,
            "manual_review_required": label,
        })
    path = ROOT / "data/harbor_review_routing.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}; reviewed={sum(r['manual_review_required'] for r in rows)}")


if __name__ == "__main__":
    main()
