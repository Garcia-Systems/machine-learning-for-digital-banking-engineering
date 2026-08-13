"""Generate Chapter 12's deterministic, fictional transaction observations."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 1212
ROWS = 800
ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "debit_purchase": {
        "weight": 42, "channels": ("mobile", "web", "atm"),
        "amounts": ("under_25", "25_to_99", "100_to_499", "500_to_999"),
        "amount_weights": (30, 48, 19, 3), "hour": 15, "spread": 4.5,
    },
    "bill_payment": {
        "weight": 16, "channels": ("web", "mobile", "branch_assisted"),
        "amounts": ("25_to_99", "100_to_499", "500_to_999", "1000_to_2499"),
        "amount_weights": (12, 50, 30, 8), "hour": 13, "spread": 4,
    },
    "internal_transfer": {
        "weight": 14, "channels": ("mobile", "web", "branch_assisted"),
        "amounts": ("25_to_99", "100_to_499", "500_to_999", "1000_to_2499", "2500_plus"),
        "amount_weights": (12, 38, 28, 17, 5), "hour": 14, "spread": 4,
    },
    "external_transfer": {
        "weight": 9, "channels": ("web", "mobile", "branch_assisted"),
        "amounts": ("100_to_499", "500_to_999", "1000_to_2499", "2500_plus"),
        "amount_weights": (25, 30, 28, 17), "hour": 13, "spread": 3.5,
    },
    "atm_withdrawal": {
        "weight": 11, "channels": ("atm",),
        "amounts": ("under_25", "25_to_99", "100_to_499", "500_to_999"),
        "amount_weights": (8, 40, 47, 5), "hour": 14, "spread": 4.5,
    },
    "deposit": {
        "weight": 8, "channels": ("mobile", "atm", "branch_assisted"),
        "amounts": ("under_25", "25_to_99", "100_to_499", "500_to_999", "1000_to_2499", "2500_plus"),
        "amount_weights": (4, 12, 38, 25, 15, 6), "hour": 12, "spread": 4,
    },
}


def main() -> None:
    rng = random.Random(SEED)
    names = list(PROFILES)
    weights = [PROFILES[name]["weight"] for name in names]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fields = ["timestamp", "transaction_type", "channel", "amount_band", "hour_of_day",
              "recent_transaction_count", "minutes_since_previous_transaction",
              "device_change", "distance_from_recent_pattern"]
    rows = []
    for index in range(ROWS):
        kind = rng.choices(names, weights=weights)[0]
        profile = PROFILES[kind]
        hour = max(0, min(23, round(rng.gauss(profile["hour"], profile["spread"]))))
        recent = min(9, max(0, round(rng.gauss(2.0, 1.35))))
        device_change = rng.random() < (0.07 + (0.02 if recent > 4 else 0))
        distance = min(0.98, max(0.01, rng.betavariate(2.0, 7.5) + (0.12 if device_change else 0)))
        minutes = min(1440.0, max(1.0, rng.lognormvariate(4.5, 0.85) / (1 + recent * 0.12)))
        timestamp = start + timedelta(days=index // 4, hours=hour, minutes=rng.randrange(60))
        rows.append({
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "transaction_type": kind,
            "channel": rng.choice(profile["channels"]),
            "amount_band": rng.choices(profile["amounts"], profile["amount_weights"])[0],
            "hour_of_day": hour,
            "recent_transaction_count": recent,
            "minutes_since_previous_transaction": f"{minutes:.2f}",
            "device_change": str(device_change).lower(),
            "distance_from_recent_pattern": f"{distance:.3f}",
        })
    destination = ROOT / "data/harbor_transaction_observations.csv"
    with destination.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} observations to {destination}")


if __name__ == "__main__":
    main()
