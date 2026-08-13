"""Generate Chapter 8's deterministic, fictional member-event fixture."""

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/harbor_member_events.csv"
SEED = 808

ARCHETYPES = {
    "browse_only": ["dashboard_viewed", "account_viewed", "statement_viewed"],
    "successful_transfer": ["dashboard_viewed", "transfer_started", "recipient_selected", "transfer_reviewed", "transfer_completed"],
    "incomplete_transfer": ["transfer_started", "recipient_selected", "transfer_reviewed"],
    "successful_verification": ["dashboard_viewed", "verification_started", "verification_completed"],
    "incomplete_verification": ["verification_started", "verification_abandoned"],
    "search_heavy": ["dashboard_viewed", "search_performed", "account_viewed", "search_performed", "statement_viewed"],
}


def journey(event: str) -> str:
    if event.startswith("transfer") or event == "recipient_selected": return "transfer"
    if event.startswith("verification"): return "verification"
    if event == "search_performed": return "search"
    return ""


def main() -> None:
    randomizer = random.Random(SEED)
    start = datetime(2026, 8, 13, 8, tzinfo=timezone.utc)
    rows = []
    kinds = list(ARCHETYPES)
    for offset in range(90):
        session_id = f"session-{1042 + offset}"
        channel = "web" if randomizer.random() < 0.56 else "mobile"
        kind = kinds[offset % len(kinds)]
        names = ["session_started", "login_completed", *ARCHETYPES[kind], "session_ended"]
        if offset % 17 == 0:
            names.insert(1, "login_failed")
        if offset % 19 == 0 and "transfer_started" in names:
            index = names.index("transfer_started") + 1
            names.insert(index, "transfer_failed")
            names.insert(index + 1, "transfer_started")
        timestamp = start + timedelta(minutes=offset * 11)
        for name in names:
            timestamp += timedelta(seconds=randomizer.randint(3, 28))
            rows.append({"timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                         "session_id": session_id, "event_name": name,
                         "channel": channel, "journey": journey(name)})
    with OUTPUT.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("timestamp", "session_id", "event_name", "channel", "journey"))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} events in 90 sessions to {OUTPUT}")


if __name__ == "__main__":
    main()
