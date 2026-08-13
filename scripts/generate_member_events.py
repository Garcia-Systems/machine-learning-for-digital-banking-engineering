"""Generate the deterministic, fictional Chapters 8 and 9 member-event fixture."""

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
    if event.startswith("transfer") or event == "recipient_selected":
        return "transfer"
    if event.startswith("verification"):
        return "verification"
    if event == "search_performed":
        return "search"
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

    # Chapter 9 adds noisy transfer histories to the same behavioral universe.
    # Future completion is sampled from overlapping synthetic relationships; no
    # single event determines the outcome.
    for offset in range(300):
        session_id = f"session-{2000 + offset}"
        channel = "mobile" if randomizer.random() < 0.48 else "web"
        prior_failure = randomizer.random() < 0.22
        errors = randomizer.choices((0, 1, 2), weights=(0.58, 0.30, 0.12))[0]
        searches = randomizer.choices((0, 1, 2), weights=(0.55, 0.31, 0.14))[0]
        help_opened = randomizer.random() < (0.18 + 0.13 * errors + 0.08 * searches)
        delay = randomizer.randint(18, 105) + errors * 22 + searches * 13
        risk_score = (
            -0.55 + 0.008 * (delay - 60) + 0.48 * errors + 0.30 * searches
            + 0.45 * help_opened + 0.28 * prior_failure + 0.20 * (channel == "mobile")
        )
        abandonment_probability = 1 / (1 + pow(2.718281828, -risk_score))
        abandoned = randomizer.random() < abandonment_probability
        names = ["session_started"]
        if prior_failure:
            names.append("login_failed")
        names.extend(["login_completed", "dashboard_viewed", "transfer_started"])
        pre_prediction = ["transfer_failed"] * errors + ["search_performed"] * searches
        if help_opened:
            pre_prediction.append("help_opened")
        randomizer.shuffle(pre_prediction)
        names.extend(pre_prediction)
        names.append("recipient_selected")
        if randomizer.random() < 0.72:
            names.append("transfer_reviewed")
        if not abandoned:
            names.append("transfer_completed")
        elif randomizer.random() < 0.18:
            names.append("help_opened")  # label-window event, never a feature
        names.append("session_ended")
        timestamp = start + timedelta(days=2, minutes=offset * 7)
        for name in names:
            if name == "recipient_selected":
                timestamp += timedelta(seconds=max(3, delay // max(1, len(pre_prediction) + 1)))
            else:
                timestamp += timedelta(seconds=randomizer.randint(3, 22))
            rows.append({"timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                         "session_id": session_id, "event_name": name,
                         "channel": channel, "journey": journey(name)})
    with OUTPUT.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("timestamp", "session_id", "event_name", "channel", "journey"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} events in 390 sessions to {OUTPUT}")


if __name__ == "__main__":
    main()
