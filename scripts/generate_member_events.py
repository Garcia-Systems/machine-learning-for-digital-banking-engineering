"""Generate the deterministic, fictional Chapters 8–11 member-event fixture."""

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
    if event in {"landing_page_viewed", "product_details_viewed", "eligibility_info_viewed", "application_started"}:
        return "horizon_savings"
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
                         "channel": channel, "journey": journey(name),
                         "landing_source": "", "device_category": ""})

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
                         "channel": channel, "journey": journey(name),
                         "landing_source": "", "device_category": ""})

    # Chapter 10 adds product-information sessions. Relationships overlap and
    # include random outcome noise; all categories and effects are fictional.
    sources = ("direct", "search", "email_campaign", "internal_navigation")
    for offset in range(500):
        session_id = f"session-{3000 + offset}"
        channel = "mobile" if randomizer.random() < 0.49 else "web"
        device = (randomizer.choices(("phone", "tablet"), (0.87, 0.13))[0]
                  if channel == "mobile" else randomizer.choices(("desktop", "tablet"), (0.91, 0.09))[0])
        source = randomizer.choices(sources, (0.27, 0.29, 0.18, 0.26))[0]
        failures = int(randomizer.random() < 0.12)
        searches = randomizer.choices((0, 1, 2, 3), (0.58, 0.25, 0.12, 0.05))[0]
        helps = randomizer.choices((0, 1, 2), (0.76, 0.19, 0.05))[0]
        delay = randomizer.randint(25, 105) + 18 * searches + 25 * helps + 20 * failures
        score = (-0.85 - 0.007 * (delay - 65) - 0.31 * searches - 0.35 * helps
                 - 0.34 * failures + 0.22 * (source == "internal_navigation")
                 + 0.14 * (source == "email_campaign") - 0.14 * (device == "phone"))
        probability = 1 / (1 + pow(2.718281828, -score))
        converted = randomizer.random() < probability
        prefix = ["session_started"]
        if failures:
            prefix.append("login_failed")
        prefix.append("landing_page_viewed")
        friction = ["search_performed"] * searches + ["help_opened"] * helps
        randomizer.shuffle(friction)
        prefix.extend(friction)
        prefix.append("product_details_viewed")
        suffix = []
        if randomizer.random() < 0.62:
            suffix.append("eligibility_info_viewed")
        if converted:
            suffix.append("application_started")
        elif randomizer.random() < 0.15:
            suffix.append("search_performed")  # future, never a feature
        names = [*prefix, *suffix, "session_ended"]
        timestamp = start + timedelta(days=5, minutes=offset * 5)
        for name in names:
            timestamp += timedelta(seconds=max(3, delay // max(1, len(prefix) - 1))
                                   if name in prefix[1:] else randomizer.randint(4, 24))
            rows.append({"timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                         "session_id": session_id, "event_name": name,
                         "channel": channel, "journey": journey(name),
                         "landing_source": source, "device_category": device})

    # Chapter 11 adds session-pattern variation to the same event universe.
    # ``kind`` is generation metadata only: it is deliberately not written to
    # the fixture and can never become a clustering target.
    segmentation_patterns = (
        "quick_account_check", "transfer_focused", "statement_research", "help_search_heavy",
    )
    for offset in range(400):
        session_id = f"session-{4000 + offset}"
        channel = "mobile" if randomizer.random() < 0.51 else "web"
        kind = segmentation_patterns[offset % len(segmentation_patterns)]
        names = ["session_started", "login_completed", "dashboard_viewed"]
        if kind == "quick_account_check":
            names += ["account_viewed"] * randomizer.randint(2, 5)
            if randomizer.random() < 0.18:
                names.append("statement_viewed")
        elif kind == "transfer_focused":
            # Repeated transfer interactions provide a count signal without
            # creating Chapter 9's recipient-selection prediction point.
            names += ["account_viewed", "transfer_started", "transfer_reviewed"]
            if randomizer.random() < 0.76:
                names.append("transfer_completed")
            if randomizer.random() < 0.20:
                names.append("search_performed")
        elif kind == "statement_research":
            names += ["account_viewed"] * randomizer.randint(1, 3)
            names += ["statement_viewed"] * randomizer.randint(2, 5)
            if randomizer.random() < 0.25:
                names.append("search_performed")
        else:
            names += ["search_performed"] * randomizer.randint(2, 5)
            names += ["help_opened"] * randomizer.randint(1, 3)
            if randomizer.random() < 0.35:
                names.append("account_viewed")
            randomizer.shuffle(names[3:])
        if randomizer.random() < 0.09:
            names.append("verification_started")
            if randomizer.random() < 0.72:
                names.append("verification_completed")
        names.append("session_ended")
        timestamp = start + timedelta(days=10, minutes=offset * 6)
        for name in names:
            maximum_delay = 50 if kind == "help_search_heavy" else 28
            timestamp += timedelta(seconds=randomizer.randint(4, maximum_delay))
            rows.append({"timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                         "session_id": session_id, "event_name": name,
                         "channel": channel, "journey": journey(name),
                         "landing_source": "", "device_category": ""})
    with OUTPUT.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("timestamp", "session_id", "event_name", "channel", "journey",
                        "landing_source", "device_category"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} events in 1290 sessions to {OUTPUT}")


if __name__ == "__main__":
    main()
