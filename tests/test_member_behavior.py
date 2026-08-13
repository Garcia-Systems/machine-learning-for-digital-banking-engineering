import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harbor_ml import (
    EVENT_COLUMNS, EVENT_VOCABULARY, VALID_CHANNELS, MemberEvent,
    calculate_behavior_statistics, calculate_transfer_funnel, count_events,
    group_events_by_session, load_member_events, summarize_session, summarize_sessions,
)

DATASET = Path(__file__).parents[1] / "data/harbor_member_events.csv"


@pytest.fixture(scope="module")
def events(): return load_member_events(DATASET)


@pytest.fixture(scope="module")
def sessions(events): return group_events_by_session(events)


def test_fixture_loads_parsed_valid_minimal_events(events):
    assert len(events) == 9808
    assert len({event.session_id for event in events}) == 1290
    assert all(event.timestamp.tzinfo is not None for event in events)
    assert {event.event_name for event in events} <= EVENT_VOCABULARY
    assert {event.channel for event in events} == VALID_CHANNELS


def test_fixture_schema_excludes_direct_identity_fields():
    with DATASET.open(newline="", encoding="utf-8") as source:
        fields = tuple(csv.DictReader(source).fieldnames or ())
    assert fields == EVENT_COLUMNS
    assert not {"full_name", "email_address", "account_number", "ssn", "access_token", "exact_balance"} & set(fields)


def test_grouping_is_chronological_and_deterministic(events, sessions):
    again = group_events_by_session(reversed(events))
    assert sessions == again
    assert list(sessions) == sorted(sessions)
    assert all(list(group) == sorted(group, key=lambda item: (item.timestamp, item.event_name)) for group in sessions.values())


def test_session_summary_counts_duration_and_observed_flags(sessions):
    summary = summarize_session(sessions["session-1043"])
    group = sessions["session-1043"]
    assert summary.event_count == len(group)
    assert summary.duration_seconds == (group[-1].timestamp - group[0].timestamp).total_seconds()
    assert (summary.first_event, summary.last_event) == ("session_started", "session_ended")
    assert summary.transfer_started and summary.transfer_completed
    assert not summary.verification_started and not summary.verification_completed


def test_aggregate_counts_funnel_and_statistics(events, sessions):
    counts = count_events(events)
    assert sum(counts.values()) == len(events)
    assert counts["session_started"] == counts["session_ended"] == 1290
    funnel = calculate_transfer_funnel(sessions)
    assert funnel == {"transfer_started": 430, "recipient_selected": 330, "transfer_reviewed": 345, "transfer_completed": 220}
    summaries = summarize_sessions(sessions)
    stats = calculate_behavior_statistics(summaries.values())
    assert stats.total_sessions == 1290
    assert stats.sessions_with_transfer_started == 430
    assert stats.sessions_with_transfer_completed == 220
    assert stats.sessions_with_verification_started == 67
    assert stats.sessions_with_verification_completed == 41


@pytest.mark.parametrize(
    "row,message",
    [
        ("bad,session-9999,session_started,web,", "invalid timestamp"),
        ("2026-08-13T00:00:00Z,,session_started,web,", "session_id"),
        ("2026-08-13T00:00:00Z,session-9999,unknown,web,", "event_name"),
        ("2026-08-13T00:00:00Z,session-9999,session_started,kiosk,", "channel"),
    ],
)
def test_malformed_rows_produce_clear_errors(tmp_path, row, message):
    path = tmp_path / "events.csv"
    path.write_text(",".join(EVENT_COLUMNS) + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message): load_member_events(path)


def test_duplicate_events_are_preserved_for_explicit_downstream_policy():
    timestamp = datetime(2026, 8, 13, tzinfo=timezone.utc)
    event = MemberEvent(timestamp, "session-9999", "transfer_started", "web", "transfer")
    grouped = group_events_by_session([event, event])
    assert len(grouped["session-9999"]) == 2
    assert summarize_session(grouped["session-9999"]).event_count == 2


def test_empty_and_mixed_session_summaries_are_rejected():
    with pytest.raises(ValueError, match="empty"): summarize_session([])
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="one session_id"):
        summarize_session([MemberEvent(now, "session-9998", "session_started", "web", None), MemberEvent(now, "session-9999", "session_ended", "web", None)])
