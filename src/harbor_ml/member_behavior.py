"""Validated behavioral events and deterministic summaries for Chapter 8."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

EVENT_VOCABULARY = frozenset(
    {
        "session_started", "login_completed", "login_failed", "dashboard_viewed",
        "account_viewed", "search_performed", "transfer_started",
        "recipient_selected", "transfer_reviewed", "transfer_completed",
        "transfer_failed", "verification_started", "verification_completed",
        "verification_abandoned", "statement_viewed", "session_ended",
    }
)
VALID_CHANNELS = frozenset({"web", "mobile"})
EVENT_COLUMNS = ("timestamp", "session_id", "event_name", "channel", "journey")
TRANSFER_FUNNEL = (
    "transfer_started", "recipient_selected", "transfer_reviewed", "transfer_completed"
)
_SESSION_ID = re.compile(r"session-\d{4,}")


@dataclass(frozen=True)
class MemberEvent:
    timestamp: datetime
    session_id: str
    event_name: str
    channel: str
    journey: str | None


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    channel: str
    event_count: int
    duration_seconds: float
    first_event: str
    last_event: str
    transfer_started: bool
    transfer_completed: bool
    verification_started: bool
    verification_completed: bool
    search_count: int
    failed_login_count: int


@dataclass(frozen=True)
class BehaviorStatistics:
    total_sessions: int
    average_events_per_session: float
    average_session_duration_seconds: float
    sessions_with_transfer_started: int
    sessions_with_transfer_completed: int
    sessions_with_verification_started: int
    sessions_with_verification_completed: int


def _parse_timestamp(value: str, row_number: int) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"row {row_number}: invalid timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"row {row_number}: timestamp must include a timezone")
    return parsed


def load_member_events(path: str | Path) -> list[MemberEvent]:
    """Load the exact teaching schema, validate rows, and return stable time order."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != EVENT_COLUMNS:
            raise ValueError("member event dataset must use the documented event schema")
        rows = list(reader)
    if not rows:
        raise ValueError("member event dataset must contain events")

    events: list[MemberEvent] = []
    for number, row in enumerate(rows, 2):
        session_id = row["session_id"].strip()
        event_name = row["event_name"].strip()
        channel = row["channel"].strip()
        journey = row["journey"].strip() or None
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError(f"row {number}: invalid or missing session_id")
        if event_name not in EVENT_VOCABULARY:
            raise ValueError(f"row {number}: invalid event_name {event_name!r}")
        if channel not in VALID_CHANNELS:
            raise ValueError(f"row {number}: invalid channel {channel!r}")
        events.append(MemberEvent(
            _parse_timestamp(row["timestamp"].strip(), number), session_id,
            event_name, channel, journey,
        ))
    return sorted(events, key=lambda event: (event.session_id, event.timestamp, event.event_name))


def group_events_by_session(
    events: Iterable[MemberEvent],
) -> dict[str, tuple[MemberEvent, ...]]:
    """Group events by pseudonymous session ID and order each group by time."""
    grouped: dict[str, list[MemberEvent]] = defaultdict(list)
    for event in events:
        grouped[event.session_id].append(event)
    return {
        session_id: tuple(sorted(group, key=lambda event: (event.timestamp, event.event_name)))
        for session_id, group in sorted(grouped.items())
    }


def summarize_session(events: Sequence[MemberEvent]) -> SessionSummary:
    if not events:
        raise ValueError("cannot summarize an empty session")
    ordered = sorted(events, key=lambda event: (event.timestamp, event.event_name))
    session_ids = {event.session_id for event in ordered}
    channels = {event.channel for event in ordered}
    if len(session_ids) != 1 or len(channels) != 1:
        raise ValueError("a session summary requires one session_id and one channel")
    names = [event.event_name for event in ordered]
    return SessionSummary(
        session_id=ordered[0].session_id, channel=ordered[0].channel,
        event_count=len(ordered),
        duration_seconds=(ordered[-1].timestamp - ordered[0].timestamp).total_seconds(),
        first_event=names[0], last_event=names[-1],
        transfer_started="transfer_started" in names,
        transfer_completed="transfer_completed" in names,
        verification_started="verification_started" in names,
        verification_completed="verification_completed" in names,
        search_count=names.count("search_performed"),
        failed_login_count=names.count("login_failed"),
    )


def summarize_sessions(
    sessions: Mapping[str, Sequence[MemberEvent]],
) -> dict[str, SessionSummary]:
    return {session_id: summarize_session(events) for session_id, events in sessions.items()}


def count_events(events: Iterable[MemberEvent]) -> dict[str, int]:
    return dict(sorted(Counter(event.event_name for event in events).items()))


def calculate_transfer_funnel(
    sessions: Mapping[str, Sequence[MemberEvent]],
) -> dict[str, int]:
    """Count sessions containing each step; repeated events count only once."""
    return {
        step: sum(step in {event.event_name for event in events} for events in sessions.values())
        for step in TRANSFER_FUNNEL
    }


def calculate_behavior_statistics(
    summaries: Iterable[SessionSummary],
) -> BehaviorStatistics:
    values = list(summaries)
    if not values:
        raise ValueError("cannot calculate statistics without sessions")
    return BehaviorStatistics(
        total_sessions=len(values),
        average_events_per_session=sum(item.event_count for item in values) / len(values),
        average_session_duration_seconds=sum(item.duration_seconds for item in values) / len(values),
        sessions_with_transfer_started=sum(item.transfer_started for item in values),
        sessions_with_transfer_completed=sum(item.transfer_completed for item in values),
        sessions_with_verification_started=sum(item.verification_started for item in values),
        sessions_with_verification_completed=sum(item.verification_completed for item in values),
    )
