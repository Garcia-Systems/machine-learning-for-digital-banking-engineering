"""Run Chapter 8's deterministic member behavior laboratory."""

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml import (  # noqa: E402
    calculate_behavior_statistics, calculate_transfer_funnel, count_events,
    group_events_by_session, load_member_events, summarize_sessions,
)


def main() -> None:
    events = load_member_events(ROOT / "data/harbor_member_events.csv")
    sessions = group_events_by_session(events)
    summaries = summarize_sessions(sessions)
    statistics = calculate_behavior_statistics(summaries.values())

    print("Harbor Federal Credit Union\nMember Behavior Laboratory")
    print(f"\nEvents loaded: {len(events)}\nSessions: {len(sessions)}")
    print("\nChannels:")
    for channel, count in sorted(Counter(summary.channel for summary in summaries.values()).items()):
        print(f"  {channel}: {count}")
    print("\nEvent counts:")
    for name, count in count_events(events).items():
        print(f"  {name:<26} {count}")

    print("\nExample sessions:")
    for session_id in list(sessions)[:3]:
        summary = summaries[session_id]
        print(f"\nSession: {session_id}\nChannel: {summary.channel}")
        print(f"Events: {summary.event_count}\nDuration: {summary.duration_seconds:.0f} seconds")
        print("Journey:")
        print("\n".join(f"  {event.event_name}" for event in sessions[session_id]))
        print(
            "Summary: "
            f"transfer_started={summary.transfer_started}, "
            f"transfer_completed={summary.transfer_completed}, "
            f"verification_started={summary.verification_started}, "
            f"verification_completed={summary.verification_completed}"
        )

    print("\nDescriptive session statistics (not ML predictions):")
    print(f"  total sessions: {statistics.total_sessions}")
    print(f"  average events per session: {statistics.average_events_per_session:.2f}")
    print(f"  average duration: {statistics.average_session_duration_seconds:.2f} seconds")
    print(f"  sessions containing transfer_started: {statistics.sessions_with_transfer_started}")
    print(f"  sessions containing transfer_completed: {statistics.sessions_with_transfer_completed}")
    print(f"  sessions containing verification_started: {statistics.sessions_with_verification_started}")
    print(f"  sessions containing verification_completed: {statistics.sessions_with_verification_completed}")
    print("\nTransfer funnel (sessions reaching each observed step):")
    for step, count in calculate_transfer_funnel(sessions).items():
        print(f"  {step:<26} {count}")
    print("\nThese fictional aggregates describe recorded behavior; they do not infer motive")
    print("and do not predict whether a future journey will complete.")


if __name__ == "__main__":
    main()
