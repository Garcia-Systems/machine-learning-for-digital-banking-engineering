# Chapter 8 — Understanding Member Behavior

> Part III — Machine Learning for Member Digital Experiences

[Part III overview](README.md) · [Complete contents](../../CONTENTS.md) · [Previous: Chapter 7](../part-02-production-troubleshooting/chapter-07-predicting-integration-failures.md)

## Central question

> How can Harbor represent member interactions with digital banking applications in a structured way that can later support machine-learning analysis?

Harbor Federal Credit Union has used ML to help developers understand production behavior. Now the Digital Banking team asks: **Members are using Harbor's web and mobile applications, but how are they actually moving through the experience?**

```text
session_started → login_completed → account_viewed → transfer_started
    → recipient_selected → transfer_reviewed → transfer_completed

session_started → login_completed → transfer_started → recipient_selected
    → transfer_reviewed → session_ended
```

A full-stack developer can see different application behaviors. Prediction comes later. First Harbor needs a consistent answer to **what happened?** Front-end events, routes, API requests, workflow transitions, logs, timestamps, and database records are familiar raw materials; an event contract turns them into analyzable evidence.

All data in this chapter is deterministic, synthetic educational material. It does not describe a real credit union or establish real behavioral rates.

## Learning objectives

By the end, you should be able to:

1. distinguish operational telemetry from member behavior events;
2. define an analytics event;
3. explain sessions and journeys;
4. distinguish event names from properties;
5. order events by timestamp;
6. derive session-level summaries;
7. distinguish raw events from behavioral features;
8. identify privacy-sensitive fields and explain pseudonymous identifiers;
9. explain why instrumentation quality matters;
10. construct a small behavioral dataset; and
11. explain how behavior data may later support classification, clustering, and prediction.

## From Part II telemetry to Part III behavior

```text
OPERATIONAL TELEMETRY                 MEMBER BEHAVIOR EVENTS

api_latency_ms                        session_started
error_rate                            account_viewed
db_connections                        transfer_started
queue_depth                           transfer_completed
vendor_latency_ms                     verification_abandoned

How is the system behaving?           How is the member interacting?
```

Both streams can occur together:

```text
Member action:          transfer_started
Operational observation: POST /transfers latency = 186 ms
```

The first records an interaction; the second records system performance. They could later be joined under an approved, carefully designed question, but they must not be conceptually confused. Neither alone explains a member's motive.

## Anatomy of an analytics event

An **analytics event** is a structured record of an observed interaction at a time.

```python
{
    "timestamp": "2026-08-13T14:22:10Z",
    "session_id": "session-1042",
    "event_name": "transfer_started",
    "channel": "web",
}
```

Context can be nested properties:

```python
{
    "timestamp": "2026-08-13T14:22:18Z",
    "session_id": "session-1042",
    "event_name": "transfer_amount_entered",
    "channel": "web",
    "properties": {"amount_band": "100_to_499"},
}
```

The categorized band is preferable to an exact amount when the engineering question needs only coarse context. Do not collect balances, full amounts, account numbers, names, or addresses merely because the application can access them.

```text
event
  ├── name                 what happened
  ├── timestamp            when it was recorded
  ├── session identifier   grouping key
  └── properties           context, such as channel or transfer_type
```

`transfer_started` is the **event name**. `channel = web` and `transfer_type = internal` are properties. Four spellings—`transfer-start`, `transferStarted`, `begin_transfer`, and `Transfer Clicked`—fragment one concept. A stable documented `transfer_started` makes aggregation reliable.

### Harbor's modest vocabulary

```text
session_started          login_completed          login_failed
dashboard_viewed         account_viewed           search_performed
transfer_started         recipient_selected       transfer_reviewed
transfer_completed       transfer_failed          verification_started
verification_completed   verification_abandoned   statement_viewed
session_ended
```

A small vocabulary is teachable and governable. Adding an event should require a semantic definition, not merely a new UI click handler.

## Sessions and journeys

> A session is a bounded sequence of related member interactions.

```text
session-1042                         session-1043
14:22:01 session_started             15:04:02 session_started
14:22:05 login_completed             15:04:07 login_completed
14:22:11 dashboard_viewed            15:04:13 transfer_started
14:22:18 transfer_started            15:04:20 recipient_selected
14:22:25 recipient_selected          15:06:58 session_ended
14:22:33 transfer_reviewed
14:22:41 transfer_completed
14:22:46 session_ended
```

The second recorded sequence ends before a completion event. That observation does not establish why, and this chapter does not turn it into an abandonment target.

A **journey** is a meaningful sequence toward a goal:

```text
TRANSFER JOURNEY                     IDENTITY VERIFICATION JOURNEY
transfer_started                     verification_started
       ↓                                    ↓
recipient_selected                   verification_completed
       ↓                             or verification_abandoned
transfer_reviewed
       ↓
transfer_completed
```

A session is a technical grouping boundary; a journey is a business workflow. One session may include browsing, a transfer, and verification—multiple journeys. Conversely, product rules might someday define a journey spanning sessions. The fixture uses `web` and `mobile`; channel differences may be worth describing, but unequal instrumentation can masquerade as behavioral difference.

`session-1042` is pseudonymous and useful for grouping. It is not direct identity, but **pseudonymous identifiers can still be sensitive when combined with other information**. Pseudonymization does not eliminate access control, retention, consent, governance, or privacy obligations.

## From events to deterministic summaries

```text
RAW EVENTS → GROUP BY SESSION → ORDER BY TIME → DERIVE SESSION SUMMARY
```

The module represents immutable records with `MemberEvent` and `SessionSummary`. The loader parses timezone-aware timestamps, enforces the vocabulary, checks `web`/`mobile`, validates session IDs, and returns stable order. `group_events_by_session` then chronologically orders every session.

```python
from harbor_ml import group_events_by_session, load_member_events, summarize_sessions

events = load_member_events("data/harbor_member_events.csv")
sessions = group_events_by_session(events)
summaries = summarize_sessions(sessions)
```

A summary contains:

```text
session_id                 event_count
channel                    duration_seconds
first_event                last_event
transfer_started           transfer_completed
verification_started       verification_completed
search_count               failed_login_count
```

Duration is last timestamp minus first timestamp. Counts and flags are mechanically derived from recorded events; they are descriptive facts under the event contract, **not ML predictions**.

### Raw values, derived values, and possible features

```text
RAW EVENT                         DERIVED VALUE
event_name=transfer_completed     session_duration_seconds=142
timestamp=...                     transfer_event_count=4

POSSIBLE LATER ML FEATURE
steps_before_transfer_exit
```

A derived value is not automatically a suitable ML feature. Feature design depends on a later prediction question, prediction time, causal assumptions, and leakage analysis. Chapter 8 builds no `abandoned` target.

## Time, missingness, and duplicates

`transfer_completed` followed in time by its first `transfer_started` is invalid under the simple transfer contract. Behavioral analysis depends on timestamps, ordering, session boundaries, duplicate policy, and missing-event interpretation.

```text
EVENT LOG → VALIDATE → SORT → GROUP → ANALYZE JOURNEY
```

Duplicate `transfer_started` records could arise from a client retry, duplicate send, refresh, or instrumentation bug. They might also represent a legitimate repeat attempt. The chapter code preserves duplicates rather than silently guessing; a production contract would ideally carry an event ID and an explicit deduplication policy.

If `transfer_started → recipient_selected → transfer_completed` lacks `transfer_reviewed`, possible explanations include an instrumentation gap, alternate UI path, delivery failure, or actual workflow difference. The log alone cannot select one explanation. ML cannot automatically repair unclear event semantics.

Sequence itself carries information. A successful straight path:

```text
transfer_started → recipient_selected → transfer_reviewed → transfer_completed
```

differs from a retry path:

```text
transfer_started → transfer_failed → transfer_started → transfer_completed
```

Reducing either to a set of event names loses order and repetition.

## Instrumentation is an API contract

```text
Frontend → analytics event contract → collection pipeline → analysis / ML
```

Renaming `transfer_completed` to `transfer_success` without coordination is like an unversioned API schema change: downstream code may not crash; it may silently undercount. Treat event design like interface design.

> **ML quality cannot exceed event-definition quality.**

```text
EVENT CONTRACT
      ├── stable name
      ├── defined semantics
      ├── required properties
      ├── timestamp rule
      └── versioning
```

Specify precisely when `transfer_started` fires, who owns it, whether retries emit it again, client-time versus server-time rules, and supported versions. If it means a button click on web but a validated workflow creation on mobile, the combined dataset is ambiguous. If only one client emits `verification_abandoned`, channel comparisons are misleading. Contract tests, schema validation, staged migrations, monitoring for unknown events, and producer/consumer coordination are as relevant here as for REST or message APIs.

## The fictional fixture and executable laboratory

`data/harbor_member_events.csv` has the intentionally narrow schema:

```text
timestamp,session_id,event_name,channel,journey
```

The generator uses fixed seed `808`, 90 fictional sessions, web/mobile variation, six internal generation archetypes, occasional search repetition, incomplete recorded journeys, and valid retry sequences. Archetypes are generator mechanics, not ML labels. Regenerate and run:

```bash
python scripts/generate_member_events.py
python examples/chapter_08_member_behavior.py
```

The laboratory loads rather than hard-codes event, channel, summary, and funnel counts. It prints sample sequences and descriptive aggregates: total sessions, mean event count, mean duration, and sessions containing observed transfer or verification markers.

### A transfer funnel

For the committed fixture, the program calculates:

```text
transfer_started          30
recipient_selected        30
transfer_reviewed         30
transfer_completed        15
```

A funnel is a descriptive count of sessions containing each step. It does not predict which journey will leave the recorded path. Counts can mislead when members repeat steps, one session contains multiple transfers, tracking is incomplete, alternate workflows exist, or session boundaries are wrong. This simplified funnel counts each session at most once per step; it does not assert strict step order.

## Privacy minimization

Behavioral analytics becomes privacy-sensitive quickly. Do not place these in a basic event stream:

```text
full_name                 email_address             account_number
exact_balance             SSN                       authentication_secret
raw document contents     precise location
```

Prefer only what the approved question requires:

```text
session_id   event_name   timestamp   channel   journey   coarse properties
```

Even innocuous-looking events can reveal sensitive patterns when linked over time or joined to other datasets. Apply purpose limitation, short appropriate retention, least-privilege access, encryption, auditability, deletion procedures, and review of linkage risk. Never put authentication secrets or raw identity documents into analytics.

```text
COLLECT WHAT THE ENGINEERING QUESTION REQUIRES
NOT
EVERYTHING THE APPLICATION CAN POSSIBLY OBSERVE
```

The test asserting the fixture's explicit schema is a teaching guardrail, not magical privacy detection or a substitute for governance and review.

## Where ML may eventually fit

Once reliable events become sessions and journey summaries, later questions could use classification (which known outcome category?), clustering (which recurring sequence patterns?), or prediction (what is likely next?). Those techniques require a separately justified unit of analysis, labels, evaluation, prediction-time contract, and protections against harmful use. The aggregates here make no predictive claim.

## Exercises

### 1 — Operational or behavioral?

Classify `api_latency_ms`, `transfer_started`, `db_connections`, `account_viewed`, `vendor_latency_ms`, and `verification_completed`. Explain which question each category answers.

### 2 — Event or property?

For `event_name = transfer_started` and `channel = mobile`, identify what happened and what supplies context.

### 3 — Session interpretation

Given `session_started → login_completed → transfer_started → recipient_selected → session_ended`, list only what is directly recorded and what is unknown. “The member hated the transfer experience” is unsupported: no event records motive or emotion.

### 4 — Instrumentation bug

Web emits `transfer_completed`; mobile emits `transfer_success` for the same action. Why does this fragment aggregates? Propose a contract-compatible migration.

### 5 — Privacy

Choose fields to exclude: `session_id`, `event_name`, `timestamp`, `account_number`, `channel`, `SSN`, `journey`, `access_token`. Explain why pseudonymous `session_id` still needs protection.

### Coding exercise — `help_opened`

1. add `help_opened` to `EVENT_VOCABULARY`;
2. add it to selected synthetic sessions;
3. regenerate the fixture;
4. count sessions containing it;
5. add deterministic tests; and
6. explain why opening help does not prove confusion—help may be exploratory, accidental, or task-related, and the event records no emotion.

## Key takeaways

1. Behavioral events describe recorded actions in a digital application.
2. Operational telemetry and member behavior answer different questions.
3. Events need stable names and clear semantics.
4. Sessions group related interactions; journeys describe goal-oriented sequences.
5. Timestamps, boundaries, duplicates, missingness, and order matter.
6. Raw events can become deterministic session summaries.
7. Behavioral summaries are not automatically ML features.
8. Poor instrumentation produces poor analytics and poor ML.
9. Pseudonymous data remains potentially sensitive.
10. Collect behavior data with strong privacy minimization.

## What comes next

Chapter 9 — **Predicting Digital Journey Abandonment** will ask:

> Can Harbor identify, before the journey ends, whether a digital workflow is at elevated risk of being abandoned?

```text
member events → sessions → journey summaries

KNOWN SO FAR
transfer_started, recipient_selected, time_elapsed, error_count,
help_opened, channel
        │
        ▼
      MODEL
        │
        ▼
Will this journey complete?
```

A future target might be `journey_completed` or `journey_abandoned`, but it is not implemented here. Chapter 9 must define prediction time carefully: `transfer_completed` cannot be a feature for predicting whether that transfer will complete. That would leak the answer.

[Back to the Part III overview](README.md) · [Back to complete contents](../../CONTENTS.md)
