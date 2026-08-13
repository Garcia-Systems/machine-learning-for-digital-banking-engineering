# Chapter 1 — The Digital Banking Data Landscape

[← Part I contents](README.md) · [Book contents](../../CONTENTS.md) · [Chapter 2 →](chapter-02-from-engineering-problem-to-ml-problem.md)

> **Central question:** Before a machine-learning system can recognize patterns in a digital banking system, what can the system actually observe?

## Opening scenario: evidence across Harbor

The intermittent transfer failures from Chapter 0 continue at the entirely fictional **Harbor Federal Credit Union**. The application is not one process with one explanation:

```text
MEMBER
   │
   ▼
WEB / MOBILE
   │
   ▼
HARBOR DIGITAL BANKING APPLICATION
   │
   ├────────► DATABASE
   ├────────► CORE BANKING ADAPTER
   ├────────► IDENTITY VENDOR
   ├────────► TRANSFER SERVICE
   └────────► OTHER FINTECH APIs
```

Each component leaves different evidence. In an ordinary period a developer might see:

```text
HTTP request                 Database query
GET /api/accounts            SELECT ...
status = 200                 duration = 37 ms
 duration = 184 ms            connections = 29

Vendor request               Application event       System metric
POST /verify                 member.login.completed  queue_depth = 12
status = 200
 duration = 241 ms
```

During the incident that evidence changes:

```text
HTTP request                 Vendor request           System metric
GET /api/accounts            POST /verify             queue_depth = 147
status = 500                 status = 503
 duration = 2310 ms           duration = 1870 ms
```

No item alone necessarily explains the failure. The useful information may be in the **relationships between observations**—and even a strong relationship is evidence to investigate, not automatic proof of a cause.

## Learning objectives

By the end of this chapter, you should be able to:

1. distinguish logs, metrics, traces, events, database records, and API observations;
2. explain what an observation represents and identify useful operational variables;
3. distinguish raw observations, derived values, and model features;
4. recognize numerical, categorical, boolean, temporal, and identifier fields;
5. explain why timestamps are essential to correlation and why correlation is not causation;
6. recognize sensitive or inappropriate inputs;
7. create and load a structured telemetry observation; and
8. explain how observations could later become features without training a model yet.

## What is an observation?

An **observation** is a recorded view of something measurable at a defined time and scope. It might represent one HTTP request, one trace span, one event, or an aggregate one-minute system window. Its meaning depends on units and context: `184` is useless without knowing that it is API latency in milliseconds; an error rate needs a time window and population.

Observability sources are complementary rather than interchangeable:

- **Logs** record messages and context emitted by code.
- **Metrics** quantify behavior, usually across time.
- **Traces** connect work performed across components for one flow.
- **Events** state that a defined domain or application occurrence happened.
- **Database observations** describe data-store operation and resource pressure.
- **API observations** describe calls across service boundaries.

A database record can also be an observation, but this chapter uses engineering telemetry—not member balances, transactions, or other financial records.

## The Harbor telemetry landscape

### Application logs

```text
2026-08-12T10:14:31Z INFO transfer.request.started request_id=req-8127
2026-08-12T10:14:32Z WARN vendor.timeout vendor=ClearVerify duration_ms=1804
2026-08-12T10:14:32Z ERROR transfer.request.failed request_id=req-8127
```

`ClearVerify` is fictional. An unstructured message such as `verification seems slow again` is readable but requires ambiguous text parsing. A structured record exposes stable fields such as `timestamp`, `level`, `event_name`, `vendor`, and `duration_ms`. Those fields can be filtered, validated, grouped, and joined consistently. Structure does not guarantee truth: clocks, instrumentation, naming, and missing records can still be wrong.

### Metrics

```text
api_latency_ms = 184          error_rate = 0.004
db_connections = 31          queue_depth = 12
vendor_latency_ms = 220       requests_per_minute = 840
```

Conceptually, a **gauge** is a current value that can rise or fall (`queue_depth`); a **counter** accumulates occurrences (total requests); a **rate** expresses change per interval (`requests_per_minute`); and a **distribution** retains how values such as latency vary so that averages do not hide slow requests. These semantics matter more here than any particular monitoring product.

### Traces

A distributed trace connects spans belonging to one request and shows where elapsed time was spent:

```text
Member request                         Member request
     │                                      │
     ▼                                      ▼
GET /accounts             184 ms       GET /accounts            2380 ms
     ├── authentication    12 ms             ├── authentication    14 ms
     ├── database query    38 ms             ├── database query    41 ms
     └── core banking API 109 ms             └── core banking API 2291 ms ← investigate
```

The incident trace directs attention to a boundary that consumed most time. It still does not prove why that span was slow: the adapter, network, upstream service, retries, or resource contention could contribute.

### Application events

```text
member.login.completed       member.login.failed
account.viewed                transfer.started
transfer.completed            transfer.failed
verification.started          verification.completed
verification.failed
```

An event asserts that a defined thing happened, with a stable name and contract. A log is an emitted diagnostic message and may be rewritten or omitted. Events can themselves be logged, but treating every arbitrary log line as a reliable business event confuses those contracts.

### Database observations

```text
active_connections   query_duration_ms   rows_examined
lock_wait_ms         transaction_duration_ms
```

These variables reveal connection pressure, expensive access paths, contention, and long transactions. Aggregate or query-category telemetry is usually sufficient for an engineering question; copying actual member financial records into an operational dataset is neither necessary nor appropriate.

### Vendor and API observations

```text
vendor       endpoint       status_code       latency_ms
retry_count  timeout        response_category
```

External dependencies are part of the member-visible path even though Harbor does not operate them. Boundary measurements distinguish time spent locally from time awaiting a fictional provider, and statuses, retries, and timeouts show failure behavior. Harbor must instrument its side because an upstream dashboard may use different clocks, sampling, or definitions.

## Architecture: evidence becomes telemetry

```text
                     HARBOR DIGITAL BANKING

 Member
   │
   ▼
Web / Mobile
   │
   ▼
Application ────────────────► Database
   │                            └── query metrics
   ├─────────────────────────► Vendor APIs
   │                            └── latency / status / retries
   ├── logs
   ├── metrics
   ├── traces
   └── events
        │
        ▼
   TELEMETRY DATA
        │
        ▼
   structured observations
        │
        ▼
 future ML pipeline
```

Instrumentation selects a partial view of reality. Missing telemetry means “not recorded,” not zero and not healthy. More collection is not automatically better: data has storage, quality, privacy, security, and operational costs.

## Observation versus feature

A one-minute observation could be:

```python
{
    "timestamp": "2026-08-12T10:15:00Z",
    "api_latency_ms": 184,
    "http_status": 200,
    "db_connections": 31,
    "queue_depth": 12,
    "vendor_latency_ms": 220,
}
```

A future pipeline might deliberately derive:

```text
error_rate_last_5_minutes
average_vendor_latency_last_10_minutes
queue_growth_rate
requests_since_deployment
latency_change_from_baseline
```

```text
RAW SYSTEM DATA
      │
      ▼
OBSERVATIONS
      │
      ▼
DERIVED VALUES
      │
      ▼
FEATURES
      │
      ▼
MODEL
```

Raw data is what systems emit. An observation gives selected values a schema, time, units, and scope. A derived value transforms one or more observations. A **feature** is a deliberate representation supplied to a model. A field does not become a feature merely because it exists, and not every useful correlation key should be one. Feature engineering and model training belong to later chapters.

## Data types and operational meaning

- **Numerical:** `api_latency_ms = 184`, `queue_depth = 12`, `db_connections = 31`. Units, valid ranges, and whether values are counts or continuous measurements matter.
- **Categorical:** `endpoint = "/api/accounts"`, `vendor = "ClearVerify"`, `deployment_version = "2026.08.12.3"`. Categories are labels, even when a version resembles a number.
- **Boolean:** `timeout = true`, `retry = false`. Define whether false means “did not happen” or merely “was not recorded.”
- **Temporal:** `timestamp = 2026-08-12T10:15:00Z`. Use an unambiguous timezone and record window boundaries where relevant.
- **Identifier:** `request_id = req-8127`, `trace_id = trace-4198`. Identifiers join logs and spans extremely well, but their arbitrary uniqueness usually makes them poor direct model features. They may also be pseudonymous data requiring protection.

## Time, correlation, and causation

Timestamps let the developer align otherwise separate evidence:

```text
10:14 vendor latency rises
10:15 retry count rises
10:16 application queue grows
10:17 API latency rises
10:18 error rate rises
```

> Did the vendor slowdown cause the application incident?

The ordering makes that hypothesis plausible; it does not prove it. A deployment, network problem, traffic burst, clock error, or other common cause could affect both systems. Aggregation can also conceal request-level differences.

```text
OBSERVATION   Vendor latency increased.
CORRELATION   Vendor latency and Harbor API latency increased together.
HYPOTHESIS    Vendor degradation may be contributing to Harbor latency.
DIAGNOSIS     Requires additional evidence.
ACTION        Requires engineering judgment.
```

Chapter 0's boundary remains: an observation records evidence; a future prediction estimates an outcome; diagnosis explains the incident using corroborated evidence; a decision selects a response; and an action changes the system. A timeline moves an investigation forward but does not collapse those stages.

## Executable laboratory

The fixture [`data/harbor_incident_telemetry.csv`](../../data/harbor_incident_telemetry.csv) contains eight synthetic, one-minute observations from normal behavior into an emerging incident. It contains no real member information. `TelemetryObservation` is an immutable dataclass with a timezone-aware timestamp, integer counts and latencies, and a floating-point error rate. It rejects negative engineering measurements, missing timezone offsets, and rates outside 0–1.

The standard-library CSV loader:

1. checks the exact schema;
2. parses ISO 8601 timestamps and the `Z` UTC suffix;
3. converts each numerical field to its declared type;
4. validates every dataclass instance; and
5. rejects malformed or nonchronological rows with the CSV line number.

From the repository root, run:

```bash
python examples/chapter_01_telemetry.py
```

The timeline begins like this and continues through 10:17:

```text
Harbor Federal Credit Union — Telemetry Timeline

10:10  API 180  ms  Vendor 220  ms  Queue 12   Errors 0.40%
10:11  API 191  ms  Vendor 235  ms  Queue 14   Errors 0.50%
...
10:17  API 2380 ms  Vendor 1900 ms  Queue 147  Errors 8.70%
```

The example then reports minimum, maximum, and average API latency; minimum and maximum vendor latency; queue growth; and error-rate change. These are **descriptive statistics and deterministic transformations**, not predictions. No training algorithm or model is present. The sequence supports questions—vendor latency rises early, then queues and errors worsen—but does not answer root cause.

Run the executable checks with:

```bash
pytest tests/test_telemetry.py
pytest
```

Tests cover loading, timestamp and numerical conversion, count and ordering, exact summaries, malformed values, validation, and out-of-order timestamps.

## Privacy, security, and minimum necessary data

The ability to collect a field is not permission or justification to use it. Operational ML should prefer the minimum information necessary for the engineering problem. Account numbers, authentication credentials, access tokens, Social Security numbers, raw identity-verification documents, sensitive member communications, and unnecessary personally identifiable information do not belong in this incident telemetry dataset.

```text
BAD TELEMETRY DESIGN                 BETTER ENGINEERING TELEMETRY

member_ssn                           request_id
account_number                       endpoint_category
full_name                            status_code
access_token                         api_latency_ms
api_latency                          vendor_latency_ms
                                     retry_count
                                     timestamp
```

The better design records system behavior without copying identity or financial contents. It is not risk-free: request IDs and trace IDs can be linkable, endpoint categories may reveal behavior, and operational data can expose architecture. Apply access control, encryption, retention limits, auditability, redaction, and approved data governance to telemetry as appropriate. Never log secrets. Data minimization also improves relevance: collecting unrelated sensitive fields increases harm and noise, not causal understanding.

## Exercises

1. **Classify fields.** Label each as numerical, categorical, boolean, temporal, or identifier: `queue_depth`, `deployment_version`, `timeout`, `timestamp`, `trace_id`, `status_code`, and `vendor_latency_ms`. Explain any field whose representation could be ambiguous.
2. **Observation or derived feature?** Classify `vendor_latency_ms` at 10:15, `average_vendor_latency_last_10_minutes`, a raw timeout event, and `latency_change_from_baseline`. State what additional definition each derived value needs before it could be a feature.
3. **Minimize sensitive data.** Review this proposed row: `timestamp, full_name, account_number, access_token, endpoint, latency_ms, status_code`. Identify removals, safer replacements, and controls still needed for what remains.
4. **Correlation.** Vendor latency and queue depth rise together. Give two competing explanations and name evidence that could distinguish them. Explain why simultaneity alone is insufficient.
5. **Choose a source.** For each scenario, choose what to inspect first and why: (a) one request is slow across several services; (b) database capacity degrades over 20 minutes; (c) completed transfers fall while HTTP responses remain successful; (d) a fictional vendor begins timing out. Consider traces, metrics, application events, database observations, logs, and API boundary observations.
6. **Coding—extend the observation.** Add `requests_per_minute` to the CSV fixture and `TelemetryObservation`; parse and validate it; display it in the timeline; and update tests for conversion, output-related behavior, and malformed input. Do not turn it into a model feature merely by adding it. Run the chapter example and full test suite.

## Key takeaways

- Operational evidence comes from sources with distinct meaning and scope.
- Structured observations make types, units, time, and validation explicit.
- Timestamps enable alignment; correlation supports hypotheses but does not prove causation.
- Identifiers are valuable join keys yet usually poor direct features.
- A feature is an intentional model input, not any available field.
- Deterministic summaries describe this fixture; Chapter 1 trains no model.
- Minimum-necessary telemetry is safer and often more useful than indiscriminate collection.

## What comes next

The reader now has observations. Chapter 2, **From Engineering Problem to ML Problem**, asks the harder question:

> What exactly are we asking a machine-learning system to learn?

```text
Engineering question
        ↓
measurable target
        ↓
available observations
        ↓
features
        ↓
training examples
        ↓
evaluation criteria
```

That framing comes next. This chapter does not implement it.
