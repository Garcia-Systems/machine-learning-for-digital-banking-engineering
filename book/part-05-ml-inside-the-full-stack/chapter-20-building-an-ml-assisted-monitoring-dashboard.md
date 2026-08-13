# Chapter 20 — Building an ML-Assisted Monitoring Dashboard

Harbor Federal Credit Union is fictional. Its telemetry, incidents, vendors, members, and model results are synthetic educational material.

> **Central question:** How should Harbor present ML predictions alongside ordinary system telemetry so developers can use them effectively during monitoring and troubleshooting?

Chapter 19 gave Harbor's application a typed way to obtain an advisory prediction. That does not yet give an engineer situational awareness. During an incident, a developer might find this in ordinary telemetry:

```text
api_latency_ms = 1480
error_rate = 0.047
queue_depth = 96
db_connections = 78
vendor_latency_ms = 1320
```

At the same instant, three models may report an anomaly, an incident pattern, an integration-failure probability, and a model version. If those facts remain scattered across logs, APIs, terminals, and scripts, the developer still has to assemble the picture mentally. Harbor needs one internal view that asks:

> What is happening right now, what signals look unusual, what does the model suggest, and what evidence should I investigate next?

```text
APPLICATION + DATABASE + VENDOR TELEMETRY
                  │
                  ▼
             DASHBOARD
                  ▲
                  │
         ML PREDICTIONS
```

The goal is **situational awareness**, not an automated root-cause declaration.

## Learning objectives

By the end of this chapter, you will be able to:

1. distinguish telemetry from ML predictions;
2. keep those signals separate but related in a monitoring view;
3. display model probabilities without presenting certainty;
4. display anomaly scores without calling them probabilities;
5. expose model version and prediction time operationally;
6. show deterministic thresholds beside model output;
7. represent ordered incident-class probabilities and ambiguity;
8. show recent prediction history;
9. explain and reduce alert fatigue;
10. avoid automated root-cause claims;
11. degrade gracefully when ML is unavailable or stale;
12. build and test a small server-rendered dashboard; and
13. use dashboards to support—not replace—investigation.

## Evidence is not all the same

The strongest dashboard design principle is an explicit evidence hierarchy:

```text
DIRECT OBSERVATION
vendor latency = 1320 ms
        ↓
DERIVED METRIC
vendor latency increased 420% from baseline
        ↓
ML SIGNAL
model assigns 0.61 probability to vendor_degradation
        ↓
ENGINEERING HYPOTHESIS
vendor performance may be contributing
        ↓
INVESTIGATION
inspect traces, status codes, retries, vendor behavior
```

A direct observation is evidence from the running system. A derived metric is deterministic arithmetic over observations. An ML signal is an interpretation learned from historical examples. A hypothesis is a developer's provisional explanation. Only investigation can validate it. Moving upward through this hierarchy adds interpretation; it does not magically add proof.

Harbor's dashboard therefore uses three visibly distinct sections:

```text
OBSERVED
Vendor latency: 1320 ms
Queue depth: 96
Error rate: 4.7%

MODEL SUGGESTS
Most similar known pattern: vendor_degradation
Model-assigned probability: 0.61

INVESTIGATE
- vendor traces
- retry activity
- dependent endpoints
- recent deployment
```

“Model pattern: `vendor_degradation`” is precise. “ROOT CAUSE: VENDOR” is not. Likewise, “model-assigned probability for `vendor_degradation`: 0.62” describes a classifier output; “62% confident the vendor is down” silently changes what the model predicts.

## Teaching architecture

```text
                         HARBOR SYSTEM

Web / Mobile
     │
     ▼
Application
     │
     ├────────────► Database
     │
     ├────────────► Vendor APIs
     │
     └────────────► ML Service
                       │
                       ▼
                prediction signals
                       │
     ┌─────────────────┘
     │
     ▼
Telemetry / Observation Layer
     │
     ▼
Engineering Dashboard
     │
     ▼
Developer Investigation
```

A production design might score asynchronously, read from a time-series store, use separate services, or integrate with an established observability platform. This teaching system deliberately stays small. It reuses FastAPI from Chapter 18 and adds Jinja2 server rendering—no React, Vue, CDN, production database, or charting library.

The implementation separates responsibilities:

```text
src/harbor_ml/dashboard/
├── models.py                  typed dashboard state
├── service.py                 model calls and assembly
├── app.py                     GET /dashboard
└── templates/dashboard.html   semantic presentation
```

Model calls never occur in the template. The HTML receives a fully assembled `DashboardSnapshot`.

## A structured snapshot

`TelemetrySnapshot` contains direct observations. `DashboardSnapshot` relates them to separately labeled ML fields:

```python
@dataclass(frozen=True)
class DashboardSnapshot:
    telemetry: TelemetrySnapshot
    generated_at: datetime
    severity: str
    ml_status: MLAvailability
    prediction_timestamp: datetime | None = None
    prediction_age_seconds: float | None = None
    telemetry_anomaly: bool | None = None
    anomaly_score: float | None = None
    predicted_incident_class: str | None = None
    incident_probabilities: dict[str, float] = field(default_factory=dict)
    integration_failure_probability: float | None = None
    integration_failure_threshold: float | None = None
    integration_failure_prediction: bool | None = None
    model_name: str | None = None
    model_version: str | None = None
```

The optional ML fields are intentional. `None` means no value exists. It must not be transformed into `0.00`, because an unavailable predictor has supplied no evidence of low risk.

## Reusing the models already built

The laboratory does not train a dashboard-specific model. `build_teaching_service()` fits the existing deterministic educational paths:

- Chapter 4's Isolation Forest learns from `harbor_normal_telemetry.csv`;
- Chapter 5's logistic classifier learns the five known incident patterns in `harbor_incident_telemetry.csv`; and
- Chapter 16's controlled integration-failure training workflow supplies its pipeline, threshold, model name, and version.

In a deployed system, use Chapter 16's trusted local artifact and validated metadata rather than fitting at dashboard startup. Fixture fitting here keeps the lab executable without committing a binary or requiring external infrastructure.

```text
telemetry
    + anomaly result
    + incident classification
    + integration prediction
    + metadata
        │
        ▼
DashboardSnapshot
```

## System health: deterministic observations first

The first dashboard section shows API latency, error rate, DB connections, queue depth, vendor latency, requests per minute, and telemetry timestamp. These are observations. They retain value even if every model is offline.

Severity is also deterministic policy:

```python
if error_rate >= 0.05 or api_latency_ms >= 2_000:
    severity = "critical"
elif error_rate >= 0.02 or api_latency_ms >= 800:
    severity = "warning"
else:
    severity = "normal"
```

These are fictional teaching thresholds, not universal service-level objectives. Harbor would configure real thresholds from its own objectives and operational experience.

```text
SEVERITY                         ML SIGNAL
deterministic operational       additional evidence
policy
```

An ML probability alone never sets dashboard severity. This prevents a model deployment from silently rewriting paging or operational policy.

## Anomaly signal: score is not probability

Chapter 4 negates Isolation Forest's decision function so larger values mean “more unusual relative to the learned baseline.” The dashboard displays:

```text
Anomaly status: YES
Anomaly score: 0.0842 (unusualness score; not a probability)
```

It does **not** say “8.42% probability the system is broken.” The score is uncalibrated, may be negative, and has meaning only with the fitted model and baseline. Its explanatory text is:

> Indicates how unusual the current telemetry appears relative to the learned baseline.

## Incident classification and ambiguity

The classifier produces a category and a probability distribution over its known historical labels. The dashboard orders the actual computed values from largest to smallest and captions them **model-assigned incident-class probabilities (not confirmed causes)**.

When the top two probabilities are separated by less than the configured `AMBIGUITY_GAP` (0.10 in this lab), the page adds:

```text
Pattern classification is ambiguous.
```

The rule is deterministic and configurable. For example, `0.27` versus `0.24` is diffuse; forcing a strong-looking label would hide useful uncertainty. The classifier still cannot recognize every possible incident: it chooses among the categories it learned.

## Integration failure signal and metadata

The integration panel shows four related facts:

```text
Failure probability: 0.74 — model-assigned risk
Threshold:           0.50
Predicted class:     elevated failure risk
Model:               harbor-integration-failure
Model version:       3a91c427...
```

Probability and threshold are separate. The Boolean class is the deterministic comparison `probability >= threshold`; it does not mean failure is certain. Avoid theatrical labels such as `DANGER` or `VENDOR FAILURE CERTAIN`.

Model identity matters because troubleshooting must answer **which model produced this prediction?** A history that changes from model v1 at 14:00 to v2 at 15:00 lets an engineer ask whether changed predictions came from changed system behavior, a changed model, or both.

## Availability, prediction time, and staleness

ML evidence has three presentation states:

```text
AVAILABLE    value exists within the configured freshness window
UNAVAILABLE  prediction was not obtained
STALE        value exists but is older than the freshness window
```

The service calculates age relative to an injected `now`. Its teaching default is five minutes, but that value is configuration—not a universal truth. At exactly the boundary the prediction remains available; after the boundary it is stale.

```text
prediction older than configured freshness window → STALE
```

Clamping future clock skew to age zero avoids a negative age display, though production systems should also monitor clock synchronization. A stale result remains visible as historical context, but the page warns that it must not be treated as current evidence.

Most importantly:

```text
TELEMETRY AVAILABLE
ML UNAVAILABLE
       │
       ▼
DASHBOARD STILL WORKS
```

The unavailable view says “ML prediction unavailable.” It never says `failure probability = 0.00`. This repeats Chapter 19's distinction between low predicted risk and absence of a prediction.

## Safe investigation guidance

The model selects a known incident class. Normal application logic maps that label to suggested directions:

```python
INVESTIGATION_GUIDANCE = {
    "vendor_degradation": (
        "Inspect vendor latency and timeout telemetry.",
        "Review retry behavior.",
        "Inspect distributed traces involving the vendor.",
    ),
    "database_pressure": (
        "Inspect connection-pool pressure.",
        "Review slow-query telemetry.",
        "Inspect lock and queue behavior.",
    ),
}
```

This mapping is normal application logic. It is reviewed, predictable, testable, and cautious. No LLM generates a diagnosis. A label selects investigation directions, not a conclusion.

```text
Dashboard signal
      │
      ▼
Check deterministic telemetry
      │
      ▼
Inspect traces
      │
      ▼
Inspect logs
      │
      ▼
Inspect database/vendor details
      │
      ▼
Form hypothesis
      │
      ▼
Validate
```

ML is one input to this workflow.

## Recent history

The service retains at most eight snapshots in memory, and the page renders a semantic table with headers for time, API latency, errors, queue, anomaly, incident pattern, failure probability, availability, and model version. This is enough to teach sequence and correlation without pretending to be a production time-series database.

In-memory history disappears on restart, does not coordinate across workers, and is not suitable for audit or production retention. Production should use its approved observation store. Trend arrows are omitted; if added, compute them deterministically from previous/current values and label them only “rising,” “falling,” or “stable”—never statistically significant.

## Route, semantic HTML, and accessibility

The application exposes `GET /dashboard`. Chapter 18's JSON prediction API remains under `/api/v1`; HTML presentation is not mixed into that contract. The app takes an injected service and current snapshot, which makes the route testable with `TestClient` and no socket.

The self-contained template provides:

- a meaningful page title and generated timestamp;
- `<main>` and named `<section>` elements;
- definition lists for labeled values;
- captions, column headers, and scoped headers on tables;
- text words for every status, rather than color alone; and
- probability descriptions that state what each number means.

Minimal local CSS makes it readable as engineering tooling. There are no external assets.

## Executable laboratory

Install the development dependencies and run:

```bash
python examples/chapter_20_monitoring_dashboard.py
```

The program fits only the existing deterministic fixtures, assembles five scenarios, requests `/dashboard` in process, checks key semantic content, and prints the actual model results. No long-running server is required.

The scenarios are:

1. **Healthy:** low latency, errors, and queue; deterministic severity is normal.
2. **Vendor degradation:** vendor/API latency and queue are elevated.
3. **Database pressure:** DB connections and queue are high while vendor latency is ordinary.
4. **ML unavailable:** telemetry renders while all ML fields remain `None`.
5. **Stale prediction:** a 20-minute-old result is retained but explicitly marked stale.

The values fed to the models are exactly the telemetry shown. The integration model also receives its established Chapter 7 request features—vendor, endpoint, retry count, request size, and hour—which are not invented from the six system-health fields.

To run a local teaching server, construct the same injected app and serve it with Uvicorn. The laboratory intentionally avoids asking readers to manage that process. A production deployment would additionally need authentication, authorization, TLS, approved storage, and operating controls.

## Testing rendering as behavior

`tests/test_dashboard.py` checks snapshot assembly, deterministic severity boundaries, actual anomaly and classification results, probability ordering, thresholded failure output, model metadata, five scenarios, guidance, ambiguity, availability boundaries, and HTML semantics. Assertions target meaningful text rather than whitespace or CSS details.

Two negative tests are especially important:

- a snapshot whose ML fields are `None` still returns HTTP 200 and clearly says unavailable; and
- rendered output contains no root-cause certainty claim.

This treats language as part of the operational contract. Misleading words can be a dashboard defect even when the arithmetic is correct.

## Dashboard versus alert: avoiding fatigue

```text
DASHBOARD
developer looks when investigating

ALERT
system interrupts developer
```

A dashboard signal and an interrupting alert are different products. A moderately useful ML signal can add context to an investigation while still being too noisy for paging. If every unusual score creates a red interruption:

```text
too many alerts
      │
      ▼
engineers ignore them
```

Before an ML signal contributes to alerts, evaluate:

- **thresholds:** what operating point produces acceptable false interruptions?
- **persistence windows:** must the condition persist rather than appear once?
- **deduplication:** can related events become one incident?
- **severity:** which deterministic service impact justifies urgency?
- **context:** does the alert include observations, timestamps, model identity, and runbook direction?
- **routing:** which team can act, and should this be a ticket, chat notification, or page?

Do not route alerts merely because a model emits a number. Measure alert precision, actionability, and engineer response. A full alert engine is intentionally outside this chapter.

## Internal-only security boundary

This is an internal engineering dashboard, never a member-facing screen. Production must:

- restrict access with proper authentication and authorization;
- avoid public exposure of sensitive topology or telemetry;
- display no credentials, tokens, member data, or secrets;
- avoid rendering raw exceptions and downstream response bodies;
- sanitize or escape externally sourced text (Jinja autoescaping remains enabled here);
- treat model/version metadata as useful but internal; and
- define approved retention for telemetry and prediction history.

Authentication is not implemented because the laboratory is an in-process teaching boundary, not a deployment template.

## Exercises

### Exercise 1 — Observation or prediction?

Classify each value: `vendor_latency_ms = 1320`, `error_rate = 4.7%`, and `queue_depth = 96` are direct observations. An anomaly score and incident-class probability are model outputs. What label and explanatory sentence should accompany each?

### Exercise 2 — Root cause

Why is `ROOT CAUSE: DATABASE` dangerous? Replace it with language such as `Model pattern: database_pressure` and `Suggested investigation: database dependency`, then list evidence required to validate the hypothesis.

### Exercise 3 — Missing ML

What should appear when the ML service is unavailable? Explain why `failure probability = 0` falsely converts missing evidence into a prediction of no risk.

### Exercise 4 — Model version

Why must history record the producing model? Consider a distribution shift that begins at the same time as model v2 deployment.

### Exercise 5 — Alert fatigue

Give an example of a useful dashboard signal that is not reliable or actionable enough to page an engineer. Which persistence, routing, and severity rules would you evaluate?

### Exercise 6 — Stale prediction

Why could a prediction from 20 minutes ago mislead during a fast-changing incident? What direct evidence should take precedence?

### Coding exercise — prediction age

Recreate `prediction_age_seconds`:

1. calculate it from an injected current time and prediction timestamp;
2. accept a configurable freshness limit;
3. classify the boundary as current and values beyond it as stale;
4. display age and status;
5. test current, exact-boundary, beyond-boundary, missing, and future-clock cases; and
6. document why stale ML output cannot be presented as current evidence.

## Key takeaways

1. ML predictions belong beside deterministic telemetry, not in place of it.
2. Direct observations, model suggestions, hypotheses, and engineering conclusions are different evidence layers.
3. Anomaly scores are not probabilities.
4. Incident-class probabilities are not confirmed root causes.
5. Model version and prediction time are operationally important.
6. Missing ML must be unavailable, never fake zero risk.
7. Stale predictions must be clearly marked.
8. A dashboard and an interrupting alert have different reliability requirements.
9. ML-assisted monitoring helps engineers form better hypotheses.
10. Final diagnosis comes from investigation and evidence.

## Part V conclusion

```text
PART V — PUTTING MACHINE LEARNING INSIDE THE FULL STACK

Chapter 16
Train a controlled model artifact
        │
        ▼
Chapter 17
Evaluate whether it is useful
        │
        ▼
Chapter 18
Serve it through a Python API
        │
        ▼
Chapter 19
Integrate it with PHP
        │
        ▼
Chapter 20
Present predictions to engineers
```

Harbor now has real ML integration: controlled training, evaluation, serving, a cross-language application adapter, and an internal monitoring surface. That capability creates harder questions:

> What data is safe to use?
>
> Can developers understand why a prediction happened?
>
> Could a model behave unfairly?
>
> When should humans remain in control?
>
> How do we monitor the model itself?

Those questions lead to **Part VI — Responsible Machine Learning in Financial Systems**.

## Next: Chapter 21 — Security and Sensitive Financial Data

```text
DATA COLLECTION
      │
      ▼
MINIMIZATION
      │
      ▼
VALIDATION
      │
      ▼
ACCESS CONTROL
      │
      ▼
TRAINING / INFERENCE
      │
      ▼
LOGGING / STORAGE
```

Chapter 21 will protect sensitive data throughout the ML lifecycle while keeping Harbor's examples fictional and privacy-minimized. It remains unimplemented.

[Previous: Chapter 19](chapter-19-integrating-machine-learning-with-php.md) · [Back to Part V](README.md) · [Complete contents](../../CONTENTS.md)
