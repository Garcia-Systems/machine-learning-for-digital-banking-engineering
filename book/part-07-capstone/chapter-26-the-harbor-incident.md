# Chapter 26 — The Harbor Incident

> **Part VII — Capstone: The Intelligent Digital Credit Union**

At fictional **Harbor Federal Credit Union**, an identity-verification workflow starts an ordinary Wednesday morning looking healthy. Thirty minutes later, some requests fail. Several models contribute useful signals—but none observes root cause directly.

This chapter integrates tools already developed. It introduces **no new ML algorithm**. Its central question is:

> During a realistic digital-banking incident, how can Harbor combine deterministic telemetry, ML predictions, explanations, model metadata, and engineering investigation without confusing prediction with diagnosis?

The discipline throughout the incident is:

```text
OBSERVATION
     ↓
MODEL SIGNAL
     ↓
HYPOTHESIS
     ↓
INVESTIGATION
     ↓
EVIDENCE
     ↓
DIAGNOSIS
     ↓
ACTION
```

Skipping a step is how “the classifier resembles vendor degradation” becomes the unsupported claim “the vendor is the root cause.”

## Learning objectives

By the end, you will be able to:

1. follow a time-ordered production incident;
2. distinguish direct telemetry from derived metrics;
3. distinguish anomaly detection from incident classification;
4. interpret request-failure probabilities;
5. explain why classifier probabilities change as incidents evolve;
6. track the model and version behind every signal;
7. preserve ambiguous outputs;
8. form hypotheses without overstating certainty;
9. choose a next investigation based on evidence;
10. recognize database pressure as a possible secondary effect;
11. distinguish symptoms, contributing factors, and confirmed diagnosis; and
12. explain how ML shortens investigation without replacing engineering judgment.

## The executable evidence

All incident data is deterministic, fictional, and privacy-minimized:

- `data/harbor_capstone_incident.csv` contains one aligned observation every two minutes from 10:00 through 10:38 UTC;
- `data/harbor_capstone_traces.csv` contains component durations and statuses for two synthetic requests;
- `src/harbor_ml/capstone_incident.py` validates fixtures, constructs representative requests, invokes the existing models, assigns editorial phases, preserves unavailable states, and formats evidence layers; and
- `examples/chapter_26_harbor_incident.py` trains the existing Chapter 4, 5, and integration-failure models, then evaluates every timestamp.

Run it from the repository root:

```bash
python examples/chapter_26_harbor_incident.py
```

Repeated runs use committed training data, fixed random states, a fixed metadata timestamp, and the same capstone fixture. Probabilities in the output are computed by fitted models, not stored in the incident CSV. The fixture stores facts available to inference—not model answers.

## Four distinct kinds of information

The laboratory does not place every number under a generic “insight” heading.

| Layer | Example | What it can establish |
| --- | --- | --- |
| Direct telemetry | `vendor_latency_ms = 1690` | the instrument reported elevated latency |
| Deterministic derivation | severity is `warning` | configured thresholds were crossed |
| Model signal | `vendor_degradation` has the highest probability | the current feature vector resembles that learned historical class |
| Investigation evidence | trace spends 2218 ms in `ClearVerify_call` and records `timeout` | where selected requests spent time and what that span reported |

An anomaly score is a derived, uncalibrated model score. An incident-class probability and request-failure probability are model estimates. Neither is direct telemetry. A deterministic severity is not “more intelligent”; it answers a different, explicit rules question.

## The fixture and narrative phases

The CSV includes API latency, error rate, database connections, queue depth, external-vendor latency, request volume, retries, vendor timeout rate, deployment version, and classifier availability. It contains no member name, account number, address, credential, or transaction.

The module assigns these **editorial**, deterministic phases:

```text
healthy → early_signal → degradation → compound_pressure
        → confirmed_incident → recovery
```

These labels organize the story. They are not classifier targets, learned outputs, or proof of causality.

The deployment remains `web-2026.08.11.3` throughout. It was deployed the previous day. That fact weakens a recent-deployment regression hypothesis; it does not prove that application code cannot be defective.

## 10:00–10:04 — healthy baseline

At 10:00 the system reports approximately:

```text
API latency:        184 ms
error rate:         0.4%
queue depth:        11
DB connections:     31
vendor latency:     218 ms
requests/minute:    704
retries:            0
```

### Observation

All direct metrics are near the normal fictional baseline. The deterministic status is `normal`.

### Model signal

The anomaly detector evaluates the current six-feature vector. The incident classifier generally favors `normal`. The integration model calculates a probability for a representative ClearVerify `identity_verify` request using the current vendor latency, timeout rate, queue, retry count, a fixed privacy-safe request size, and hour of day.

### Engineering interpretation

There is no incident evidence yet. A nonzero failure probability is expected: probability is not a declaration that this request will fail.

## 10:06–10:10 — a localized early signal

At 10:06 the directly observed ClearVerify latency reaches 335 ms while API latency, database connections, volume, and errors remain close to baseline. At 10:08 vendor latency reaches 510 ms and one retry appears. At 10:10 vendor latency is 690 ms, retries are two, and the representative request-failure probability has risen from its earlier computed value.

This ordering matters.

```text
Observed:
ClearVerify latency is elevated and retry activity has begun.

Model suggests:
The representative request has a higher fitted failure probability.

Hypothesis:
The external identity-verification dependency may be contributing.

Next investigation:
Compare vendor spans and retry behavior with healthy requests.
```

Harbor does **not** announce a root cause. At this point, a network path, client configuration, instrumentation defect, or localized request mix could also contribute.

## 10:12–10:18 — multiple signals accumulate

Queue depth grows after vendor latency and retries have already risen. At 10:14, the anomaly detector can flag that the combined behavior no longer resembles its normal-operation training baseline. At 10:16 API latency crosses Harbor's fictional 800 ms warning threshold. By 10:18:

```text
vendor latency:     1690 ms
queue depth:        108
API latency:        1210 ms
DB connections:     55
error rate:         2.6%
requests/minute:    710
```

Request volume remains close to its 10:00 value. This weakens a traffic-spike hypothesis even though queue depth is growing.

### Anomaly detection is not classification

The Isolation Forest asks:

> Does this observation look unusual relative to the learned healthy baseline?

It does not answer *which incident is occurring*. The logistic classifier asks:

> Which labeled historical pattern does this observation most resemble?

It cannot prove that the matching historical explanation is the present cause. The integration model asks a third question:

> For this representative request context, what probability does the fitted model assign to request failure?

The three answers can change at different times because they have different feature contracts, training data, and objectives.

### Versioned signals

Each evaluated observation records a mapping such as:

```text
harbor-telemetry-anomaly:    chapter-04-iforest-v1
harbor-incident-classifier:  chapter-05-logreg-v1
harbor-integration-failure:  harbor-integration-failure-<dataset hash>
```

If the classifier is unavailable, its identity is not falsely attached as though it produced a result. This answers: **which model produced this signal at this time?** A model name without its version is insufficient incident evidence.

## 10:20–10:28 — compound pressure

Database connections rise from 55 at 10:18 to 112 at 10:28. Queue depth reaches 218, API latency reaches 2380 ms, and errors reach 10.8%. Some representative requests now fail. Vendor latency remains very high, while request volume remains near baseline.

The following chain is now a plausible hypothesis:

```text
vendor latency rises
        ↓
requests remain in flight longer
        ↓
retries increase
        ↓
queue grows
        ↓
DB connections increase
        ↓
API latency rises further
        ↓
errors increase
```

It is still a hypothesis. Time ordering and plausibility do not by themselves establish every causal arrow.

### Why probabilities evolve

The classifier evaluates the **current** observation, not an immutable incident identity. Early observations may favor `normal`; later observations can favor `vendor_degradation`; recovery observations with persistent database connections may resemble `database_pressure` as well.

> Secondary system effects can make the current observation resemble more than one historical class.

At 10:34, vendor latency and retries are falling but database connections and queue depth remain elevated. With Chapter 20's top-two-gap rule, the actual computed leading probabilities are close enough for:

```text
classification = ambiguous
```

The laboratory retains every actual probability. It does not replace them with a confident label. Ambiguity is operational information: investigate both dependency recovery and lagging database pressure.

### 10:24 — classifier unavailable

The fixture deliberately marks the incident classifier unavailable at 10:24. The output says:

```text
Incident classifier: UNAVAILABLE
```

It does **not** print a fake `normal` score and does not reuse the 10:22 prediction. Direct telemetry, deterministic critical severity, anomaly result, representative request-failure probability, deployment version, and investigation guidance remain available. Engineering work continues. Availability is itself part of trustworthy presentation.

## Hypotheses before confirmation

Immediately before trace evidence is gathered, the team maintains this table:

| Hypothesis | Supporting evidence | Contradicting or weakening evidence | Status |
| --- | --- | --- | --- |
| Vendor degradation | vendor latency and timeout rate rise first; retries rise; classifier resemblance | no confirming trace examined yet | strong hypothesis |
| Primary database problem | connections and queue are high | connection rise is later; selected query durations not yet abnormal | possible secondary pressure |
| Traffic spike | queue grows | requests/minute remains about 699–721 | weak |
| Application regression | API latency and errors are high | application version unchanged since previous day | weak, not impossible |

The table prevents a high model probability from silently becoming a diagnosis.

## 10:30 — deterministic evidence arrives

At 10:30 the engineer loads trace spans for `capstone-request-001`. The executable lab computes the total and dominant component from the trace CSV:

```text
identity_verify request total: 2410 ms

application_code:                38 ms  ok
database_work:                   71 ms  ok
ClearVerify_call:              2218 ms  timeout
retry_delay:                     83 ms  completed
```

The ClearVerify span is longer than all other recorded spans combined. A second synthetic request shows the same ordering and another timeout. Together with rising vendor latency, timeout rate, and retries across the timeline, this is materially stronger evidence than class resemblance alone.

The diagnosis is therefore recorded no earlier than the configured confirming-evidence time:

> **ClearVerify degradation is the primary confirmed contributor to the Harbor identity-verification incident. Increased retries and longer-lived requests subsequently increased queue depth and database connection pressure, amplifying application latency.**

The fixture supports “primary confirmed contributor,” not a claim that every failure has exactly one cause or that the database pressure is imaginary.

```text
PRIMARY CONTRIBUTING CONDITION
ClearVerify degradation, supported by trace timeouts and dominant duration

SECONDARY SYSTEM PRESSURE
longer-lived and retried requests; growing queue and DB connections
```

**ML helps prioritize hypotheses. Evidence establishes diagnosis.**

## Safe action and recovery

Harbor's response is operational rather than punitive:

1. stop aggressive retry behavior;
2. activate deterministic degraded-mode handling for verification;
3. preserve and improve vendor, retry, queue, and trace observability;
4. notify internal engineering and vendor-management channels; and
5. monitor recovery across all layers.

From 10:32 through 10:38, vendor latency and retries fall first. Queue depth and database connections drain more slowly. API latency and errors follow. Different recovery rates support the idea of downstream pressure, but the team still validates database health independently.

## Structured incident log

`IncidentNote` restricts categories to:

```text
observation | model_signal | hypothesis | investigation
evidence | diagnosis | action
```

Before 10:30, evaluated observations contain observations, hypotheses, and investigations—but no confirmed diagnosis. At and after 10:30, trace evidence, diagnosis, and action may be recorded. This mechanical boundary is tested. It makes the conceptual layers reviewable rather than relying only on careful prose.

## Incident retrospective

**What detected the incident first?** Direct vendor latency and retry telemetry showed localized change before broad API failure. The request-level model also produced an early localized risk signal.

**What did the anomaly detector contribute?** It surfaced unusual combined telemetry relative to normal operation before every deterministic threshold became critical. It did not identify a cause.

**What did the classifier contribute?** It prioritized familiar operational hypotheses and exposed how resemblance evolved. It was unavailable at one timestamp and ambiguous during lagging secondary pressure.

**What did the request-failure model contribute?** It translated request-time context into a changing probability for a representative identity-verification request. That was useful for localized warning, not certainty about an individual outcome.

**Which signal was ambiguous or potentially misleading?** The classifier later saw database-pressure resemblance because database connections stayed high while other metrics recovered. Treating its top class as a root-cause label would be misleading.

**What confirmed the diagnosis?** Computed trace component durations and timeout statuses, interpreted alongside retry patterns and time-ordered operational telemetry.

## Exercises

### Exercise 1 — Observation or interpretation?

Classify each statement:

```text
vendor_latency_ms = 1600
vendor_degradation probability = 0.68
the vendor may be contributing
the vendor is confirmed as the primary contributor
```

The first is observation, the second model signal, the third hypothesis, and the fourth diagnosis. The diagnosis requires supporting evidence unavailable from the probability alone.

### Exercise 2 — Ambiguity

Suppose `vendor_degradation = 0.43` and `database_pressure = 0.39`. Why should Harbor avoid a confident one-line diagnosis? Discuss model uncertainty, class resemblance, secondary effects, and the evidence needed to discriminate the hypotheses.

### Exercise 3 — Secondary effects

Explain how external vendor slowness could leave requests open, stimulate retries, grow a queue, and retain database connections without the database being the original problem. Identify which arrows require trace or system evidence.

### Exercise 4 — Evidence

Which is stronger evidence of a vendor bottleneck?

- A. classifier says `vendor_degradation`
- B. trace shows 2.2 seconds spent in the vendor call

B directly measures where selected requests spent time. A helps prioritize that investigation but remains learned resemblance. Neither alone proves that every affected request behaved identically.

### Exercise 5 — No recent deployment

Does absence of a recent deployment prove application regression is impossible? **No.** It weakens one hypothesis but does not prove impossibility; latent defects and external changes can expose unchanged code.

### Exercise 6 — ML unavailable

What should engineers do if the classifier is unavailable? Preserve the unavailable state; continue using direct telemetry, deterministic severity, traces, logs, deployment facts, and any independently available signals; investigate the model service separately.

### Coding exercise — a different anomaly

Add a `CapstoneTelemetry` observation whose `requests_per_minute` spikes sharply while vendor latency remains normal. Then:

1. evaluate the existing anomaly detector;
2. evaluate the existing incident classifier;
3. compare it with the vendor-degradation incident;
4. print the top class probabilities; and
5. explain why the same anomaly mechanism can represent a different operational pattern.

Do not add a new model or outcome-derived feature merely to make the result attractive.

## Key takeaways

1. Production incidents unfold over time.
2. Early evidence is incomplete.
3. Direct telemetry and ML signals are different kinds of evidence.
4. Anomaly detection can surface unusual behavior before a known incident class is clear.
5. Incident-class probabilities evolve as secondary effects appear.
6. Request-level predictions can provide early localized warning.
7. Model ambiguity and unavailability must remain visible.
8. Traces and deterministic system evidence are essential for confirming diagnosis.
9. Secondary pressure should not automatically be mistaken for primary cause.
10. ML is most useful when it helps engineers investigate the right hypotheses faster.

## What comes next: Chapter 27 — Building the Telemetry Dataset

Chapter 26 showed the complete incident as an engineering experience. Chapter 27 will step behind the scenes and ask:

> How do we build one coherent capstone dataset from application, vendor, database, trace, and model-monitoring information without creating leakage or inconsistent timestamps?

```text
APPLICATION TELEMETRY
VENDOR TELEMETRY
DATABASE METRICS
TRACE DATA
REQUEST OUTCOMES
        │
        ▼
TIME ALIGNMENT
        │
        ▼
CAPSTONE DATASET
```

It will focus on schemas, timestamps, joins, windows, missing data, source semantics, feature availability, leakage avoidance, and dataset validation. **Chapter 27 remains planned and is not implemented here.**

[Previous: Chapter 25 — Monitoring Models in Production](../part-06-responsible-ml/chapter-25-monitoring-models-in-production.md) · [Back to Part VII](README.md) · [Complete contents](../../CONTENTS.md) · Next: Chapter 27 — Building the Telemetry Dataset *(planned)*
