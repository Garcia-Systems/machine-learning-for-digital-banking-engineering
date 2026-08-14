# Chapter 32 — Building the Engineering Dashboard

During the Chapter 26 incident, **Harbor Federal Credit Union** engineers have API latency, error rate, queue depth, database connections, vendor latency, request volume, and retries. They also have an anomaly score and flag, incident probabilities and ambiguity, a request-failure probability, three availability states, three versions, three prediction times, and a local explanation. Showing every value without its meaning would add confusion rather than understanding.

This chapter asks:

> How can Harbor present deterministic telemetry, several different ML signals, model availability, explanations, and historical context in one engineering interface without turning model output into a fake diagnosis?

The answer is an evidence-oriented debugging interface. It preserves this order:

```text
DIRECT OBSERVATIONS
        │
        ▼
DERIVED SYSTEM STATUS
        │
        ▼
ML SIGNALS
        │
        ▼
MODEL EXPLANATIONS
        │
        ▼
ENGINEERING HYPOTHESES
        │
        ▼
INVESTIGATION
```

It never compresses those layers into `AI ROOT CAUSE`.

## Learning objectives

By the end, you can design a dashboard around evidence layers; distinguish observation from interpretation; combine structured ML results without flattening their semantics; represent unavailable and stale signals; expose independent timestamps and versions; show anomaly unusualness, class probabilities, ambiguity, and request-level risk accurately; present a local explanation without causal language; map signals to deterministic investigation guidance; implement time-bounded incident playback; reduce automation bias through wording; retain utility through ML outage; and test the presentation without external infrastructure.

## Twelve operational questions

The screen must answer, in order: what is observed; whether threshold-derived health is deteriorating; what anomaly detection reports; which known patterns the classifier resembles; whether those classes are close; what request-level failure risk says; whether each model is available; how old each result is; which independent version produced it; which inputs influenced the fitted model; what to inspect next; and what evidence has actually been confirmed.

That order is intentional. A high anomaly score cannot overwrite a low error rate. A class probability cannot become trace evidence. An unavailable classifier cannot become `normal`.

## Architecture: extend Chapter 20

Chapter 32 extends the small FastAPI/Jinja dashboard from Chapter 20 rather than creating a new frontend stack:

```text
src/harbor_ml/dashboard/
  app.py                    # injected HTTP/presentation boundary
  models.py                 # immutable presentation DTOs
  service.py                # model calls and time-bounded assembly
  templates/
    dashboard.html
    incident_playback.html
    signal_metadata.html
  static/dashboard.css
```

There is no React application, client-side chart dependency, or browser model call. `CapstoneDashboardService` owns assembly; templates receive a completed snapshot. The simple probability bars retain numeric text, table headers use `scope`, and every state has a textual label.

The sources remain deliberately separate:

```text
telemetry source       → deterministic observations
ML service             → prediction signals
explainability module  → fitted-model explanation
trace fixture          → confirmed incident evidence
```

This is the architectural defense against fake diagnosis.

## A model that preserves semantics

`DashboardSnapshot` contains telemetry plus three dedicated signal objects. Each signal has `SignalMetadata`: availability, model name, independent version, prediction timestamp, and age. The enum contains `AVAILABLE`, `UNAVAILABLE`, and `STALE`. Values are not scattered across dozens of nullable fields, and null never silently means normal.

The three signal payloads remain different:

* `AnomalyDashboardSignal` carries an unusualness score and flag.
* `IncidentDashboardSignal` carries every class probability, a top class, and ambiguity.
* `IntegrationFailureDashboardSignal` carries a binary request probability, threshold, decision, and local contributions.

This mirrors Chapter 30's service and Chapter 31's `MlObservation` semantics. Cross-language field containers differ, but `ambiguous`, `anomaly_score`, `failure_probability`, availability, and independent versions mean the same thing.

## 1. Observed system state

The first panel is labeled **OBSERVED** and **Observed system state**. It contains only API latency, error rate, queue depth, database connections, vendor latency, requests/minute, and retry count. No anomaly label appears there.

The next panel derives `normal`, `warning`, or `critical` using the established explicit API-latency and error-rate thresholds. Its mandatory explanation is:

> Severity is based on explicit operational thresholds, not ML probability.

This is normal application logic. A dashboard warning is context for an engineer; it does not automatically justify paging, and this chapter creates no alerting system.

## 2. Anomaly detection

The anomaly panel shows status, score, identity, version, age, and `CURRENT`, `STALE`, or `UNAVAILABLE`. Its definition is exact:

> Measures unusualness relative to the learned healthy baseline.

An Isolation Forest score is not a calibrated probability of an incident, wrongdoing, or failure. The UI therefore never labels it “probability of incident.” If scoring is missing, it says **Prediction unavailable**, not **No anomaly detected**.

## 3. Incident pattern and ambiguity

The classifier panel displays its top pattern and every competing probability, sorted but never discarded. A short HTML bar aids scanning while the percentage remains accessible text.

When the top-two gap is below the configured ambiguity threshold, the panel says:

```text
Interpretation: AMBIGUOUS
The model's top incident classes are close.
Treat this as an ambiguous pattern classification.
```

“AI is unsure” would be anthropomorphic and imprecise. Ambiguity is a deterministic property of this probability vector and threshold. The dashboard does not hide a close `database_pressure` probability behind a slightly higher `vendor_degradation` class.

## 4. Integration request signal

The integration panel displays failure probability, classification threshold, `Elevated failure risk` or `Not elevated`, and independent metadata. Beside it is the critical boundary:

> This is a request-level prediction, not an incident diagnosis.

The model answers a question about a representative integration request. It does not prove why the system-wide incident exists.

## 5. ML capability is not system health

The capability table reports the anomaly detector, incident classifier, and integration model independently. Its aggregate is `all_available`, `partially_available`, or `unavailable`, matching Chapter 31. It is explicitly named **ML capability**, never “system health.”

If only the classifier fails, telemetry, deterministic severity, anomaly scoring, and integration risk remain visible. Guidance uses deterministic telemetry but does not invent a class. If all ML fails, the operational panels and history still work while every ML panel says **Prediction unavailable**. Failure of an advisory subsystem cannot erase evidence.

## 6. Versions and timing

There is no combined “dashboard model version.” The inventory lists the Chapter 28 anomaly artifact, Chapter 29 incident artifact, and integration-failure artifact separately. An unavailable signal has version `unavailable` rather than a guessed last-known identity.

Freshness is also per signal. The service compares each prediction timestamp with the snapshot time under the configurable five-minute window. A classifier updated eight minutes ago says:

```text
Incident prediction: STALE — last updated 8 minutes ago
```

The stale payload may remain useful historical context, but it is not styled or worded as current. Anomaly, incident, and integration ages can differ.

## 7. Model explanation

Chapter 22's exact logistic-regression decomposition supplies actual transformed-value × coefficient contributions. The panel separates factors increasing and decreasing the fitted failure score and is labeled **MODEL EXPLANATION**. It prominently says:

> These contributions explain the fitted model's output. They do not establish the real-world cause of a failure.

A contribution is about the fitted function, not the world. Correlation, proxies, missing variables, and feedback remain possible. We avoid extra multiclass explanation complexity because the request model already teaches the required distinction cleanly.

## 8. Deterministic investigation guidance

Investigation guidance is application code, not a fourth model. A known pattern maps to stable directions: vendor degradation maps to ClearVerify latency/timeout traces and retries; database pressure maps to connection-pool and slow-query telemetry; application regression maps to deployments; traffic spike maps to volume baselines and rate limits.

When `vendor_degradation` and `database_pressure` are the close top classes, the service combines both mappings. Engineers inspect vendor traces **and** database pressure rather than pretending the narrow winner is root cause. Queue thresholds can add queue inspection independently. During classifier outage, only telemetry-derived guidance is included.

## 9. Confirmed evidence

The separate **CONFIRMED** panel has exactly one authority: time-bounded fixture evidence. Before 10:30 it says:

```text
No root cause confirmed yet.
```

At 10:30, trace evidence can say that the ClearVerify call consumed the dominant share of sampled failing-request duration. No model field controls that transition.

The recurring labels—**OBSERVED**, **MODEL SUGGESTS**, and **CONFIRMED**—are more than decoration. They tell an engineer what epistemic claim each value supports.

## Incident playback and the information boundary

`GET /dashboard/incident?time=10:18` selects an evaluated Chapter 26 observation. Its history table includes time, threshold severity, API and vendor latency, queue, anomaly, incident pattern, and request failure probability. These values come from actual fitted educational models.

Playback is a query and server-rendered link rather than complex JavaScript. More importantly, the detailed view is **dashboard state at time t**. At 10:18 it cannot contain traces collected at 10:30. A retrospective is allowed only at or after the final evidence boundary and must be explicitly labeled. This is the UI equivalent of Chapter 27's as-of join: future information is leakage.

The historical-leakage test renders 10:18 and asserts the confirmed trace sentence is absent, then renders 10:30 and asserts it is present. This is one of the most important dashboard tests.

## Minimal model monitoring

The secondary panel borrows Chapter 25 concepts: prediction API latency, unknown categories, input drift status, and last labeled-performance status. It stays below incident evidence so it does not overwhelm response work. Wording matters:

> Input drift detected — investigate model operating conditions.

Drift does not itself mean “model broken.” Delayed outcome performance and service health are distinct observations.

## Security and accessibility

This internal dashboard uses only privacy-minimized capstone telemetry. It displays no member names, account numbers, tokens, raw identity documents, API secrets, artifact paths, or stack traces. The server must still sit behind Harbor's internal authentication and authorization in a real deployment; that infrastructure is outside this educational repository.

Headings establish section order. Tables have headers. Status never relies on color. Percentages retain labels, and stale/unavailable/ambiguous are explicit words. The CSS is progressive enhancement; the page remains meaningful without it.

## Anti-patterns

### One AI status

```text
AI STATUS: RED
```

This destroys signal semantics, uncertainty, evidence hierarchy, model independence, and debugging value. Red could mean anomaly, high request risk, outage, or a threshold. The engineer cannot tell.

### A root-cause banner

Bad:

```text
ROOT CAUSE
ClearVerify
Confidence 82%
```

Better:

```text
MODEL SUGGESTS
Top pattern: vendor_degradation
Probability: ...

CONFIRMED EVIDENCE
Not yet established.
```

A class probability measures resemblance under a fitted classifier. It is not 82% confidence in real-world causation.

### Hiding outage

`Incident pattern: normal` while the classifier is down converts missing evidence into reassurance. The honest state is `Incident classifier unavailable.`

## Executable laboratory

Run the deterministic scenarios from the repository root:

```bash
python examples/chapter_32_engineering_dashboard.py
```

The laboratory fits the existing educational anomaly, incident, and integration models, then builds healthy, early-degradation, compound-pressure, classifier-outage, all-ML-outage, final-confirmation, and independently stale snapshots. It prints observations, deterministic status, capability, model results, confirmation, and investigation guidance. No external API or database is required.

Run the browser application with the repository on `PYTHONPATH`:

```bash
PYTHONPATH=src uvicorn harbor_ml.dashboard.run:app --reload
# open http://127.0.0.1:8000/dashboard
# open http://127.0.0.1:8000/dashboard/incident?time=10:18
```

Tests use an in-process ASGI transport. They assert semantic text rather than exact HTML formatting, cover partial and complete outage, independent staleness, model identities, explanation boundaries, accessible labels, and chronological reveal.

## Dashboard as a debugging interface

A good engineering dashboard has always exposed state, dependencies, timing, versions, errors, and history. ML adds predictions, explanations, and drift; it does not replace full-stack debugging discipline:

```text
Dashboard
   │
   ▼
Observe
   │
   ▼
Form hypothesis
   │
   ▼
Inspect traces/logs/DB/vendor
   │
   ▼
Validate
   │
   ▼
Act
```

## Key takeaways

1. Preserve observation, model suggestion, and confirmed evidence.
2. Keep different ML signals' different semantics.
3. Anomaly scores are not probabilities.
4. Expose competing incident classes and ambiguity.
5. Request-failure predictions are request-level, not incident diagnoses.
6. Keep unavailable ML visibly unavailable.
7. Never display stale predictions as current.
8. Independent versions are essential during debugging.
9. Explanation describes the fitted model, not real-world causation.
10. A good ML-assisted dashboard accelerates investigation without pretending to replace it.

## What comes next: Chapter 33 — Operating the Intelligent Digital Credit Union

Chapter 32 completes Harbor's technical presentation layer. Chapter 33 brings the implemented system together as an operating exam and asks what responsible ML-assisted full-stack engineering looks like as a complete discipline:

```text
DATA CONTRACT → TRAINING → EVALUATION → ARTIFACT → ML SERVICE
→ PHP APPLICATION → DASHBOARD → INCIDENT → INVESTIGATION
→ HUMAN / DETERMINISTIC DECISION → MODEL MONITORING
```

The final synthesis will cover complete validation and run order; incident, ML-outage, stale-model, rollback, and human-review exercises; a security/fairness/explainability checklist; what ML helped with and did not establish; and final architectural principles.

[Previous: Chapter 31 — Integrating the Banking Application](chapter-31-integrating-the-banking-application.md) · [Back to Part VII](README.md) · [Complete contents](../../CONTENTS.md) · [Next: Chapter 33 — Operating the Intelligent Digital Credit Union](chapter-33-operating-the-intelligent-digital-credit-union.md)
