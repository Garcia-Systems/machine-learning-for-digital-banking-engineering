# Chapter 0 — The ML-Assisted Full-Stack Developer

[← Part I contents](README.md) · [Book contents](../../CONTENTS.md) · [Chapter 1 →](chapter-01-the-digital-banking-data-landscape.md)

## Opening scenario: Monday morning at Harbor

It is Monday morning at Harbor Federal Credit Union. Members begin reporting that a transfer workflow sometimes spins and then fails. Support tickets are intermittent: some members complete the workflow, and the primary health check remains green. At first, the system appears healthy.

Within minutes, however, engineers see rising API latency and increasing HTTP 5xx responses. Calls to an external fintech service are retrying. Database connections are approaching the pool limit, the application queue is growing, and scattered exceptions appear across several services. No single signal tells the whole story.

The on-call developer has thousands of logs, metrics, traces, API responses, and database records. The engineering problem is not simply:

> Is the system broken?

The harder questions are:

> What changed?

> Which signals matter?

> Is this normal variation or the beginning of an incident?

> Where should the developer investigate first?

A threshold alert can identify a known failure condition, but it cannot express every combination of traffic, time of day, deployment version, dependency behavior, and recent history. Machine learning can help transform a large operational record into useful signals: an anomaly score, a likely category, or a forecast. Those signals narrow an investigation; they do not conduct it.

## Learning objectives

After completing this chapter, you should be able to:

1. Explain how deterministic software and ML models solve different kinds of engineering problems.
2. Recognize supervised learning, unsupervised learning, classification, regression, anomaly detection, and clustering in digital-banking scenarios.
3. Distinguish an observation, prediction, diagnosis, decision, and action.
4. Identify financial controls that must remain deterministic even when an application uses ML.
5. Run and interpret a typed Python threshold detector and describe where it stops being sufficient.

## Two ways to turn inputs into outputs

### Deterministic software

Application developers usually write explicit behavior. Given the same inputs and state, a rule produces an expected output:

```text
DETERMINISTIC SOFTWARE

input
  ↓
explicit rules written by developer
  ↓
output
```

This is the right approach when a requirement can and must be stated exactly. Harbor's application should deterministically validate an authenticated session, enforce an account's transfer permissions, reject an amount over a configured limit, and preserve accounting invariants. These are policies and controls, not patterns for a model to infer.

A familiar operational rule is equally direct:

```python
if error_rate > 0.05:
    alert()
```

Its strengths are transparency, predictable execution, and ease of testing. Its limitations are also clear: someone must select `0.05`, and the rule considers no context. A 4% error rate might be highly unusual for a particular endpoint at 03:00, while a brief 6% rate during a known recovery might already be understood.

### Machine learning

ML reverses part of the process. Rather than encoding every relationship explicitly, an algorithm uses historical examples or structure in historical observations to produce a model:

```text
MACHINE LEARNING

historical examples
  ↓
learning algorithm
  ↓
model
  ↓
new observation
  ↓
prediction / score / classification
```

The model is still software. Its inputs, training procedure, evaluation criteria, deployment, monitoring, and allowed use all require engineering decisions. It is useful when a developer cannot reasonably encode every possible pattern by hand. For Harbor, an ML-assisted question might be:

```text
Given latency, traffic, errors, database load,
vendor response times, deployment version,
time of day, and recent history:

Does the current system behavior look abnormal?
```

The result is evidence with measured limitations—not truth and not magic.

## Architecture: ML inside an application system

Harbor's applications continue to execute deterministic rules while telemetry flows to a separate analytical path:

```text
DIGITAL BANKING APPLICATION
          │
          ├── deterministic business rules
          │
          ├── APIs
          │
          ├── database
          │
          └── telemetry
                 │
                 ▼
          MACHINE LEARNING
                 │
                 ▼
       anomaly / prediction
                 │
                 ▼
          developer decision
```

This boundary matters. Telemetry reports what the system observed. A model derives a prediction or score. A person investigates and forms a diagnosis. The person or a deliberately constrained workflow makes a decision, and an application or operator takes an action. Collapsing these stages makes systems difficult to audit and unsafe to operate.

## A conceptual ML vocabulary

No mathematics is required yet. Begin by asking what examples are available and what output would be useful.

### Supervised learning

Supervised learning uses examples paired with known outcomes, often called labels. Harbor might construct historical incident windows labeled `vendor_degradation`, `database_pressure`, or `deployment_regression`. A learning algorithm searches for relationships between each window's telemetry and its label. Labels can be incomplete or mistaken, so ticket quality and incident review practices affect the resulting model.

Two common supervised tasks are:

- **Classification:** choose among discrete categories. *What type of incident is occurring?* A result could rank vendor degradation above database pressure, but an engineer must verify it.
- **Regression:** estimate a numeric value. *What will API latency likely be 10 minutes from now?* A forecast can support capacity planning or an early warning; it cannot guarantee future latency.

### Unsupervised learning

Unsupervised learning works without a supplied outcome label. It finds structure or unusual observations in the inputs. That is attractive when Harbor has abundant telemetry but few consistently labeled incidents. It also makes interpretation important: a mathematically unusual period is not automatically harmful.

Two common unsupervised tasks are:

- **Anomaly detection:** *Does current system behavior differ significantly from normal?* An unusual combination of moderate error rate, long vendor latency, and rapid queue growth might deserve investigation even if no individual threshold fires.
- **Clustering:** *Are there natural groups of member behavior or application events?* Clusters might reveal recurring navigation patterns or families of integration failures. They do not inherently describe people, intent, risk, or causation; engineers must examine and responsibly name them.

Some techniques can be implemented in supervised or unsupervised ways, and real systems may combine them. The important first step is not selecting an algorithm. It is defining the engineering question, the available evidence, the cost of errors, and how a result will be used.

## Harbor engineering examples

| Technique | Harbor question | Possible output | Appropriate engineering use |
| --- | --- | --- | --- |
| Classification | What type of incident is occurring? | Ranked incident categories | Prioritize runbooks and telemetry to inspect |
| Regression | What will API latency likely be in 10 minutes? | Latency estimate and uncertainty | Provide earlier warning of degradation |
| Anomaly detection | Does current behavior differ from normal? | Anomaly score | Surface a period for investigation |
| Clustering | Which events exhibit similar behavior? | Group identifiers | Explore repeated patterns and form hypotheses |

These outputs do not establish root cause. Correlated database pressure may be a consequence of vendor retries, not the initiating fault. A classification is not a diagnosis, and a high anomaly score is not authorization to disable a service.

## Prediction is not the same as decision

Suppose a future model reports:

```text
ML model:
82% probability that the vendor integration is contributing
to the current incident.

Developer:
Investigates vendor telemetry, recent deployments,
request failures, and retry behavior.

Application:
Continues enforcing deterministic security,
authorization, transaction, and accounting rules.
```

The prediction prioritizes attention. The developer may discover that a recent application change, not the vendor, altered retry timing. Diagnosis comes from corroborating evidence such as trace spans, deployment diffs, request outcomes, database wait time, and the vendor's service status.

A useful operational sequence is:

1. **Observation:** vendor latency is 1,900 ms and the queue depth is rising.
2. **Prediction:** a model assigns an 82% probability to vendor contribution.
3. **Diagnosis:** an engineer correlates retry storms with slow vendor responses and rules out a recent deployment.
4. **Decision:** the incident commander chooses a documented degradation procedure.
5. **Action:** the application team applies that procedure and monitors recovery.

Each transition needs ownership, evidence, and an understood failure mode. ML must not bypass authentication, authorization, transaction limits, ledger integrity, regulatory controls, or human approval requirements.

## Executable example: explicit thresholds

Before training a model, examine traditional programming. The Chapter 0 example represents a normal period and an incident period:

```python
NORMAL = {
    "api_latency_ms": 180,
    "error_rate": 0.004,
    "db_connections": 31,
    "vendor_latency_ms": 220,
}

INCIDENT = {
    "api_latency_ms": 2400,
    "error_rate": 0.087,
    "db_connections": 96,
    "vendor_latency_ms": 1900,
}
```

From the repository root, run:

```bash
python examples/chapter_00_thresholds.py
```

Expected output:

```text
normal: 0 threshold violation(s)
incident: 4 threshold violation(s)
  - api_latency_ms: observed 2400 exceeds threshold 500
  - error_rate: observed 0.087 exceeds threshold 0.05
  - db_connections: observed 96 exceeds threshold 80
  - vendor_latency_ms: observed 1900 exceeds threshold 750
```

The reusable function is intentionally small:

```python
def find_threshold_violations(observation, thresholds):
    violations = []
    for metric, limit in thresholds.items():
        value = observation.get(metric)
        if value is not None and value > limit:
            violations.append(
                f"{metric}: observed {value} exceeds threshold {limit}"
            )
    return violations
```

The repository implementation adds type hints and documentation. Run its automated tests with:

```bash
pytest
```

## Explanation of the code

`find_threshold_violations` receives an observation and an explicit mapping of maximum values. It iterates in threshold order, compares a present metric with its limit, and returns human-readable violations. It does not mutate its inputs, call an external service, or depend on randomness. A missing metric is ignored rather than treated as a measured failure; a production telemetry pipeline would need a separate missing-data policy.

The command-line module keeps scenario data and thresholds visible. Both periods pass through exactly the same logic. Tests cover a normal observation, multiple exceeded limits, and a missing metric. This is traditional programming: the developer chose every metric, operator, and limit. Nothing is learned from the examples.

## Engineering interpretation

The result answers a narrow, auditable question: **which observed values exceed the limits we wrote?** It does not answer what changed, identify a cause, or determine which violation appeared first. The four messages are observations relative to policy, not four diagnoses.

Thresholds are often an excellent production choice. They encode hard capacity limits and known service objectives clearly, require little data, and fail in understandable ways. ML should not be introduced merely because it is available.

Complexity appears when normal behavior depends on interacting conditions. Traffic has daily and weekly cycles. A new deployment changes response distributions. Vendor latency matters differently at different request volumes. Retries amplify connection usage and queue depth. Thousands of independent thresholds can become noisy, while fixed limits can miss unusual combinations below each limit.

Later chapters will investigate whether historical data can represent those relationships usefully. Even then, deterministic alerts remain valuable baselines and safeguards. A trained model adds another signal; it does not erase the operational system around it.

## Key takeaways

- Deterministic software expresses known rules; ML learns statistical relationships from examples or observed structure.
- ML is most useful when relevant patterns are too numerous, contextual, or variable to encode reasonably by hand.
- Classification, regression, anomaly detection, and clustering answer different questions and produce evidence—not guaranteed explanations.
- Observation, prediction, diagnosis, decision, and action are distinct engineering stages.
- Financial security, authorization, transaction, and accounting controls remain deterministic.
- Simple thresholds are transparent and useful. They provide a baseline against which later complexity must justify itself.

## Exercises

1. **Conceptual:** For each item, decide whether it is an observation, prediction, diagnosis, decision, or action: (a) the queue contains 4,200 jobs; (b) the team activates a documented vendor-degradation mode; (c) the vendor is forecast to exceed 1,000 ms latency; (d) traces confirm vendor timeouts initiated the retry storm; (e) the incident commander approves traffic reduction.
2. **Technique selection:** Would classification, regression, anomaly detection, or clustering best frame each question? (a) What will tomorrow's peak request rate be? (b) Which known incident family best matches this telemetry? (c) Is this combination of metrics unusual? Explain what data and validation you would need.
3. **Controls:** Name three Harbor application behaviors that should remain deterministic. For each, state why replacing it with a probabilistic model would be inappropriate.
4. **Engineering judgment:** Give one situation where a threshold is preferable to ML and one where an ML-assisted signal may add value. Include the likely cost of a false positive and false negative.
5. **Coding:** Add a `queue_depth` value to `NORMAL`, `INCIDENT`, and `THRESHOLDS` in `examples/chapter_00_thresholds.py`. Add a test showing that equality with a threshold does not violate the current strict `>` rule. Run the example and `pytest`, then explain whether `>=` would better represent the intended operational policy.

## What comes next

Chapter 1 will map the digital-banking data landscape: logs, metrics, traces, events, API responses, and database records. It will examine what these sources actually represent and the quality, privacy, and ownership questions that must be addressed before they become model inputs. We will not assume that having abundant telemetry means having suitable training data.
