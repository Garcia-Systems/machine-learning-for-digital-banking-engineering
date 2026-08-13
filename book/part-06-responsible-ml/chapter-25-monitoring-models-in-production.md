# Chapter 25 — Monitoring Models in Production

> Once Harbor deploys a model, how can engineers tell whether the model and its surrounding system are still behaving as expected?

**Harbor Federal Credit Union** is fictional. Every request, vendor, outcome, and production period in this chapter is synthetic educational material.

[Back to Part VI](README.md) · [Complete contents](../../CONTENTS.md)

## Opening scenario: nothing crashed

Harbor's integration-failure model has run for several weeks. Its API is healthy, requests still return `200`, and prediction latency is normal. Yet engineers see:

```text
average failure probability rising
unknown vendor categories increasing
one endpoint receiving many more high-risk scores
actual failures no longer matching prediction behavior as well
```

Nothing technically crashed. The model may be meeting a world different from its training data. **How do we detect that?**

```text
DEPLOYMENT
is not the end of the ML lifecycle.
```

A model can stay online while becoming less useful:

```text
TRAIN
  ↓
EVALUATE
  ↓
DEPLOY
  ↓
MONITOR
  ↓
INVESTIGATE
  ↓
RETRAIN / ROLLBACK / KEEP
```

Monitoring supplies evidence for the last decision. It does not make that decision automatically.

## Learning objectives

By the end, you should be able to:

1. distinguish service monitoring from model monitoring;
2. monitor prediction volume, inference latency, and API failures;
3. detect missing, malformed, and unknown feature values;
4. compare current inputs with a versioned training baseline;
5. explain feature, prediction, target, and performance drift;
6. distinguish a drift signal from proof of model failure;
7. monitor eventual outcomes and reason about delayed labels;
8. define careful retraining triggers and rollback criteria;
9. monitor model and policy versions independently;
10. explain shadow and staged rollouts;
11. distinguish dashboards from alerts; and
12. explain why automatic retraining can be dangerous.

## Service health is not model health

```text
SERVICE MONITORING

Is the prediction API running?
Is latency acceptable?
Are requests failing?
Is the artifact loaded?

        versus

MODEL MONITORING

Do inputs still resemble training?
Are predictions changing?
Are errors increasing?
Are important slices degrading?
```

A perfectly healthy service can serve a poor model. Conversely, a useful artifact cannot help when its service is unavailable. Harbor monitors both through four layers:

```text
LAYER 1 — SERVICE
availability
latency
HTTP errors
        ↓
LAYER 2 — INPUT DATA
missing fields
invalid values
unknown categories
feature distributions
        ↓
LAYER 3 — MODEL OUTPUT
probability distribution
predicted-positive rate
slice behavior
        ↓
LAYER 4 — EVENTUAL OUTCOMES
accuracy
precision
recall
false positives
false negatives
```

### Layer 1: service metrics and volume

The laboratory uses an in-memory `ServiceHealth`, not a monitoring platform:

```text
prediction_requests_total
prediction_success_total
prediction_error_total
prediction_latency_ms
model_loaded
```

Track predictions per minute or hour. A sudden drop may reveal a broken integration; a sudden increase may reflect traffic or retry behavior and change the model's operating context. **Volume change alone is not proof of drift.** Requests and successes are separate: validation failures count as requests and errors, but not predictions.

Latency percentiles summarize operations. `p50` is median latency. `p95` means 95% of observed requests are at or below that duration. The lab's nearest-rank helper sorts a small window; it is deliberately not a full metrics system. These are operational measurements, not ML performance.

### Layer 2: contract failures and unknown categories

Required fields should fail validation rather than receive arbitrary defaults:

```text
recent_vendor_latency_ms missing
queue_depth invalid
hour_of_day outside range
```

Count these contract errors outside the prediction record. Invalid requests never reach inference.

Harbor's pipeline uses `OneHotEncoder(handle_unknown="ignore")`. That preserves API availability for a new vendor or endpoint, but the model has no learned coefficient for that category:

> The API may continue working while the model effectively lacks learned information about that category.

Therefore Harbor separately counts `unknown_vendor_count` and `unknown_endpoint_count`. An unknown is not necessarily invalid; it is an important compatibility and coverage signal.

## A baseline tied to the artifact

Monitoring comparisons need a reference. `build_training_monitoring_baseline` derives a small JSON-serializable artifact from the exact deterministic training partition used to fit the model. It records:

```text
model_version
dataset_sha256
created_at
numeric summaries
categorical frequencies
```

Each numeric summary contains mean, population standard deviation, minimum, maximum, and p25/p50/p75/p95. Each categorical summary contains observed frequencies, which sum to one per feature. This is sufficient for an interpretable lab; Harbor does not need a feature store.

If persisted, an appropriate generated path is:

```text
artifacts/integration-failure/monitoring_baseline.json
```

That directory remains gitignored under the existing artifact convention. The baseline must travel with its model metadata. Never apply a version-one baseline to version two, and never silently combine their monitoring windows.

Every prediction therefore carries:

```text
model_name
model_version
prediction_timestamp
```

The immutable record also carries `policy_version`. A summary rejects mixed model or policy versions rather than producing an ambiguous aggregate.

## Feature drift: inputs changed

> **Feature drift** means the distribution of model inputs changes over time.

For example:

```text
training: recent_vendor_latency_ms mostly 200–500
current:  recent_vendor_latency_ms mostly 700–1400
```

Or mobile traffic might move from 30% of requests to 70%. This does **not** automatically mean the model is wrong. It means the model operates in a different input environment.

### An intentionally simple numeric heuristic

For each numeric feature the lab computes:

```text
                  current_mean - training_mean
standardized_mean_shift = ----------------------------
                       training_standard_deviation
```

An educational configuration may mark `abs(standardized_mean_shift) >= 1.0` for investigation. This threshold is a **monitoring heuristic**, not proof that the model is invalid and not a universal production standard. If training standard deviation is zero, no change produces zero shift; any change produces signed infinity and an investigation signal. That explicit behavior avoids division by zero and exposes a broken constant-feature assumption.

### Categorical differences

For every category, the code subtracts its baseline frequency from current frequency:

```text
training ClearVerify frequency = 0.40
current  ClearVerify frequency = 0.65
difference                       = +0.25
```

New categories have a baseline frequency of zero. Categories absent from the current window have a current frequency of zero. A configurable absolute difference supplies another teaching heuristic.

A report saying “vendor latency distribution changed” does not establish “the vendor caused degradation.” It identifies where to investigate. Correlation, root cause, and model quality are separate questions.

## Prediction, target, and performance drift

> **Prediction drift** means the distribution of model outputs changes.

```text
approved period: average probability 0.24; predicted failure rate 18%
current period:  average probability 0.51; predicted failure rate 47%
```

The executable comparison uses actual probabilities produced by the trained pipeline. It compares average probability and positive rate at the same threshold. A real system may have become less reliable, features or traffic mix may have changed, a new vendor may have appeared, or the model may be stale. Prediction drift is a signal for investigation, not its conclusion.

> **Target drift** means the actual outcome rate changes.

A historical failure rate of 15% could become 28% even when recorded feature distributions look similar. Labels arrive later, so target monitoring is delayed.

> **Performance drift** means the relationship between predictions and eventual labels changes.

Once outcomes exist, Harbor calculates accuracy, precision, recall, F1, false-positive rate (`1 - specificity`), and false-negative rate (`1 - recall`) with Chapter 17's metrics. Compare like with like:

```text
MODEL VERSION v1
approved deployment baseline recall = ...
current rolling recall              = ...
```

An acceptance criterion must state its metric, window, minimum support, relevant slice, version, and response. “Metrics seem lower” is not an operating criterion.

## Delayed labels are central

At prediction time Harbor knows only request-time facts:

```text
request created
  ↓
prediction
```

Later it receives:

```text
vendor response
  ↓
actual request outcome
```

Other use cases can delay labels by days or months:

```text
PREDICTIONS NOW
      │
      ▼
OUTCOMES LATER
```

Monitoring consequently happens in stages:

```text
IMMEDIATE                 LATER
service health            actual outcome rate
input distributions       accuracy / precision / recall
unknown categories        false positives / negatives
prediction drift          technical-slice performance
```

`PredictionRecord` contains only an identifier, timestamp, model and policy versions, vendor, endpoint, probability, threshold result, and optional eventual outcome. It contains no member identity, account number, email, or full request payload.

`attach_outcome` and `attach_labels` return new frozen records. The original prediction remains unchanged:

```text
prediction
and
later outcome
are separate facts
```

That matches Chapter 24's audit principle. Rewriting the historical score after observing the result would destroy evidence.

### Rolling windows and minimum support

The lab uses deterministic count-based periods of 40 predictions; a deployed implementation might examine the last 100 predictions or a fixed time interval. Each summary reports:

- prediction and labeled counts;
- average probability and predicted-positive rate;
- actual-positive rate when any labels exist;
- performance metrics only after sufficient labels;
- unknown vendor and endpoint counts; and
- a drift signal and status.

Three labeled observations cannot support a strong conclusion. With fewer than the configured minimum, `performance_status` is `insufficient_labels`, metrics are absent, and overall status is `insufficient_data` unless another investigation signal exists. Vocabulary is deliberately bounded to `healthy`, `investigate`, `insufficient_data`, and, for service contexts, `unavailable`; it does not claim “safe.”

## The executable production periods

Run:

```bash
python examples/chapter_25_model_monitoring.py
```

The laboratory trains the Chapter 16 pipeline, recreates its deterministic training split, derives the baseline from those actual training rows, and scores four fixed synthetic periods:

- **Period A — similar to training.** Its inputs and outcomes broadly resemble the reference.
- **Period B — vendor behavior shifts.** Latency rises and one vendor is more common, so computed feature and output signals can change.
- **Period C — a new endpoint appears.** The API continues through the unknown-tolerant encoder while the endpoint counter rises.
- **Period D — performance degrades.** Eventual outcomes are less aligned with prediction behavior; not every metric is forced to fail.

All reported drift values, probabilities, positive rates, and performance metrics are computed. The service simulation also reports deterministic request errors and p50/p95 latency. Repeated runs produce the same periods and results.

The module exposes the following focused utilities:

```python
build_training_monitoring_baseline(...)
calculate_numeric_drift(...)
calculate_categorical_drift(...)
calculate_prediction_drift(...)
summarize_prediction_window(...)
attach_outcome(...)
attach_labels(...)
latency_percentile(...)
compare_shadow_predictions(...)
drift_persists(...)
```

The implementation intentionally does not train, deploy, page, or roll back anything.

## Model age is context, not a verdict

Chapter 16 metadata already records `trained_at`. A dashboard may derive `artifact_age_days` at display time. Age alone does not prove staleness: a stable relationship can remain useful, while a week-old artifact can become obsolete after an integration redesign. Never encode “older than 30 days means retrain” as a universal rule.

Useful retraining investigation triggers include sustained feature drift, performance below approved criteria with adequate labels, important new categories, a material system redesign, a changed label definition, or a vendor integration change. “It is Friday” is not enough unless Harbor deliberately designed, evaluated, and governed a scheduled process.

## Do not turn monitoring into blind deployment

A dangerous loop is:

```text
drift detected
     ↓
automatically train
     ↓
automatically deploy
```

The trigger could reflect corrupted data, wrong labels, a transient incident, changed semantics, inadequate evaluation, incompatible artifacts, or a regression. Blind automation can encode the incident into the next artifact.

Prefer:

```text
monitor
   ↓
trigger investigation
   ↓
prepare candidate model
   ↓
evaluate
   ↓
approve
   ↓
deploy
```

The **production model** currently serves system decisions. A **candidate model** is newly trained and evaluated. A newer timestamp does not make a candidate better.

## Shadow mode and staged rollout

Shadow mode applies both artifacts to the same observation, but only production affects the system:

```text
LIVE REQUEST
    │
    ├── production model → used by system
    └── candidate model  → score only, no effect
```

The local lab compares aligned score arrays and calculates average absolute probability difference and disagreement rate at the same threshold. Candidate disagreement is information, not proof the candidate is better; eventual labels and approved evaluation are still needed. Shadow processing also needs resource, privacy, and access review.

A staged rollout sends a deliberately small traffic portion to an approved new version:

```text
small traffic portion
      ↓
new model
      ↓
monitor
      ↓
expand if acceptable
```

This chapter does not build network deployment infrastructure. Both practices reduce exposure while creating evidence.

## Rollback is normal engineering

> **Rollback means restoring the previously approved model artifact/version.**

Retain the previous model artifact, metadata, and monitoring baseline. Do not delete them immediately after deployment. Configurable rollback criteria might cover a service-error spike, unacceptable latency, materially substandard labeled performance, unexpected slice behavior, or a critical input-contract issue. The rollback procedure must verify artifact compatibility and explicitly record the restored version.

Rollback is not an admission of failure. It is a planned production capability.

## Model, policy, data, and system changes

Chapter 24 separated prediction from deterministic review policy. Monitoring must preserve that separation:

```text
MODEL CHANGED?
POLICY CHANGED?
DATA CHANGED?
SYSTEM CHANGED?
```

A changed policy threshold can change routed volume and observed outcomes without changing probabilities. A system retry change can alter traffic composition. The `model_version` and `policy_version` must be independently queryable, while dataset fingerprints and deployment events provide additional context. Never automatically blame the model after a policy change.

Explanations follow the same rule: coefficients and contributions can change with an artifact, so every explanation remains tied to `model_version`.

## Dashboard, alerting, and persistence

A Chapter 20-style engineering dashboard could add a compact panel:

```text
Model monitoring
Version:
Prediction volume:
Unknown categories:
Drift status:
Last labeled performance:
Artifact age:
```

A **monitoring signal** is not automatically a reason to **page an engineer**. Complete service unavailability, a sudden contract-failure spike, an unknown-category surge, or severe sustained degradation after sufficient labels might be page-worthy under Harbor's operating policy. Modest drift may first belong on a dashboard or ticket.

Avoid paging on a noisy window:

```text
one window exceeds threshold     → observe
several consecutive windows      → stronger investigation signal
```

`drift_persists` implements the simple coding-exercise rule “the last three windows are true.” Persistence reduces noise but delays detection, so urgent service failures should follow separate rules. Thresholds, duration, severity, and escalation belong in configuration and runbooks.

## Monitor technical slices responsibly

Chapter 23 remains planned, but the principle is clear: when labels arrive, compare performance by approved operational slices such as `vendor` and `endpoint`, and compare each with its deployment baseline. Aggregate performance can hide uneven degradation. Small slices need minimum support too.

Do not add sensitive demographic attributes merely to simplify this lab. Chapter 21's security boundary still applies. Log approved monitoring fields rather than entire requests. Prediction/outcome data needs access control, purpose limitation, retention decisions, integrity protection, and auditability even when direct member identity is absent.

## Exercises

### Exercise 1 — Service or model monitoring?

Classify `HTTP 500 rate`, `prediction latency`, `vendor-latency feature distribution`, `recall`, and `unknown endpoint count`. The first two are service signals; the others are model input/performance signals, although unknowns can also reveal an integration-contract change.

### Exercise 2 — Feature drift

Training vendor-latency mean is 300 ms and current mean is 900 ms. What can Harbor conclude?

> Input conditions changed materially.

It cannot conclude, “the model is definitely wrong.” Inspect variation, duration, traffic composition, outputs, and eventual labels.

### Exercise 3 — Prediction drift

Why might predicted-failure rate rise with an unchanged artifact? Consider genuine reliability change, feature distributions, a new vendor, endpoint mix, retries, seasonality, and upstream contract changes.

### Exercise 4 — Delayed labels

Why can Harbor detect input drift before it knows whether recall degraded? Identify which facts exist at request time and which depend on a later vendor response.

### Exercise 5 — Automatic retraining

Explain why `drift detected → train and deploy automatically` could promote corrupted data, altered label semantics, or a transient incident. Design gates between investigation, candidate preparation, evaluation, approval, and deployment.

### Exercise 6 — Shadow mode

What does scoring a request with a non-serving candidate reveal? Why must both models receive the same observations, and why does disagreement not establish superiority?

### Exercise 7 — Rollback

Why retain the previously approved artifact, metadata, and baseline? Describe what Harbor would need to verify and record before restoring them.

### Coding exercise — persistent drift

Add the rule:

```text
investigate only when the same drift condition
appears in 3 consecutive monitoring windows
```

1. Implement it (or extend `drift_persists` to identify named conditions).
2. Test one-, two-, and three-window cases.
3. Run it over synthetic periods.
4. Explain how persistence reduces noise.
5. Explain how waiting can delay detection and why availability alerts stay separate.

## Key takeaways

1. A healthy prediction API does not guarantee a healthy model.
2. Production monitoring covers service, inputs, outputs, and eventual outcomes.
3. Feature drift means input distributions changed; it does not prove model failure.
4. Prediction drift means outputs changed and requires investigation.
5. Performance monitoring requires eventual labels and adequate support.
6. Unknown categories matter when encoders silently ignore them.
7. Model and policy versions must be monitored separately.
8. Retraining should follow evidence, evaluation, and approval—not blind automation.
9. Shadow deployment and staged rollout reduce deployment risk.
10. Rollback is a normal capability, not an admission of failure.

## Part VI conclusion

```text
PART VI — RESPONSIBLE MACHINE LEARNING

Chapter 21
What data should ML be allowed to use?
        │
        ▼
Chapter 22
What can we explain about model behavior?
        │
        ▼
Chapter 23
Where does the model behave unevenly?
        │
        ▼
Chapter 24
Where do humans and deterministic policy remain in control?
        │
        ▼
Chapter 25
How do we know the deployed model is still healthy?
```

Chapter 25 completes Part VI's executable implementation while Chapter 23 remains explicitly planned. Together, these questions keep ML subordinate to evidence, engineering controls, and accountable human decisions.

## What comes next: Part VII — Capstone: The Intelligent Digital Credit Union

The next planned chapter is **Chapter 26 — The Harbor Incident**. It will combine the earlier ideas; it is not implemented here.

```text
Harbor system healthy
        │
        ▼
subtle vendor degradation begins
        │
        ▼
anomaly signals emerge
        │
        ▼
request-failure probabilities rise
        │
        ▼
incident classifier suggests a pattern
        │
        ▼
dashboard assembles evidence
        │
        ▼
engineer investigates
```

The capstone will require readers to distinguish **observation**, **prediction**, **explanation**, **hypothesis**, **diagnosis**, **policy**, and **action**. None is a substitute for the others.

[Previous: Chapter 24 — Human-in-the-Loop Systems](chapter-24-human-in-the-loop-systems.md) · [Back to Part VI](README.md) · [Complete contents](../../CONTENTS.md) · Next: Chapter 26 — The Harbor Incident *(planned)*
