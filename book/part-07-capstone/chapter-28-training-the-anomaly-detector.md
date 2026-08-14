# Chapter 28 — Training the Anomaly Detector

Chapter 27 created a trustworthy, time-aligned dataset for fictional **Harbor Federal Credit Union**. It contains healthy operation, early degradation, vendor degradation, compound pressure, and recovery. This chapter asks:

> Can Harbor train a reproducible anomaly detector on healthy capstone operation and use it to identify when the incident timeline begins departing from that learned baseline?

```text
VALIDATED CAPSTONE DATASET
        │
        ▼
SELECT HEALTHY BASELINE
        │
        ▼
ANOMALY FEATURE CONTRACT
        │
        ▼
TRAIN
        │
        ▼
EVALUATE AGAINST INCIDENT TIMELINE
        │
        ▼
SAVE ARTIFACT + METADATA
```

A careless implementation calls `detector.fit(all_rows)`. The incident then becomes part of “normal,” undermining detection. The engineering question is: **what did Harbor look like before the incident began?**

```text
TRAINING BASELINE                 EVALUATION TIMELINE
healthy historical operation     healthy + degraded observations
                    ≠
```

## Learning objectives

By the end, you can define and validate a baseline; separate training and evaluation; defend a prediction-time feature contract; fit Isolation Forest reproducibly; interpret contamination, scores, flags, healthy false positives, and time-to-detection; persist a fingerprinted artifact; verify round-trip behavior; and explain why unusualness is neither incident class nor root cause.

## From the Chapter 4 laboratory to a lifecycle

Chapter 4 introduced Isolation Forest and the convention “higher transformed score means more unusual.” Chapter 28 retains that algorithm and convention, but adds a temporal split, explicit contract, dataset fingerprint, version, evaluation report, trusted full pipeline, dependency versions, CLI, and round-trip test. This is still a small teaching fixture—not evidence that the detector is production-ready.

The committed timeline has only three observations in its explicit 10:00–10:04 healthy phase. Therefore `MINIMUM_BASELINE_ROWS = 3`. That modest guard catches empty/truncated fixtures; it is emphatically **not** a universal production minimum. A real baseline should span representative load cycles and operational variation.

## The anomaly feature contract

`CAPSTONE_ANOMALY_FEATURES` deliberately selects seven Chapter 27 numerical fields in fixed order:

| Feature | Why it is legitimate at prediction time |
|---|---|
| `api_latency_ms` | current completed application-metric observation |
| `error_rate` | aggregate errors known at observation time |
| `db_connections` | current database connection telemetry |
| `queue_depth` | current observed queue state |
| `vendor_latency_ms` | latest completed vendor telemetry, not a future trace duration |
| `requests_per_minute` | current aggregate traffic telemetry |
| `retry_count` | retries observed by that time |

The small contract is coherent and is available for all three pre-incident rows. Chapter 27 also provides useful trailing-window fields such as `vendor_latency_mean_5m`, `error_rate_mean_5m`, `queue_growth_5m`, and `db_connections_mean_5m`; the earliest rolling observations are unavailable by design, so this detector does not silently impute them merely to enlarge its contract.

Explicitly absent are `incident_type`, `request_failed`, future trace duration, final status, diagnosis, incident phase, and sensitive identifiers. The label may help construct the educational split, but it is not supplied to the unsupervised detector. `build_anomaly_feature_matrix` accepts feature mappings and fixed names only; phase stays beside each result for retrospective display.

## Select and validate the baseline

Chapter 26's editorial marker defines 10:00–10:04 as `healthy`. `select_anomaly_baseline` applies that deterministic rule. The later timeline starts at 10:06. Validation checks:

- the configurable minimum row count;
- exact feature presence and finite values;
- absence of leakage and prohibited sensitive names;
- exclusively `healthy` baseline rows; and
- when evaluation is supplied, `max(baseline time) < min(evaluation time)`.

An empty, short, non-finite, contract-breaking, degraded, or chronologically overlapping baseline fails clearly. Phase is permitted to select known synthetic historical rows; it never enters `X`.

```text
BAD BASELINE
      ↓
BAD DEFINITION OF NORMAL
```

If a supposedly healthy interval contains ongoing degradation, the fitted model may learn degradation as normal. No algorithm can repair a dishonest baseline.

## Fit Isolation Forest reproducibly

The capstone-specific builder creates a scikit-learn `Pipeline` containing:

```python
IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
)
```

`n_estimators=100` makes the ensemble choice explicit; `random_state=42` makes repeated teaching runs stable in the supported dependency range; `contamination=0.05` sets the binary cutoff. The full fitted pipeline is persisted, not merely a few parameters reconstructed later.

### Contamination is a cutoff assumption

Even intended-healthy operation can contain unusual observations. Here contamination tells Isolation Forest which extreme share of baseline scores informs its fitted decision threshold.

> It is not the true incident rate.
>
> It is not a probability.
>
> It is a teaching configuration for deciding which baseline observations are treated as most unusual.

With only three baseline points, granularity is severe. The setting should not be overinterpreted. Sensitivity comparisons teach a tradeoff; they do not estimate incidence prevalence.

## Raw score and flag

scikit-learn's `decision_function` is positive on the inlier side and negative beyond its learned threshold. Harbor exposes:

```python
anomaly_score = -pipeline.decision_function(X)
is_anomaly = pipeline.predict(X) == -1
```

Thus a larger `anomaly_score` means “more unusual.” It is an uncalibrated relative score—not a probability, percentage, severity, or expected loss. The binary flag is the model's thresholded decision. Keep both: scores show progression while flags support an operational decision point.

`CapstoneAnomalyResult` carries timestamp, score, flag, and evaluation-only phase. The first three fields form the operational result; phase helps readers inspect synthetic history and is not model input.

## Teaching evaluation against the timeline

Unsupervised anomaly detection does not have one universally correct supervised label. The synthetic phases let us ask narrower questions:

- how often were recovery-proxy rows flagged?
- when did the first flag appear?
- did unusualness rise during known degradation?
- did recovery move back toward baseline?

The executable lab prints actual fitted outputs rather than hard-coded scores. It computes:

```text
healthy evaluation rows
healthy flagged rows
healthy anomaly flag rate
first_anomaly_timestamp
lead_time = reference_timestamp - first_anomaly_timestamp
```

Because there are only three initial healthy observations, all three train the detector. The distinct recovery rows provide the out-of-baseline healthy evaluation proxy. Recovery is not identical to steady-state healthy operation, so the rate is reported as `recovery proxy` and remains a fixture limitation. A larger study should use earlier healthy rows for training and later stable healthy rows for evaluation.

Chapter 26 identifies a deterministic high-API-latency warning at 10:16 and describes representative request failures beginning in the 10:20 compound-pressure period. The lab uses 10:20 as its documented request-failure milestone and computes lead time from the **actual first fitted flag**, honestly allowing negative lead time if detection is late. It does not tune the model to win the comparison.

The displayed table has this shape:

```text
time    anomaly_score  anomaly?  phase
10:00   (computed)     ...       healthy
10:06   (computed)     ...       early_degradation
10:14   (computed)     ...       vendor_degradation
...
```

Phase is editorial evaluation metadata. It never enters `fit`, `decision_function`, or `predict`.

## Threshold sensitivity

Compare only a few purposeful configurations—for example contamination `0.01` (conservative assumption), `0.05` (teaching default), and `0.10` (more sensitive assumption)—using the exact same baseline, feature order, and timeline.

```text
more sensitive cutoff
→ potentially earlier/more flags
→ potentially more healthy false positives
```

With this tiny baseline, different values may yield identical decisions. Report that honestly. Never choose a cutoff solely because it makes a synthetic timeline attractive. Operational selection needs alert cost, missed-event cost, representative healthy history, and review capacity.

A useful ablation removes vendor latency and retry features, retrains on the same baseline, and compares the first flag and incident-period score progression. Later detection would show that feature choice controls what the detector notices; identical or earlier detection would show other correlated telemetry carries signal. Neither contract is universally superior.

## Artifact, metadata, and trust

Run:

```bash
python scripts/train_capstone_anomaly.py
```

Optional flags are `--output-dir`, `--contamination`, and `--random-state`. Generated, gitignored output is:

```text
artifacts/capstone-anomaly/model.joblib
artifacts/capstone-anomaly/metadata.json
```

Metadata records model name/version/type, UTC training time, dataset name/version/SHA-256, baseline bounds/count, ordered features, contamination, random state, estimator count, and Python/scikit-learn/joblib versions. `harbor-capstone-anomaly-<hash-prefix>` ties the version to a canonical fingerprint of timestamped feature rows and editorial evaluation phase.

The JSON separates `training_metadata` (baseline rule and score orientation) from `synthetic_evaluation_metadata` (healthy/recovery rate, first flag, reference milestone, and lead time). Evaluation facts do not become training features.

Joblib artifacts can execute code while loading. Load only local artifacts whose provenance Harbor controls. The round-trip check scores several rows before and after loading and uses numerical tolerance (`numpy.allclose`), not text serialization equality.

Same dataset bytes and ordering, feature order, random state, estimator count, contamination, and supported dependency environment produce the documented reproducibility level. A changed library version is recorded because cross-version bit-for-bit identity is not promised.

## Run the executable laboratory

```bash
python examples/chapter_28_training_anomaly_detector.py
```

It loads the validated source views, displays the range, selects and validates the healthy baseline, trains, scores the complete timeline, prints progression, computes the recovery-proxy anomaly rate, first flag, and lead time, saves model plus metadata, reloads, and verifies scores. All records are fictional educational telemetry; no production or member data is used.

## Drift: yesterday's normal may not be today's normal

A new vendor, queue design, faster database, or changed traffic pattern can make a once-valid baseline obsolete. The detector may flag legitimate architecture changes. Chapter 25's input and service monitoring therefore applies to the baseline itself: version it, compare distributions, review alerts, and retrain only through a governed workflow. A model recognizes deviation only from the version of normal it was taught.

## Detection is not diagnosis

`anomaly = true` means current multivariate telemetry differs from the learned healthy baseline. It does **not** mean `vendor_degradation` and does not establish cause.

```text
ANOMALY DETECTOR
"Something changed."

INCIDENT CLASSIFIER
"It resembles a known pattern."

TRACE INVESTIGATION
"Here is where the request spent its time."
```

The second task belongs to Chapter 29; the third requires operational evidence. Resemblance is not confirmed root cause.

## Exercises

### Exercise 1 — Baseline selection

Why is `detector.fit(all_capstone_rows)` wrong when rows include the incident? **Answer:** it leaks the event into the learned definition of normal, reducing the meaning of later deviation.

### Exercise 2 — Target

Does training require `incident_type`? **No.** It may define this synthetic historical split but is absent from `X` and unsupervised fitting.

### Exercise 3 — Contamination

What does `contamination = 0.05` mean? It configures cutoff estimation so the most unusual baseline tail influences binary decisions. It is not a 5% incident rate, a probability, or proof that 5% of operation is bad.

### Exercise 4 — Score versus probability

Why is `anomaly_score = 0.72` not “72% probability of an incident”? Negated decision-function output is not calibrated to event frequency and has no probability semantics.

### Exercise 5 — Lead time

First anomaly 10:12 and first failure 10:21 gives `10:21 - 10:12 = 9 minutes` lead time.

### Exercise 6 — False positives

Why can an overly sensitive threshold hurt? Repeated benign alerts consume review capacity, create alert fatigue, and reduce trust even when recall improves.

### Coding exercise — a sensitive detector

Add `capstone-anomaly-sensitive` with a more sensitive contamination assumption. Train on the exact same baseline, score the same timeline, and compare first flag, healthy false-positive rate, total incident-period flags, and operational tradeoff. Do not declare either configuration universally superior.

## Key takeaways

1. An anomaly detector must learn from a carefully chosen baseline.
2. Incident rows must not silently define normal.
3. Only prediction-time features enter the detector.
4. Isolation Forest scores unusualness, not root cause or incident class.
5. Binary flags depend on a cutoff assumption.
6. Healthy false positives and detection lead time both matter.
7. Sensitivity may trade earlier warning for noise.
8. Feature choice shapes what the detector notices.
9. Artifacts must record baseline, fingerprint, contract, and configuration.
10. A model recognizes deviation from the version of normal it was taught.

## What comes next: Chapter 29 — Training the Incident Classifier

Chapter 28 asks when Harbor departs from healthy behavior. Chapter 29 will ask which known incident pattern unusual telemetry most resembles:

```text
LABELED HISTORICAL INCIDENT EXAMPLES
        │
        ▼
CAPSTONE FEATURE CONTRACT
        │
        ▼
MULTI-CLASS CLASSIFIER
        │
        ▼
normal | vendor_degradation | database_pressure
traffic_spike | application_regression
```

It revisits Chapter 5 with the capstone feature contract, artifact metadata, confusion matrix, probabilities, ambiguity, and timeline evolution—without treating a predicted class as confirmed root cause.

[Previous: Chapter 27 — Building the Telemetry Dataset](chapter-27-building-the-telemetry-dataset.md) · [Back to Part VII](README.md) · [Complete contents](../../CONTENTS.md) · [Next: Chapter 29 — Training the Incident Classifier](chapter-29-training-the-incident-classifier.md)
