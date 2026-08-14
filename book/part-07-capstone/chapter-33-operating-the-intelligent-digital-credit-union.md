# Chapter 33 — Operating the Intelligent Digital Credit Union

The **Harbor Federal Credit Union** engineering team now owns the complete ML-assisted architecture:

```text
CAPSTONE DATASET
      │
      ├── anomaly detector
      ├── incident classifier
      └── integration-failure model
             │
             ▼
       Python ML service
             │
             ▼
        PHP application
             │
             ▼
     Engineering dashboard
```

Everything works. Production engineering, however, is not “start the service and admire the predictions.” Can Harbor reproduce the models, identify their versions, recognize a missing model or stale output, continue without ML, detect drift, restore an earlier artifact, preserve human disagreement, and reconstruct an incident later?

```text
BUILDING THE SYSTEM
        ≠
OPERATING THE SYSTEM
```

This final chapter is a practical operating exam, not a summary. Every dataset, vendor, member, incident, and result is fictional and synthetic. Completing it demonstrates the repository's educational contracts; it does **not** establish readiness for a real financial system.

## Central question

> What does it actually mean to operate a full-stack digital banking system in which machine learning assists engineering without becoming an uncontrolled decision-maker?

It means treating ML as one measured, replaceable, failure-prone subsystem. Deterministic controls remain authoritative; model semantics and uncertainty remain visible; engineers confirm diagnoses with direct evidence.

## Learning objectives

By the end, you can:

1. run the complete Harbor ML-assisted stack;
2. validate capstone data before training;
3. train reproducible, evaluated artifacts;
4. inspect dataset fingerprints, model versions, types, and timestamps;
5. start and test the ML service and PHP integration;
6. operate the dashboard without confusing observation, suggestion, and confirmation;
7. contain partial and complete ML failure and reject contract violations;
8. recognize stale output and unknown categories;
9. inspect monitoring, shadow comparison, explanations, and operational slices;
10. execute review, retraining, rollback, security, and audit exercises;
11. state what the complete system can and cannot legitimately claim.

## Final architecture and lifecycle

```text
                         HARBOR FEDERAL CREDIT UNION

                                  Members
                                     │
                                     ▼
                           Web / Mobile Experience
                                     │
                                     ▼
                            PHP Application Layer
                                     │
               ┌─────────────────────┼─────────────────────┐
               │                     │                     │
               ▼                     ▼                     ▼
          Database              Vendor APIs          ML Gateway
                                                           │
                                                           ▼
                                                Python ML Service
                                                           │
                           ┌───────────────────────────────┼───────────────────────────────┐
                           │                               │                               │
                           ▼                               ▼                               ▼
                    Anomaly Detector              Incident Classifier          Integration Failure
                           │                               │                               │
                           └───────────────────────────────┼───────────────────────────────┘
                                                           ▼
                                                Engineering Signals
                                                           │
                                                           ▼
                                                Monitoring Dashboard
                                                           │
                                                           ▼
                                               Engineer Investigation
                                                           │
                                                           ▼
                                           Deterministic / Human Action
```

```text
DATA → TRAIN → EVALUATE → DEPLOY → MONITOR → INVESTIGATE → RETRAIN/ROLLBACK
```

## Repository runbook

Run commands from the repository root unless a step says otherwise. These are the repository's actual entry points.

### Step 1 — Validate the repository

```bash
pytest
cd php && composer test && composer lint
```

Python tests cover the models, service, incident chronology, and dashboard. PHPUnit covers typed adapters and deterministic application fallback; Composer lint syntax-checks PHP source.

### Step 2 — Build and validate the capstone dataset

```bash
python examples/chapter_27_building_telemetry_dataset.py
```

Verify source inventories, as-of temporal integrity, the feature contract, row count, leakage validation, and the output SHA-256 fingerprint. A missing source is not silently converted to zero.

### Step 3 — Train the capstone anomaly detector

```bash
python scripts/train_capstone_anomaly.py
```

Inspect `artifacts/capstone-anomaly/model.joblib` and `metadata.json`. Verify the baseline range, feature list, dataset hash, `IsolationForest` type, trained timestamp, model version, and reported detection behavior. A structural `PASS` is not a promise that every metric is desirable.

### Step 4 — Train the incident classifier

```bash
python scripts/train_capstone_incident_classifier.py
```

Verify class counts, held-out accuracy, majority baseline, macro F1, per-timestamp probabilities, ambiguity, artifact, metadata, and independent version. Inspect the confusion matrix/evaluation in the Chapter 29 laboratory and tests; do not reduce evaluation to one score.

### Step 5 — Train integration-failure prediction

Reuse Chapter 16 rather than creating another trainer:

```bash
python scripts/train_integration_failure_model.py
```

Its saved pipeline contains preprocessing and the fitted classifier. Verify its numerical and categorical contracts, dataset SHA-256, evaluation metrics, threshold, and version.

### Step 6 — Inspect artifact inventory

The master lab prints each model name, version, model type, and shortened dataset hash. Its internal inventory also retains the controlled local path and trained timestamp:

```bash
python examples/chapter_33_operating_harbor.py
```

Expected shape:

```text
MODEL                           VERSION                         STATUS
harbor-capstone-anomaly        independently versioned         available
harbor-capstone-incident       independently versioned         available
harbor-integration-failure     independently versioned         available
```

Local operator tooling may show controlled paths. Public prediction and health responses must not expose artifact filesystem paths.

### Step 7 — Start the ML service

After generating all three artifacts:

```bash
PYTHONPATH=src uvicorn harbor_ml.service.app:app --reload
curl -s http://127.0.0.1:8000/api/v1/health
```

Verify `status=ok`, `ready=true`, three independently loaded models, their versions, and applicable feature-contract versions. Chapter 30 does not define `/api/v1/models`; health is the supported inventory boundary.

### Step 8 — Exercise prediction endpoints

Use the complete request examples in Chapter 30 or its executable:

```bash
python examples/chapter_30_capstone_ml_service.py
```

For every response, state its semantics:

| Endpoint | Output meaning |
| --- | --- |
| `/api/v1/score/telemetry-anomaly` | anomaly score and fitted detector flag |
| `/api/v1/predict/incident` | probabilities over known synthetic incident classes |
| `/api/v1/predict/integration-failure` | probability that a request fails at prediction time |

None is a root-cause verdict or an authorization decision. Also omit or rename a required field: a `422` schema rejection is a controlled **contract violation**, not model unavailability.

### Step 9 — Run PHP integration

```bash
cd php
composer install
composer test
php examples/chapter_31_banking_application_integration.php
```

Verify typed responses, independent model versions, all/partial/unavailable capability states, and an unchanged deterministic vendor result. ML observation is advisory and failures are contained per signal.

### Step 10 — Run the dashboard

```bash
python examples/chapter_32_engineering_dashboard.py
PYTHONPATH=src uvicorn harbor_ml.dashboard.run:app --reload --port 8001
```

Inspect healthy, early-incident, compound-pressure, classifier-down, all-ML-down, retrospective, and stale snapshots. Keep these headings distinct:

```text
OBSERVED          measured system facts
MODEL SUGGESTS    fitted resemblance or probability
CONFIRMED         time-eligible direct evidence
```

## Final incident exercise

Replay Chapter 26 from its first timestamp. At each major timestamp record observed evidence, anomaly result, incident probabilities, failure probability, ML availability, all model versions, engineering hypothesis, next investigation, and confirmed evidence.

| Time | Observed | ML suggests | Hypothesis | Confirmed? | Next action |
| --- | --- | --- | --- | --- | --- |
| 14:00 | low latency/error/queue; ordinary vendor latency | normal; no anomaly | system appears healthy | health telemetry only; no future cause | continue monitoring |
| first anomaly | _reader records_ | _reader records_ | _reader records_ | _reader records_ | _reader records_ |
| compound pressure | _reader records_ | _reader records_ | _reader records_ | _reader records_ | _reader records_ |
| trace timestamp | _reader records_ | _reader records_ | _reader records_ | _reader records_ | _reader records_ |

### Exercise A — Healthy operation

What directly indicates health? Does ML add material information? Why is a normal prediction not a guarantee? Direct measurements support a bounded, current statement; a learned normal classification cannot exclude unobserved failure.

### Exercise B — First anomaly

Identify what changed and which detector flagged it. No incident category is confirmed. Inspect current logs, traces, queue behavior, database telemetry, and vendor evidence next.

### Exercise C — Incident classifier signal

When `vendor_degradation` becomes the top class, Harbor may say:

> Current telemetry most resembles the synthetic vendor-degradation examples.

Harbor may **not** say “ClearVerify has been proven to be the root cause.”

### Exercise D — Compound pressure

When `vendor_degradation` and `database_pressure` are close, ambiguity is useful information: several interacting symptoms are present. Preserve both investigation paths rather than forcing false certainty.

### Exercise E — Confirmed trace evidence

At the later evidence timestamp compare the earlier **model hypothesis** with the now-eligible **trace evidence**. The prediction prioritized inquiry; the trace supports diagnosis. Historical playback must never reveal that trace early.

## Failure drills

### 1. Incident classifier unavailable

The master lab constructs an in-process runtime with anomaly and integration models loaded but incident classification absent. Verify degraded health, `503` only on incident prediction, success on other endpoints, partial PHP/dashboard capability, and unchanged deterministic workflow.

### 2. All ML unavailable

Verify degraded/not-ready health and `503` model endpoints. The PHP application and dashboard retain deterministic vendor behavior and telemetry. Lost capabilities are anomaly assistance, classification assistance, and failure prediction—not authentication, authorization, banking logic, or core observability.

### 3. Contract mismatch

Send a request with an omitted or renamed field. Expect `422` at the API boundary or a typed PHP contract violation for a malformed response. A contract mismatch says caller and service disagree; unavailability says an otherwise valid capability cannot currently run. Alert and remediate them differently.

### 4. Stale prediction

Chapter 32's final scenario supplies a prediction older than its five-minute freshness limit and displays `STALE`. Stale means a prediction exists but no longer describes the current operating moment; unavailable means no prediction was obtained. Neither becomes a low-risk score.

### 5. Unknown category

Send a new fictional vendor or endpoint to the integration pipeline. `handle_unknown="ignore"` allows inference, but Chapter 25 monitoring must increment the unknown-category count. No crash demonstrates contract tolerance, not strong model understanding or validity.

## Monitoring, retraining, and shadow comparison

```bash
python examples/chapter_25_model_monitoring.py
```

Compare Period A (baseline-like), B (drift), C (unknown category), and D (performance degradation). Feature, category, prediction, availability, and latency signals can be immediate. Accuracy, precision, recall, and label-dependent degradation require eventual outcomes. A signal justifies investigation; persistent, explained evidence may justify considering retraining. Neither justifies automatic deployment.

Train a **candidate** into a temporary directory, leaving production untouched:

```text
monitoring signal → candidate training → evaluation → compare production vs candidate
```

Score identical observations with production and candidate and report average absolute probability difference, threshold disagreement rate, and held-out metrics when labels exist. Is the candidate better? Recency cannot answer; relevant evaluation, slice behavior, operational effect, and review must.

## Rollback laboratory

The master lab simulates:

```text
production v1 → deploy candidate v2 → unacceptable monitoring → restore v1
```

It retains v1's artifact and metadata, loads a candidate identity, then reloads the controlled v1 files and checks health reports the restored version. In a live integration, repeat the health and prediction checks and verify PHP/dashboard metadata also show v1.

```text
ROLLBACK
is ordinary production engineering

not
failure of ML
```

A model rollback is distinct from a Chapter 24 policy rollback. Harbor may retain an artifact while changing `review threshold` or `ambiguity threshold/policy`. Always ask: **Was behavior caused by model version or policy version?** Record both.

## Human-review and automation-bias exercise

```bash
python examples/chapter_24_human_in_the_loop.py
```

Generate a probability, apply a versioned policy threshold, route the case, begin review, override routing/model implication when evidence supports it, choose a constrained reason code, and inspect the append-only audit trail.

```text
MODEL PREDICTION remains historically unchanged
REVIEW OUTCOME is a separate fact
```

For `review_probability = 0.91`, suppose direct evidence shows a benign expected pattern. Should the reviewer accept the model merely because its score is high? **No.** Record the evidence-based outcome and reason without rewriting the historical score.

## Explainability exercise

```bash
python examples/chapter_22_explainability.py
```

For one integration request print its prediction, top positive contributions, and top negative contributions. Confirm the exact linear decomposition reproduces model probability. Supported: “Elevated recent vendor latency increased the fitted model score.” Unsupported: “Elevated vendor latency caused the request failure.” Explanation describes fitted behavior, not causation.

## Operational slice exercise

The master lab calculates support, base rate, and threshold error rate for `vendor` and `endpoint`; it introduces no demographic data. Ask which slices have enough observations, where error rates and base rates differ, and what data or instrumentation needs investigation. Small synthetic slices cannot establish fairness, absence of harm, or causal vendor differences.

## Security exercises

```bash
python examples/chapter_21_data_security.py
python scripts/audit_ml_data.py
```

Temporarily add `access_token` or `account_number` to a prediction mapping and verify the allowlist guard rejects it. Ask: **Did the ML problem actually need this field?** Remove it after the exercise.

Artifacts load only from explicitly configured, controlled local paths. Harbor provides no arbitrary artifact-upload endpoint and never loads unknown joblib files. Do not turn this into a malicious-pickle exercise.

## Operational checklist

### Before training

- [ ] dataset provenance known
- [ ] feature contract reviewed
- [ ] leakage checks pass
- [ ] sensitive-field checks pass
- [ ] label semantics documented

### Before deployment

- [ ] meaningful baseline and evaluation complete
- [ ] important supported slices reviewed
- [ ] metadata complete and model version assigned
- [ ] rollback artifact available
- [ ] API contract tested

### During operation

- [ ] service availability and inference latency monitored
- [ ] unknown categories and feature/prediction drift monitored
- [ ] model versions and stale output visible
- [ ] downstream policy version visible

### During incident

- [ ] observe before diagnosing
- [ ] keep model semantics distinct and preserve ambiguity
- [ ] inspect traces, logs, database, and vendor evidence
- [ ] use ML to prioritize investigation
- [ ] record confirmed evidence separately

### During retraining

- [ ] investigate trigger and validate new dataset
- [ ] train and evaluate a candidate
- [ ] shadow/compare where appropriate
- [ ] approve deliberately and retain rollback path

## Model/system responsibility matrix

| Concern | Deterministic software | ML | Human |
| --- | --- | --- | --- |
| Request schema validation | primary | no | review design |
| Authentication | primary | no | policy/governance |
| Anomaly signal | support | primary model role | investigate |
| Incident suggestion | support | primary model role | diagnose |
| Root-cause confirmation | evidence tools | no | engineer |
| Review routing | policy | advisory score | review |
| Final review outcome | workflow | no | reviewer |
| Model monitoring | tooling | monitored object | engineer |
| Retraining approval | tooling | candidate | human/process |

## What ML helped Harbor do—and what it did not do

ML helped Harbor spot unusual telemetry; classify known operational patterns; identify useful predictive signals; estimate integration-failure risk; study behavioral journeys; forecast demand; estimate database latency; and prioritize engineering investigation.

Equally important, ML did **not** prove root cause; replace authentication or authorization; determine transaction validity; prove fraud or wrongdoing; determine member worth; replace database diagnostics, traces, or human review; remove deployment risk; make biased data unbiased; or make stale models trustworthy.

## The complete operating contract

```text
DATA CONTRACT
      │
      ▼
TRAINING
      │
      ▼
EVALUATION
      │
      ▼
VERSIONED ARTIFACT
      │
      ▼
ML SERVICE
      │
      ▼
PHP APPLICATION
      │
      ▼
ENGINEERING DASHBOARD
      │
      ▼
INCIDENT / REVIEW WORKFLOW
      │
      ▼
ENGINEERING INVESTIGATION
      │
      ▼
DETERMINISTIC OR HUMAN DECISION
      │
      ▼
MODEL MONITORING
      │
      ▼
RETRAIN / ROLLBACK / KEEP
```

## Final engineering principles

1. **Start with the engineering question, not the algorithm.**
2. **Use ML only when deterministic logic is insufficient for the question.**
3. **Define prediction time before defining features.**
4. **Treat leakage as a correctness defect.**
5. **Compare every model to a meaningful baseline.**
6. **Persist preprocessing with the model.**
7. **Version models independently from APIs and policies.**
8. **Treat ML services like normal external dependencies: timeout, validate, observe, degrade safely.**
9. **Never turn unavailability into a fake low-risk score.**
10. **Preserve uncertainty and ambiguity.**
11. **Model explanations describe model behavior, not causation.**
12. **Use direct system evidence to establish diagnosis.**
13. **Evaluate the full human-machine workflow, not only the model.**
14. **Monitor models after deployment and retain rollback paths.**
15. **The purpose of ML is to improve measurable engineering outcomes, not to make the architecture look intelligent.**

## Where to go from here

Apply the same discipline to approved real datasets; deepen statistics and ML theory; study observability, experimentation and causal inference, data engineering, model governance, distributed systems, and API design.

> The next step is not necessarily a more complicated model.

Harbor is not intelligent because it contains machine learning.

Harbor is intelligently engineered because each subsystem has a clear responsibility, every prediction has defined semantics, deterministic controls remain authoritative, uncertainty stays visible, models are measured and monitored, and engineers retain the evidence and judgment needed to operate the system safely.
