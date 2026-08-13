# Chapter 24 — Human-in-the-Loop Systems

Harbor Federal Credit Union's operational review model returns:

```text
review_probability = 0.72
```

A developer proposes:

```python
if probability >= 0.70:
    automatically_reject_transaction()
```

That is a category error. Chapter 13's target was `manual_review_required`: a historical record that an observation entered review. It was not `transaction_invalid`, `fraud`, or `member_wrongdoing`. The score means only that the learned historical pattern resembles cases labeled for review. It supplies neither a verdict nor permission for a punitive action.

A safer boundary is:

```text
observation
    │
    ▼
model score
    │
    ▼
routing policy
    │
    ▼
review queue
    │
    ▼
human investigation
```

This chapter asks:

> How should Harbor design workflows where machine-learning signals assist people and deterministic systems without silently becoming the final authority?

The short answer is to make every responsibility explicit:

```text
OBSERVATION
    │
    ▼
MODEL
    │
    ▼
score / recommendation
    │
    ▼
POLICY
    │
    ├── deterministic action
    ├── human review
    └── no action
```

```text
PREDICTION ≠ POLICY ≠ DECISION ≠ ACTION
```

## Learning objectives

By the end, you can:

1. distinguish prediction, policy, decision, and action;
2. distinguish advisory from authoritative ML;
3. define a human-review workflow;
4. design explicit, versioned routing thresholds;
5. model outcomes separately from model outputs;
6. support and record reviewer override and reason codes;
7. explain automation bias, anchoring, and reviewer disagreement;
8. connect thresholds to review workload and queue capacity;
9. handle unavailable models and unavailable reviewers safely;
10. define deterministic escalation rules;
11. recognize selective-label and reviewer-behavior feedback loops;
12. preserve privacy-minimized, append-only audit history;
13. measure workflow latency and review quality; and
14. explain why deterministic controls remain authoritative.

## Four layers, four meanings

**Prediction** is the model's numerical estimate. **Policy** is a deterministic rule that consumes the estimate and other explicit signals. **Decision** is a recorded conclusion reached by an authorized deterministic system or reviewer. **Action** is the resulting state change. These may be adjacent in code, but they are not interchangeable facts.

```text
MODEL
"historical pattern resembles reviewed cases"

        ↓

POLICY
"route scores above threshold to review"

        ↓

REVIEWER
examines evidence

        ↓

OUTCOME
recorded separately
```

A `review_probability` of `0.68` is a prediction. A configured threshold of `0.70` is policy. Placing a case in a queue is a routing action. `resolved_no_issue` is a review outcome. None of those claims that an underlying transaction was invalid.

## Advisory and authoritative ML

```text
ADVISORY ML                      AUTHORITATIVE ML
model output                     model output
    │                                │
    ▼                                ▼
supports another                 directly determines
 decision-maker                  action
```

Harbor's teaching systems prefer advisory use:

```text
anomaly detector           → investigation signal
incident classifier        → suggested incident pattern
integration failure model  → observability signal
review-routing model       → queue-prioritization signal
```

An authoritative design can be appropriate only after an independently justified policy, careful validation, explicit ownership, failure handling, and applicable governance. It must never emerge accidentally because application code treats a probability as an instruction.

### Deterministic controls remain authoritative

Authentication, authorization, required-field validation, idempotency, transaction-state validation, API contract validation, and security rules express requirements rather than uncertain patterns. Retain:

```text
DETERMINISTIC RULE

if authentication invalid:
    reject request
```

Do not substitute:

```text
ML says authentication probably valid
```

A model can help investigate authentication telemetry; it cannot relax the authentication contract. This preserves the deterministic boundaries established throughout the book and the security controls from Chapter 21.

## Architecture and implementation

Chapter 24 adds `harbor_ml.human_review`, a deliberately small in-memory workflow:

```text
OBSERVATION
    │
    ▼
ML MODEL
    │
    ▼
probability
    │
    ▼
ROUTING POLICY
    │
    ├── not routed
    └── queued
            │
            ▼
       HUMAN REVIEW
            │
            ▼
       review outcome
            │
            ▼
        AUDIT LOG
```

The policy is separate and versioned:

```python
policy = ReviewRoutingPolicy(version="review-policy-v2", threshold=0.70)
routed = should_route(probability, policy)
```

`should_route` is intentionally deterministic and includes scores exactly equal to the threshold. The threshold is not hidden in a controller or model wrapper. These values are educational examples—not recommended financial-operation thresholds—and should be configured, evaluated, approved, and versioned for their actual setting.

### Policy and model metadata

Each case retains:

```text
Which model?     model_name + model_version
Which policy?    policy_version
Which boundary?  routing_threshold
```

Changing `review-policy-v1` from `0.50` to `review-policy-v2` at `0.70` changes routed volume without changing the model artifact or its scores:

```text
MODEL VERSION ≠ POLICY VERSION
```

The laboratory compares both policies against the *same* held-out probabilities. This isolates policy effects from retraining effects.

A richer policy could define three configurable bands:

```text
0.00–0.39  → no action
0.40–0.69  → log / monitor
0.70–1.00  → queue for review
```

Those illustrative boundaries are not universally suitable. The coding exercise builds this explicitly rather than burying it in unrelated application code.

## A small review state machine

The implementation uses generic operational states, not criminal or fraud verdicts:

```text
not_routed
queued → in_review → resolved_no_issue
                   └→ resolved_follow_up
queued / in_review → escalated → in_review
```

Terminal outcomes cannot silently return to `queued`. Reopening would require a separately designed, explicit transition. Resolution requires a pseudonymous reviewer identifier and a supported reason code:

```text
expected_pattern
temporary_vendor_issue
duplicate_operational_signal
insufficient_context
requires_follow_up
```

Structured reasons make aggregation and quality review more consistent than free text alone. Optional notes could exist in a production design, but they require minimization, access control, retention rules, and redaction. They are intentionally absent from this lab.

### Override is legitimate workflow data

```text
model routed case
        │
        ▼
reviewer determines:
no follow-up needed
```

That is an override under this chapter's precise metric definition: a model-policy-routed, resolved case whose outcome is `resolved_no_issue`. It is not an error to suppress. Nor does a high override rate prove model failure; possible explanations include a low threshold, changed label meaning, drift, inconsistent reviewers, or changed operational context.

A case below threshold can still be escalated by an independent deterministic rule or another operational signal. Therefore:

```text
MODEL ROUTING ≠ FINAL OUTCOME
```

Most importantly, resolving `no_issue` never changes the historical `model_probability` from `0.82` to `0`. Keep both:

```text
model prediction  = what the artifact emitted then
review outcome    = what the workflow concluded later
```

They are different facts. Immutable dataclasses and replacement on transition make that distinction visible in the lab.

## What a reviewer should see

A useful, privacy-minimized review view contains:

- relevant observed operational fields;
- model score and model version;
- a safely labeled model explanation, if available; and
- deterministic telemetry and operational context.

```text
REVIEW VIEW
observed fields
model score
Top model contributions       ← model explanation
operational context
Reviewer reason               ← human conclusion
```

Chapter 21's allowlist-first principle applies: do not dump entire member records, secrets, access tokens, passwords, raw authentication cookies, or unnecessary identity into a review view or audit record. Chapter 22's contributions may explain model behavior, but they are **not** the “reason for review” or proof of root cause. Display “Model explanation” separately from “Reviewer reason.”

## Automation bias and anchoring

> Automation bias is the tendency to over-trust an automated recommendation because it came from a system.

```text
MODEL                         REVIEWER
0.84 review probability  →   "the model is probably right"
                              without independently examining evidence
```

This is a central human-in-the-loop risk. A human click does not create meaningful oversight when the interface pressures reviewers to endorse the recommendation.

Bad wording:

```text
HIGH RISK — REVIEW REQUIRED
```

Better wording:

```text
Model routing score: 0.84
Routed under policy v2
Review the supporting evidence independently.
```

Mitigations include showing evidence rather than only a score, using neutral language, training reviewers on limitations, making disagreement easy, sampling model errors for review, and avoiding dramatic red/green verdict styling.

Even neutral scores anchor judgment. One interface might show evidence first and reveal the model explanation later. Another might show both at once but clearly label the recommendation. Neither is universally correct: workflow risk, time constraints, reviewer needs, and the ability to evaluate interface experiments all matter. Harbor should test the combined workflow rather than assume a presentation is neutral.

## Humans disagree and humans fail

Two reviewers can reach different outcomes from identical evidence:

```text
CASE
  │
  ├── reviewer A → follow_up
  └── reviewer B → no_issue
```

A simple agreement rate is:

```text
             same outcome
agreement = ─────────────────────
             double-reviewed cases
```

Agreement does not prove correctness; reviewers can agree for the same mistaken reason. Disagreement does reveal label ambiguity or process inconsistency worth investigating. Outcomes reused as Chapter 13-style labels are therefore potentially noisy.

Humans also face inconsistency, fatigue, bias, incomplete information, time pressure, and anchoring. Chapter 23-style slice analysis should examine route and override rates across relevant operational contexts—for example endpoint or channel—without treating a difference alone as proof of unfairness. Sampling, workload, label quality, and process differences require investigation.

```text
HUMAN-IN-THE-LOOP ≠ PROBLEM SOLVED
```

```text
MODEL STRENGTHS                   HUMAN STRENGTHS
consistent scoring               context
fast processing                  novel situations
large-scale pattern recognition  policy interpretation
                                  investigation and judgment

COMBINED SYSTEM
only useful if responsibilities are explicit
```

## Capacity makes threshold selection operational

Suppose a model flags 500 cases per day while reviewers can process 80. Even a technically strong model is operationally unusable at that operating point.

```text
MODEL THRESHOLD
       │
       ▼
FLAG VOLUME
       │
       ▼
REVIEW CAPACITY
```

The executable queue calculation accepts `observations_per_day`, `predicted_positive_rate`, `reviewers`, and `cases_per_reviewer_per_day`:

```text
routed = ceil(observations × predicted positive rate)
capacity = reviewers × cases per reviewer
backlog change = routed − capacity
```

For 120 routed cases and capacity of 80, backlog grows by 40 per day. A negative change indicates spare capacity, not a negative queue. Real simulations would also carry existing backlog, arrivals by time, priorities, shifts, abandonment, and service-level targets.

The lab reports threshold, routed count, held-out precision, held-out recall, capacity, and backlog change. This adds an essential dimension:

```text
MODEL PERFORMANCE + OPERATING CAPACITY
```

Threshold selection remains a policy decision informed by evaluation—not something accuracy chooses automatically.

### Human unavailable

When no reviewer is available, the case must follow explicit policy: remain queued and age, escalate to backup review, or use an already approved deterministic default. The model must not silently become authoritative merely because staffing is unavailable. Monitor queue length and oldest age so delayed cases are visible.

### Workflow latency

Cases retain `queued_at`, `review_started_at`, and `resolved_at` and derive:

```text
queue_wait_seconds     = review_started_at − queued_at
review_duration_seconds = resolved_at − review_started_at
```

Queue length, average wait, resolution count, override rate, and agreement on double-reviewed cases are system metrics. Human workflow performance is part of ML system performance.

## Deterministic escalation and unavailable ML

Escalation should be inspectable policy, such as:

```text
model score above configured threshold
+
deterministic system-error signal
→ priority review
```

or:

```text
repeated same operational condition
→ escalate
```

These are technical routing examples, not punitive member actions. Independent deterministic rules may route a case even below the model threshold.

When the model is unavailable, preserve `NO MODEL SCORE` as `None`. Never invent `probability = 0`, which falsely claims the model ran and creates misleading audit and training data. In the lab:

```text
deterministic rule triggered → route using deterministic rule
no deterministic trigger     → model-assisted routing disabled; do not route by model
```

Other systems could choose a different documented fallback, but it must be deterministic, tested, visible, and independent of a fabricated ML output.

## Auditability without building event sourcing

Every educational case records the case identifier, model name/version, probability or explicit absence, policy version, threshold, routing source, status, pseudonymous reviewer identifier, reason code, and timestamps. `reviewer-01` is fictional. Audit details use a privacy allowlist.

Important events are appended:

```text
CASE CREATED
    ↓
QUEUED
    ↓
REVIEW STARTED
    ↓
RESOLVED
```

The in-memory repository is not a database or full event-sourcing system. It demonstrates why silently overwriting the only record loses history. Production storage would need durable transactions, concurrency control, authorization, integrity protection, retention, clock discipline, and controlled access.

## Feedback loops and selective labels

If future training uses historical review outcomes while the current model controls which cases receive review, then observation is selective:

```text
model
  ↓
routing
  ↓
reviewed cases
  ↓
new labels
  ↓
future model
```

Never-routed cases may lack detailed outcomes. Training only on reviewed cases teaches the next model about the cases the old model and policy selected, not the full population. This is the selective-label problem introduced in Chapter 13. A changed threshold changes not only workload but also which future labels become observable.

Reviewers also adapt:

```text
reviewers learn model patterns
       │
       ▼
review behavior changes
       │
       ▼
future labels change
```

For example, reviewers may learn to prioritize evidence that resembles commonly displayed model contributions. This can increase apparent agreement without improving correctness. Production evaluation should retain policy/model versions, document process changes, study unrouted samples where lawful and appropriate, and avoid interpreting review labels as an unbiased ground truth.

## Run the laboratory

From the repository root:

```bash
python examples/chapter_24_human_in_the_loop.py
pytest tests/test_human_review.py
```

The laboratory deterministically trains Chapter 13's fictional model, reuses one held-out score vector across two policies, calculates workload, performs a reviewer override, preserves its original score, demonstrates an absent-score fallback, and prints privacy-minimized audit events.

## Exercises

### Exercise 1 — Prediction, decision, or action?

Classify `review_probability = 0.72`, `threshold = 0.70`, “route case to queue,” and reviewer selection `resolved_no_issue`. Distinguish the model prediction, policy configuration, routing action, and human review outcome.

### Exercise 2 — Authority

Why must a model trained on `manual_review_required` not automatically reject a transaction? Identify the unsupported target-semantic leap and the deterministic/human responsibilities it bypasses.

### Exercise 3 — Automation bias

Explain how `MODEL SAYS HIGH RISK` could anchor a reviewer. Propose neutral wording that identifies the score and policy and instructs independent evidence review.

### Exercise 4 — Workload

Given 120 cases routed per day, four reviewers, and 20 cases per reviewer per day, calculate capacity and daily backlog growth. Answer: capacity is 80 and backlog grows by 40 per day.

### Exercise 5 — Override

Should disagreement rewrite the historical model score? Explain why prediction and later outcome describe different facts and why mutation would damage audit and evaluation history.

### Exercise 6 — Feedback loop

Explain how using only reviewed cases for future training creates selective labels. What information is missing about unrouted observations?

### Exercise 7 — Human fallibility

Why does adding a reviewer not guarantee fairness or correctness? Discuss disagreement, fatigue, bias, incomplete context, time pressure, and automation bias.

### Coding exercise — a three-band policy

Add `review-policy-v3` with configurable `monitor_threshold` and `review_threshold`:

```text
below monitor_threshold  → no action
between thresholds       → monitor only
at/above review_threshold → queue for human review
```

Validate that both values are within `[0, 1]` and that the monitor threshold is lower than the review threshold. Then compare v3 routing volume with v1/v2 using the same probabilities, estimate review workload, and add boundary and invalid-configuration tests. Explain why the policy alters operational behavior even though the model artifact is identical.

## Key takeaways

1. Prediction, policy, decision, and action are different layers.
2. Advisory ML should not silently become authoritative.
3. Deterministic security and banking controls remain authoritative.
4. Human review needs explicit states, reasons, and audit history.
5. Reviewer overrides are legitimate data and must not erase predictions.
6. Humans can be inconsistent, biased, fatigued, or wrong.
7. Thresholds affect model metrics and human workload.
8. Model version and policy version must be tracked separately.
9. Human-in-the-loop workflows create feedback loops and selective labels.
10. Responsible ML evaluates the combined human-machine system.

## What comes next: Chapter 25 — Monitoring Models in Production

Chapter 24 establishes the operating workflow. What happens after the model has been deployed for weeks or months? Chapter 25 will cover prediction volume, feature and target drift, performance drift, missing features, unknown categories, latency, error rates, model-version rollout, stale artifacts, retraining triggers, rollback, and monitoring without immediate labels.

```text
PRODUCTION REQUESTS
      │
      ▼
MODEL
      │
      ▼
predictions
      │
      ▼
MONITOR

features
predictions
latency
errors
drift
eventual outcomes
```

Chapter 25 remains planned.

[Previous: Chapter 23 — Bias and Fairness (planned)](README.md) · [Back to Part VI](README.md) · [Complete contents](../../CONTENTS.md)
