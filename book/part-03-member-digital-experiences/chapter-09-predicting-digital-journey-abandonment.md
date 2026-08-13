# Chapter 9 — Predicting Digital Journey Abandonment

> Part III — Machine Learning for Member Digital Experiences

[Part III overview](README.md) · [Complete contents](../../CONTENTS.md) · [Previous: Chapter 8](chapter-08-understanding-member-behavior.md)

## Central question

> Can Harbor estimate, while a digital journey is still in progress, whether that journey is at elevated risk of ending without completion?

Chapter 8 established events, sessions, and journeys. This chapter adds supervised binary classification:

```text
RAW EVENTS → SESSIONS / JOURNEYS → PARTIAL JOURNEY STATE → PREDICTION
```

One Harbor Federal Credit Union member follows:

```text
transfer_started → recipient_selected → transfer_reviewed → transfer_completed
```

Another recorded journey follows:

```text
transfer_started → recipient_selected → help_opened → session_ended
```

Harbor's Digital Banking team asks whether the second kind of outcome can be estimated **while the member is still in the workflow**:

```text
CURRENT JOURNEY STATE
transfer_started
recipient_selected
elapsed_time = 92 sec
help_opened = true
error_count = 1
channel = mobile
        │
        ▼
      MODEL
        │
        ▼
probability of abandonment
```

The model must predict before the outcome exists. Giving it `transfer_completed = false`, `session_ended = true`, or `journey_abandoned = true` would reveal the answer.

All relationships and results here are deterministic synthetic teaching material. They describe no real members and do not estimate real behavior.

## Learning objectives

By the end, you should be able to:

1. define journey abandonment operationally;
2. distinguish a target from a prediction-time feature;
3. explain temporal leakage;
4. create partial-journey snapshots;
5. derive numerical, categorical, and boolean features;
6. train a binary classifier on partial states;
7. interpret predicted probabilities;
8. understand thresholds;
9. explain false positives and false negatives in context;
10. distinguish prediction from intervention;
11. recognize the risk of aggressive action on uncertainty; and
12. explain why prediction time must be explicit.

## An operational outcome, not a psychological claim

For this teaching journey, `journey_completed = 1` when `transfer_completed` appears after the prediction point. We encode the model target as:

```text
journey_abandoned = 0  eventually completed
journey_abandoned = 1  started, but ended without transfer_completed
```

> **This is an operational event definition.**

It does not prove why a member stopped. Deliberate choice, interruption, an application issue, insufficient information, deciding to finish later, timeout, or network loss could all produce the same record. The label must not be translated into motivation.

## The prediction point is the contract

Harbor chooses a reproducible contract:

> After the member reaches `recipient_selected`, predict whether the transfer journey will eventually complete.

```text
transfer_started
      │
      ▼
recipient_selected
      │
      ▼
PREDICTION POINT ── information available so far
      │
      ▼
future journey events
```

Choosing “some time during a transfer” is insufficient. Different prediction points expose different information, populations, and product opportunities. A model trained at `recipient_selected` must also be called at that point in an application. Journeys that never reach it are not eligible observations; they answer a different question.

## Journey snapshots

A **journey snapshot** freezes everything known at prediction time:

```python
{
    "session_id": "session-1042",
    "channel": "mobile",
    "elapsed_seconds": 74,
    "events_so_far": 4,
    "help_opened": False,
    "error_count": 0,
    "search_count": 1,
    "previous_failed_login": False,
    "journey_abandoned": 0,
}
```

The last field is a historical training label, never an input. `JOURNEY_FEATURES` fixes the only legal feature order:

```text
elapsed_seconds, events_so_far, error_count, search_count,
help_opened, previous_failed_login, channel
```

No member identity is needed.

## Temporal leakage: the central engineering hazard

Temporal leakage occurs when training features contain information unavailable when a real prediction would be made.

```text
PAST / PRESENT
can be used as features
        │
        ▼
PREDICTION TIME
        │
        ▼
FUTURE
cannot leak backward
```

Invalid fields include:

```text
transfer_completed       session_ended             final_event_name
journey_duration_seconds events_after_prediction_point
journey_abandoned
```

Even `journey_duration_seconds` leaks if computed from the full journey. Use `elapsed_seconds` at prediction instead. An invalid model can look excellent because it has effectively been handed the answer.

The implementation makes two windows explicit:

```text
FEATURE WINDOW
transfer_started ---------------- recipient_selected

LABEL WINDOW
recipient_selected ---------------- journey end
```

Feature construction reads the first window. It may also check earlier session history for `previous_failed_login`, which is already known. Label construction alone checks the later historical window for `transfer_completed`. Labels can inspect history after prediction during supervised training because no label is supplied at inference.

## Building snapshots from Chapter 8 events

The code reuses `data/harbor_member_events.csv`:

```python
events = load_member_events("data/harbor_member_events.csv")
sessions = group_events_by_session(events)
snapshots = build_transfer_snapshots(sessions)
```

For each session, `build_transfer_snapshots`:

1. sorts events and validates one session and channel;
2. locates the first `transfer_started`;
3. locates the first later `recipient_selected`;
4. freezes features through that timestamp;
5. counts only pre-prediction help, transfer failures, and searches;
6. inspects later events solely to construct the label; and
7. excludes a journey without the defined prediction point.

The fixture generator retains Chapter 8's event universe and adds 300 qualifying synthetic transfers, for 330 total. Completion is sampled from noisy, overlapping relationships. Errors, searches, help, earlier login failure, elapsed time, and channel may shift synthetic probability, but none determines the label. Fixed seed `808` makes regeneration repeatable. These invented relationships must not be presented as findings about actual digital banking behavior.

### A strong leakage test

The test suite creates two sessions with identical events through `recipient_selected`, then completes one and ends the other. Their feature rows are identical; only their labels differ. If a future event changes either feature row, the test fails. This is more valuable than merely checking that a suspicious column name is absent.

## Features and preprocessing

The intentionally small schema contains:

| Type | Features | Meaning |
|---|---|---|
| Numerical | `elapsed_seconds`, `events_so_far`, `error_count`, `search_count` | observed counts or elapsed time by prediction |
| Boolean | `help_opened`, `previous_failed_login` | whether the recorded event occurred by prediction |
| Categorical | `channel` | `web` or `mobile` |

`help_opened` means the help interface was opened. It does not mean the member was confused. Longer elapsed time is observed time, not proof of frustration.

As in Chapter 7, a `ColumnTransformer` prevents ad hoc preprocessing:

```text
numeric → StandardScaler ┐
boolean → pass through   ├→ combined matrix → LogisticRegression
channel → OneHotEncoder  ┘
```

`OneHotEncoder(handle_unknown="ignore")` produces channel indicator columns and safely transforms a category absent from training. Booleans remain zero/one-compatible values. The entire transformer and deterministic logistic regression live in one `Pipeline`, so training and prediction apply the same operations.

Logistic regression is reused deliberately: novelty here is time framing, not algorithm selection.

## Executable laboratory

Run:

```bash
python examples/chapter_09_journey_abandonment.py
```

The committed fixture produces:

```text
Eligible transfer journeys: 330
Completed: 150
Abandoned: 180
Training observations: 247
Test observations: 83

Accuracy: 0.554
Confusion matrix:
[[17 21]
 [16 29]]
False positives: 21
False negatives: 16
```

The classes are reasonably balanced for teaching, not altered to maximize accuracy. If nearly every journey completed, an always-complete classifier could report attractive accuracy while finding no abandonment. Always inspect class balance beside accuracy. This modest result is also a useful reminder: a non-leaking behavioral model need not be highly accurate to be executable.

## Probabilities and scenarios

`predict_proba` returns the fitted model's probability for positive class `1`:

```text
Smooth journey
web, 35 seconds, 3 events, no help/errors/searches/prior login failure
probability=0.446, class at 0.50=0

Journey with friction
mobile, 160 seconds, 7 events, help opened, 2 errors, 2 searches
probability=0.880, class at 0.50=1

Ambiguous journey
web, 80 seconds, 4 events, one search, no errors
probability=0.459, class at 0.50=0
```

These are actual output from this fitted synthetic model, not guarantees about an individual.

### Thresholds are product policy

The laboratory computes the ambiguous probability once and reuses it:

```text
same probability = 0.459
threshold 0.30 → at risk
threshold 0.50 → not at risk
threshold 0.70 → not at risk
```

A lower threshold marks more journeys at risk; a higher threshold marks fewer. Changing the threshold does not change model probability. The choice depends on the cost and intrusiveness of the downstream action and must be evaluated rather than hidden in application code.

## Errors in member-experience terms

A **false positive** is:

```text
MODEL predicts likely abandonment
ACTUAL journey completes
```

If the prediction triggers UI help, Harbor may show an unnecessary prompt, distract the member, or degrade an otherwise successful experience.

A **false negative** is:

```text
MODEL predicts likely completion
ACTUAL journey ends without completion
```

Harbor may miss an opportunity to offer optional assistance or capture useful diagnostics. The confusion matrix uses rows for actual `0/1` and columns for predicted `0/1`, making `[0,1]` false positives and `[1,0]` false negatives.

## Prediction is not intervention

```text
MODEL OUTPUT
0.68 abandonment probability
        │
        ▼
PRODUCT POLICY
What, if anything, should Harbor do?
```

Low-risk possibilities include enriching analytics, logging approved journey context, surfacing optional help, measuring friction, or flagging aggregate patterns for product review. Harbor should not force a different workflow, block a transfer, change financial terms, or deny service.

The model must never control credit, eligibility, pricing, transaction authorization, or other financial decisions. Acting aggressively on an uncertain score can make the product worse, especially through false-positive interruptions.

### Interventions change behavior

```text
model predicts high abandonment risk
        │
        ▼
optional help prompt appears
        │
        ▼
member completes transfer
```

After an intervention, outcomes may no longer follow the historical relationship used for training. Logging predictions, policies, exposures, and outcomes and evaluating product changes experimentally are separate work.

> **Prediction and intervention are separate system-design problems.**

This chapter predicts; it does not establish that an intervention causes improvement.

## Architecture and full-stack boundaries

```text
                    HARBOR MEMBER JOURNEY
Member
  │
  ▼
Web / Mobile
  │
  ▼
behavior events
  │
  ▼
journey state builder
  │
  ▼
partial journey snapshot
  │
  ▼
abandonment model
  │
  ▼
risk probability
  │
  ▼
analytics / optional product assistance
```

In a future full-stack system, the web/mobile event contract and backend state builder must agree on timestamps, deduplication, and prediction point. This chapter deliberately does not implement real-time serving or UI integration; those belong in Part V.

## Privacy and governance

The feature schema avoids exact account balance, account number, SSN, raw identity documents, name, email, access token, and full transfer amount. Behavioral workflow signals are sufficient for this exercise. Pseudonymous session IDs support grouping but are excluded from model features.

Behavior data still requires purpose limitation, access control, retention limits, security, auditability, and review for linkage and disparate impact. Minimization is a design property, not a claim that event data is harmless.

## Brief coefficient caution

A positive fitted coefficient for an input would mean that, after this pipeline's preprocessing and conditional on its other inputs, the model associates a larger value with synthetic target `1`. It does not show that the input causes abandonment. Coefficients can reflect generator design, correlations, sampling noise, and instrumentation.

## Exercises

### 1 — Feature or future information?

Classify `channel`, `elapsed_seconds_at_recipient_selected`, `help_opened_before_prediction`, `transfer_completed`, `error_count_before_prediction`, `final_event_name`, `journey_duration_seconds`, and `search_count_before_prediction`. State the prediction-time test you applied.

### 2 — What do we actually know?

Given `help_opened = true` and `elapsed_seconds = 130`, list only legitimate conclusions. Why are confusion, frustration, intent to stop, and need for assistance unsupported?

### 3 — Leakage

Explain why `session_ended` cannot be a feature when predicting whether a journey will complete before the session ends. Propose a value known at `recipient_selected` instead.

### 4 — False positive

A high-risk prediction completes successfully. Describe how an intrusive modal could worsen the product, and propose a less intrusive use of the signal.

### 5 — Intervention

Distinguish a model's probability from the decision about subsequent UI behavior. Who should own each contract, and what should be logged for evaluation?

### Coding exercise — count help events

Add `help_events_before_prediction`, the count of help-related events through `recipient_selected`:

1. update `JourneySnapshot` and `PartialJourneyState`;
2. count only the feature window;
3. add it to the numeric schema and preprocessing;
4. retrain and evaluate with the identical split policy;
5. compare results with the previous model;
6. explain whether it appears useful on this synthetic test set; and
7. make no causal or psychological claim.

Add a leakage test with help events after prediction to prove they do not change the new feature.

## Key takeaways

1. Journey abandonment is behavioral, not psychological.
2. The prediction point must be explicit.
3. Features use only information available at prediction.
4. Historical labels may use later outcomes during training.
5. Temporal leakage can make an invalid model look excellent.
6. Partial snapshots convert event streams into model inputs.
7. Probabilities do not guarantee an individual outcome.
8. False positives can cause unnecessary, annoying interventions.
9. Prediction and intervention are different system-design problems.
10. Behavioral ML should support experience, never financial eligibility or authorization.

## What comes next

Chapter 10 — **Conversion Prediction** will broaden the lens:

> Given a digital session or campaign visit, how likely is the member to complete a defined conversion event?

```text
open product-information page → start application → complete application

marketing landing page → product details → application started
```

```text
ABANDONMENT MODEL
focuses on an already-started journey

CONVERSION MODEL
estimates whether a desired digital action will occur
```

[Back to the Part III overview](README.md) · [Back to complete contents](../../CONTENTS.md) · [Next: Chapter 10 — Conversion Prediction](chapter-10-conversion-prediction.md)
