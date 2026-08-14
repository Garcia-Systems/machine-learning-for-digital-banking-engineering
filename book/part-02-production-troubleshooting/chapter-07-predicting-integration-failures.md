# Chapter 7 — Predicting Integration Failures

> Part II — Machine Learning for Production Troubleshooting

[Part II overview](README.md) · [Complete contents](../../CONTENTS.md) · [Previous: Chapter 6](chapter-06-finding-the-signals-that-matter.md)

## Central question

> Can Harbor estimate whether a specific vendor-backed request is likely to fail using only information available before the outcome is known?

Chapters 4–6 studied system telemetry. Now a member begins identity verification and **Harbor Federal Credit Union** is about to call the fictional `ClearVerify` service. The application already knows the vendor, endpoint, recent vendor latency and error rate, queue depth, retry count, request size, and hour. It cannot yet know the response status, failure reason, or actual outcome.

```text
SYSTEM-LEVEL ML                         REQUEST-LEVEL ML

Is Harbor behaving abnormally?          Is this specific integration request
What incident type does the             likely to fail?
system resemble?
```

The second question is **request-level binary classification**, a richer continuation of Chapters 2 and 3. Harbor's fictional integration landscape also includes `Northstar Payments`, `HarborLink Core Gateway`, and `BlueCurrent Documents`.

```text
PREDICTION TIME

known now
  │
  ├── vendor
  ├── endpoint
  ├── recent latency
  ├── recent error rate
  ├── queue depth
  ├── request size
  └── retry count
          │
          ▼
       MODEL
          │
          ▼
failure probability

AFTER REQUEST COMPLETES

final status
failure reason
actual outcome
```

Post-outcome fields cannot become prediction features merely because they exist later in a historical table.

## Learning objectives

By the end of this chapter, you should be able to:

1. distinguish request-level from system-level prediction;
2. define a prediction timestamp and identify request-time features;
3. identify outcome fields that cause leakage;
4. handle numerical and categorical features;
5. use a scikit-learn preprocessing pipeline and one-hot encoding;
6. train a binary request-failure classifier;
7. inspect predicted probabilities and apply classification thresholds;
8. distinguish probability from certainty;
9. explain false-positive and false-negative operational consequences;
10. recognize vendor-specific shortcut risks; and
11. treat predictions as engineering signals rather than automatic decisions.

## The prediction-time contract

> **The prediction-time contract is the set of information guaranteed to exist at the moment the model is asked to make a prediction.**

```text
REQUEST CREATED
      │
      ▼
prediction time
      │
      ├── vendor
      ├── endpoint
      ├── request size
      ├── current queue
      ├── recent vendor metrics
      └── previous retries
      │
      ▼
REQUEST SENT
      │
      ▼
vendor response
      │
      ▼
actual outcome
```

Feature design must respect that ordering. These are valid before sending:

```text
recent_vendor_latency_ms
recent_vendor_error_rate
request_size_bytes
retry_count
```

These are invalid at that time:

```text
final_http_status
failure_reason
response_duration_ms
request_failed
```

`recent_vendor_latency_ms` summarizes **completed previous requests**. By contrast, `current_request_response_latency_ms` cannot exist until the current response arrives. Likewise, a recent error rate is legitimate only if its rolling window ends before prediction; `current_request_failed` is the target.

```text
previous requests
      │
      ▼
rolling vendor metrics
      │
      ▼
CURRENT REQUEST CREATED
      │
      ▼
prediction
      │
      ▼
request sent
      │
      ▼
response
      │
      ▼
outcome recorded
```

A production feature pipeline must preserve this temporal ordering. A careless join that includes the current response in a rolling aggregate can create subtle leakage even when column names look legitimate.

## A fictional request dataset

`data/harbor_integration_requests.csv` contains 600 deterministic observations with these columns:

```text
timestamp, vendor, endpoint,
recent_vendor_latency_ms, recent_vendor_error_rate,
queue_depth, retry_count, request_size_bytes, hour_of_day,
request_failed
```

The fixture includes two endpoints for each of four fictional vendors, including `ClearVerify / identity_verify`, `Northstar Payments / transfer_submit`, `HarborLink Core Gateway / account_summary`, and `BlueCurrent Documents / statement_fetch`.

The generator in `scripts/generate_integration_requests.py` uses seed `707`. At a high level, its fictional log-odds combine recent latency, recent error rate, retry count, queue depth, request size, and modest endpoint interactions. It then adds noise and samples the outcome. Consequently, no vendor always succeeds or always fails, the classes overlap, and similar conditions can have different outcomes.

> This is an educational construct, not real financial, member, vendor, or production data. Its generated relationships do not represent real banking systems or estimate a real incident rate.

The committed CSV remains inspectable, while rerunning the script reproduces it exactly.

## Numerical and categorical inputs

The module declares the prediction-time matrix explicitly:

```python
NUMERIC_FEATURES = (
    "recent_vendor_latency_ms",
    "recent_vendor_error_rate",
    "queue_depth",
    "retry_count",
    "request_size_bytes",
    "hour_of_day",
)
CATEGORICAL_FEATURES = ("vendor", "endpoint")
TARGET = "request_failed"
```

Most ML algorithms require a numerical representation. Logistic regression cannot directly consume `vendor = "ClearVerify"` as a meaningful number.

### One-hot encoding

Conceptually, this column:

```text
vendor
ClearVerify
Northstar Payments
BlueCurrent Documents
```

becomes indicator columns:

```text
vendor_ClearVerify  vendor_Northstar_Payments  vendor_BlueCurrent_Documents
```

`ClearVerify` becomes `1 0 0`; `Northstar Payments` becomes `0 1 0`. **The values are indicators, not rankings.** Encoding vendors as 1, 2, and 3 would invent an ordering and numerical distance that the vendor names do not possess.

## `ColumnTransformer` and the pipeline

A `ColumnTransformer` sends different column groups through different preprocessing operations:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_indices),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_indices,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1_000)),
    ]
)
```

Each transformer tuple contains a name, transformer, and selected columns. `StandardScaler` expresses numerical values relative to their training-set centers and scales. `OneHotEncoder` learns categories from training data and produces indicator columns. `ColumnTransformer` combines those outputs. The `Pipeline` then passes that transformed matrix into logistic regression.

```text
RAW REQUEST

numerical fields ─────► StandardScaler ───┐
                                           │
categorical fields ───► OneHotEncoder ────┼──► transformed X
                                           │
                                           ▼
                                  LogisticRegression
                                           │
                                           ▼
                                  failure probability
```

Keeping preprocessing inside the pipeline matters: fitting and inference use the same learned scales, category mapping, and column order. The classifier never receives raw strings.

`handle_unknown="ignore"` prevents inference from crashing when a category absent from training appears. The encoder supplies zeros for that unknown category's known indicator columns. This availability behavior is useful, but it is not knowledge: the model has not learned how the new vendor behaves.

Logistic regression is deliberate here. The goal is not an algorithm tournament; it is to learn mixed-feature preprocessing, probability output, thresholds, and integration-engineering consequences.

## Train and evaluate

The executable laboratory loads the observations, extracts the target separately, performs a fixed class-stratified split, fits all preprocessing only from training rows, and evaluates 150 held-out rows:

```python
observations = load_integration_requests(DATASET)
X = build_integration_features(observations)
y = build_integration_targets(observations)
split = split_integration_dataset(X, y)
model = train_integration_model(
    build_integration_pipeline(), split.X_train, split.y_train
)
result = evaluate_integration_model(model, split.X_test, split.y_test)
```

With the committed fixture and dependencies, the actual deterministic result is:

```text
Accuracy: 0.813
Confusion matrix [[TN, FP], [FN, TP]]:
[[111   4]
 [ 24  11]]
False positives: 4
False negatives: 24
```

Accuracy is the fraction of correct held-out classifications at the default `0.50` threshold. It does not describe every error cost and is not an impressive production claim. The deliberately noisy fixture, one split, and fictional class frequency limit what this result means.

A **false positive** means:

```text
MODEL: likely failure
ACTUAL: request succeeds
```

For developer alerting, it could produce unnecessary monitoring, unnecessary fallback preparation, or noisy alerts.

A **false negative** means:

```text
MODEL: likely success
ACTUAL: request fails
```

It means a missed early warning: the developer sees the failure only after it happens. Their costs depend on use. A low-cost dashboard annotation can tolerate more false positives than a disruptive intervention—which is one reason this model must not control transactions.

## Probability is not certainty

`predict_proba` supplies the fitted model's estimated failure probability. If it reports `0.72`, it does not prove this request will fail. It describes model output under learned synthetic relationships, with all their limitations.

A threshold converts that continuous output into a class:

```text
failure probability = 0.72
threshold           = 0.50
prediction          = failure
```

```text
probability
0.00 ------------------------------ 1.00
                 ▲
              threshold

left  → success
right → failure
```

`0.50` is a common default, not a law of nature. At a lower threshold, more requests are flagged: Harbor may catch more actual failures but flag more successes. At a higher threshold, fewer are flagged: false alarms may fall while more failures go unflagged. That operational tradeoff precedes the deeper precision/recall treatment later in the book.

## Scenario laboratory

Run:

```bash
python examples/chapter_07_integration_failures.py
```

All probabilities below come from that fitted pipeline, rather than hand-written display values.

### Healthy request

```text
vendor: ClearVerify; endpoint: identity_verify
latency=180 ms; error_rate=0.015; queue=10; retries=0; size=1200 bytes
predicted failure probability: 0.023
class at threshold 0.50: success
```

### Degraded vendor

```text
vendor: Northstar Payments; endpoint: transfer_status
latency=720 ms; error_rate=0.150; queue=48; retries=0; size=1200 bytes
predicted failure probability: 0.649
class at threshold 0.50: failure
```

### Heavy request during pressure

```text
vendor: BlueCurrent Documents; endpoint: statement_fetch
latency=480 ms; error_rate=0.080; queue=88; retries=0; size=7800 bytes
predicted failure probability: 0.611
class at threshold 0.50: failure
```

### Retrying request

```text
vendor: HarborLink Core Gateway; endpoint: account_summary
latency=510 ms; error_rate=0.120; queue=45; retries=2; size=1200 bytes
predicted failure probability: 0.664
class at threshold 0.50: failure
```

These comparisons are model behavior on synthetic scenarios, not causal claims. High risk can arise from combinations; a vendor name alone is not a diagnosis.

### Compare thresholds without changing the model

For `heavy request during pressure`, the laboratory calculates one probability and applies three thresholds:

```text
Predicted probability: 0.611

threshold 0.30 → failure
threshold 0.50 → failure
threshold 0.70 → success
```

The underlying output did not change and the model was not retrained. Only the decision threshold changed.

### An unknown vendor

The laboratory also sends a valid request containing:

```text
vendor = Harbor Experimental Sandbox
```

This category does not occur in training. Inference completes and produces `0.051` because `handle_unknown="ignore"` avoids a transformation error.

> **Successful execution is not evidence that the model understands the new vendor.**

Its vendor value effectively contributes no known vendor indicator. Other numerical and endpoint inputs still contribute, but production engineering should detect unknown categories, monitor them, and establish a deliberate retraining or fallback policy.

## Vendor shortcut risk

Suppose one vendor happens to fail more often in the training sample. The model may learn a shortcut:

```text
vendor name
     ↓
failure risk
```

rather than general operational behavior. As Chapter 6 taught, a feature can appear useful without representing a durable cause. The shortcut becomes fragile when vendor reliability improves, contracts change, integrations are upgraded, or a new vendor arrives. Review category-level errors, compare models with and without vendor identity, monitor drift, and prefer request-time operational evidence. Do not “fix” this by hiding inconvenient evaluation results.

## Never automatically block a member request

This educational design explicitly rejects:

```text
ML predicts failure
        ↓
block member transaction
```

Use elevated risk to support engineers:

```text
ML predicts elevated failure risk
        │
        ├── enrich telemetry
        ├── increase monitoring
        ├── annotate developer dashboard
        └── inform retry/fallback investigation
```

The model must never bypass authentication, authorization, transaction validation, financial controls, idempotency, or vendor contract rules. If Harbor later considers retries, a deterministic policy must still enforce attempt limits, safe methods, idempotency, backoff, and contract rules. An uncertain ML signal is input to investigation, not permission to retry or route around controls.

## Architectural direction—not an implementation

```text
                         HARBOR APPLICATION

Member workflow
      │
      ▼
integration request
      │
      ├── vendor / endpoint
      ├── request metadata
      └── recent operational telemetry
                 │
                 ▼
         ML prediction service
                 │
                 ▼
       failure probability
                 │
                 ▼
       application telemetry
                 │
                 ▼
        developer dashboard
```

This chapter implements only an offline module and executable laboratory. It does **not** build an ML service; service boundaries, application integration, and deployment belong in Part V.

## Exercises

### Exercise 1 — Available at prediction time?

Classify each field and justify it from the timeline:

```text
vendor
endpoint
recent_vendor_error_rate
final_http_status
request_size_bytes
response_duration_ms
retry_count
failure_reason
```

Check: vendor, endpoint, properly bounded recent error rate, request size, and previous retry count are available. Final status, current response duration, and failure reason are outcomes.

### Exercise 2 — One-hot encoding

Why can this encoding create an artificial ordering?

```text
Vendor A = 1
Vendor B = 2
Vendor C = 3
```

Explain what relationships arithmetic would falsely imply, then sketch indicator columns instead.

### Exercise 3 — Threshold choice

Given `predicted failure probability = 0.63`, determine the class at thresholds `0.30`, `0.50`, and `0.70`. The answers are failure, failure, and success. Explain why the probability itself did not change: thresholding is a subsequent decision rule, not new inference.

### Exercise 4 — False positives and negatives

When output is used only for developer alerting, explain the operational meaning and likely cost of a false positive and false negative. Which would your team tolerate more, and what assumptions drive that answer?

### Exercise 5 — Unknown vendor

What does `handle_unknown="ignore"` guarantee? What does it not guarantee? Include inference availability, missing learned category evidence, monitoring, and model understanding in your answer.

### Coding exercise — `document_fetch_under_pressure`

Create an `IntegrationRequest` with:

```text
vendor = BlueCurrent Documents
endpoint = statement_fetch
```

Choose moderately elevated recent latency, elevated queue depth, one retry, and a larger request. Then:

1. calculate failure probability once;
2. classify it at `0.30`, `0.50`, and `0.70`;
3. print the conditions, probability, and three results;
4. write a short engineering interpretation; and
5. explicitly state that the prediction is not proof the request will fail.

Confirm that all threshold results reuse the same model probability and that your mapping contains no outcome field.

## Key takeaways

1. Request-level failure prediction is a binary supervised-learning problem.
2. Prediction-time ordering determines legitimate features.
3. Outcome fields must not leak into request-time inputs.
4. Numerical and categorical features require different preprocessing.
5. One-hot encoding creates indicators without imposing a ranking.
6. A scikit-learn pipeline keeps preprocessing and prediction consistent.
7. Model probabilities and classification thresholds are separate concepts.
8. Changing a threshold changes classification behavior without retraining.
9. Models may learn fragile vendor-specific shortcuts.
10. Failure predictions support observability; they do not bypass deterministic banking controls.

## Part II conclusion

```text
PART II — PRODUCTION TROUBLESHOOTING

Chapter 4
Does Harbor behavior look unusual?
        │
        ▼
Chapter 5
Which known incident does it resemble?
        │
        ▼
Chapter 6
Which telemetry signals are useful?
        │
        ▼
Chapter 7
Is this individual integration request likely to fail?
```

The reader now has several ML tools for **operational engineering**: anomaly detection, incident classification, feature investigation, and request-level failure prediction. Each provides evidence to an engineer, not automated authority.

Digital banking engineering also includes the member-facing product. Part III will shift the question:

# Part III — Machine Learning for Member Digital Experiences

Next: [**Chapter 8 — Understanding Member Behavior**](../part-03-member-digital-experiences/chapter-08-understanding-member-behavior.md).

```text
PRODUCTION TELEMETRY
How is the system behaving?

        ↓

DIGITAL EXPERIENCE DATA
How are members interacting with the system?
```

The next Part's fictional event vocabulary includes:

```text
session_started
login_completed
account_viewed
transfer_started
transfer_completed
verification_started
verification_abandoned
search_performed
```

[Continue to Chapter 8](../part-03-member-digital-experiences/chapter-08-understanding-member-behavior.md) · [Back to the Part II overview](README.md) · [Back to complete contents](../../CONTENTS.md)
