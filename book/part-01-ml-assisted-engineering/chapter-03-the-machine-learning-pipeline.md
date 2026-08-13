# Chapter 3 — The Machine Learning Pipeline

[← Chapter 2](chapter-02-from-engineering-problem-to-ml-problem.md) · [Part I contents](README.md) · [Book contents](../../CONTENTS.md)

> **Central question:** Once we have defined an ML problem, how do historical observations become a model that can make predictions about new observations?

## Opening scenario: from a question to a prediction

The Harbor Federal Credit Union developer has progressed through:

```text
Chapter 0                              Chapter 1
Could ML help with this problem?  →   What can Harbor observe?
        ↓
Chapter 2                              Chapter 3
What exactly should it learn?     →   How does historical data become a prediction?
```

Chapter 2 defined the engineering question:

> Will a vendor-backed Harbor request fail?

Harbor has fictional historical observations with `vendor_latency_ms`, `queue_depth`, `db_connections`, `retry_count`, and `request_failed`. The first four values existed before the outcome was known, so they are features. `request_failed` is the historical target. We will now turn those examples into a small, real model. The goal is the complete pipeline—not maximum predictive performance.

## Learning objectives

By the end of this chapter, you should be able to:

1. Explain the basic supervised-learning pipeline and distinguish training from inference.
2. Separate features from targets and construct feature matrix `X` and target vector `y`.
3. Explain why training and test data must be separated.
4. Train a simple classifier and predict unseen examples.
5. Calculate and interpret basic evaluation results.
6. Explain why evaluation on training data is insufficient.
7. Recognize the limits of tiny educational datasets.
8. Locate a trained model inside a full-stack system.

## The supervised-learning pipeline

```text
HISTORICAL OBSERVATIONS
          │
          ▼
SELECT FEATURES + TARGET
          │
          ▼
       DATASET
          │
          ▼
   TRAIN / TEST SPLIT
      │         │
      ▼         ▼
    TRAIN      TEST
      │
      ▼
LEARNING ALGORITHM
      │
      ▼
     MODEL ──────────────┐
                         ▼
                     TEST DATA
                         │
                         ▼
                    PREDICTIONS
                         │
                         ▼
                    EVALUATION
```

Every arrow represents an engineering decision. In particular, two activities must not be confused:

```text
TRAINING                         INFERENCE

historical X + known y           new X
        │                          │
        ▼                          ▼
learning algorithm              trained model
        │                          │
        ▼                          ▼
trained model                   prediction
```

**Training** learns model parameters from historical examples whose outcomes are known. **Inference** applies those fixed, learned parameters to a new observation. Production applications normally perform inference repeatedly; training is a separate, controlled process.

## What changes from application development?

Traditional application development places rules directly in code:

```text
developer writes rules → application executes rules → output
```

Supervised ML changes where some behavior comes from:

```text
developer defines problem + data representation
        ↓
algorithm learns parameters from examples
        ↓
trained model → new input → prediction
```

The developer still designs the system. A model does not independently decide what problem matters, which data is appropriate, whether a target is legitimate, which metric matters, or what action should follow. Authentication, financial controls, and explicit policy remain deterministic.

## Turning observations into `X` and `y`

The loader introduced in Chapter 2 reads the existing `data/harbor_request_outcomes.csv` fixture as typed `RequestOutcome` objects. Chapter 3 deliberately selects only the four prediction-time features declared by `REQUEST_FAILURE`; it does not include the timestamp or outcome as a feature.

Conceptually:

```text
X =

vendor_latency_ms  queue_depth  db_connections  retry_count
220                12           31              0
235                14           32              0
620                26           38              1
1410               71           61              2
1760               109          79              3
1900               147          96              4


y =

0
0
0
1
1
1
```

```text
rows    = historical examples
columns = features
y       = known outcomes
```

The executable implementation makes the order explicit:

```python
outcomes = load_request_outcomes("data/harbor_request_outcomes.csv")
X = build_feature_matrix(outcomes)
y = build_target_vector(outcomes)
```

`X` has one row per request and one column per feature. `y` has one corresponding binary label per request. Feature order matters during both training and inference.

## Holding back a test set

```text
ALL HISTORICAL DATA
        │
        ├────────► TRAINING SET
        └────────► TEST SET
```

Think of the split like a software test that the implementation cannot inspect while being written:

```text
TRAINING DATA
Examples the model is allowed to learn from.

TEST DATA
Examples held back to see whether what was learned
generalizes beyond the training examples.
```

**Generalization** is the ability of a model to perform usefully on examples it did not see during training. Memorizing history is not enough.

The laboratory uses:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)
```

`test_size=0.25` holds out one quarter of this fixture. `random_state=42` fixes the pseudo-random selection so repeated runs and tests use the same rows; 42 has no statistical significance. `stratify=y` preserves representation of both target classes in the training and test partitions. On tiny data, each test row has substantial influence, so this is a reproducible demonstration rather than a strong performance study.

Training on every example and then scoring those same examples answers only, “How well does the model reproduce data it already saw?” It does not establish generalization—even if it reports 100%.

## A deliberately simple model

We use **logistic regression**, a conventional binary classifier, for pedagogical clarity. It estimates a relationship between feature values and the probability of a binary outcome:

```text
features → weighted combination → probability → classification

vendor latency ─┐
queue depth ────┤
db connections ─┼──► MODEL ───► P(failure) ───► success/failure
retry count ────┘
```

This is not a claim that logistic regression is Harbor's ideal production incident model, and we do not need its mathematical derivation yet.

Harbor's values have very different ranges: retry count is roughly 0–4, queue depth is tens or hundreds, and vendor latency is hundreds or thousands of milliseconds. `StandardScaler` expresses each training feature on a comparable scale. A scikit-learn `Pipeline` ensures the exact fitted transformation is also applied at inference:

```python
Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1_000)),
    ]
)
```

```text
raw features → scaling → classifier
```

The scaler is fitted only from training data. Fitting it before the split would let held-out test information influence training.

## scikit-learn's common interface

scikit-learn deliberately gives many algorithms a consistent interface:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Line one means: **learn model parameters from examples whose outcomes are already known**. Here, both the scaler and logistic classifier learn parameters from the training partition.

Line two means: **apply those learned parameters to new feature values**. It returns predicted classes and does not update the model. `model.predict_proba(X_test)` additionally returns estimated class probabilities.

The repository separates loading, feature and target construction, splitting, model construction, fitting, evaluation, validation, and single-request inference into small typed functions in `src/harbor_ml/request_failure_model.py`. There is no mutable global model and no hard-coded prediction.

## Run the laboratory

Install the narrow runtime dependency and run:

```bash
python -m pip install -r requirements-dev.txt
python examples/chapter_03_request_failure_model.py
```

The program reports the fixture's actual split, every held-out actual/predicted pair, accuracy and error counts, a confusion matrix summary, and one new prediction. Values are computed on every run from the loaded fixture and trained model.

```python
new_request = {
    "vendor_latency_ms": 1650,
    "queue_depth": 103,
    "db_connections": 74,
    "retry_count": 3,
}

prediction = predict_request_failure(model, new_request)
```

The function requires exactly the four named, finite, non-negative features and orders them identically to training. It reports a `predicted_class` and a `failure_probability` derived with `predict` and `predict_proba`.

```text
MODEL OUTPUT
failure probability = a value produced by the fitted model

        ≠

DIAGNOSIS
"The vendor is definitely broken."

        ≠

ACTION
"Disable the vendor integration."
```

The prediction is evidence. The engineer must investigate vendor telemetry, retries, application behavior, and competing explanations before diagnosing or acting.

## Basic evaluation

Accuracy is the fraction of held-out predictions equal to their known labels. We also report correct and incorrect counts, because “87.5%” is more understandable when the reader sees it means seven of only eight held-out examples.

A confusion matrix separates four outcomes:

```text
                         PREDICTED
                     success   failure
ACTUAL success         TN         FP
       failure         FN         TP
```

- **True negative (TN):** predicted success and the request succeeded.
- **False positive (FP):** predicted failure, but the request succeeded. This can create unnecessary investigation or alert fatigue.
- **False negative (FN):** predicted success, but the request failed. This can delay detection and affect members before an engineer responds.
- **True positive (TP):** predicted failure and the request failed.

False-positive and false-negative costs can differ. A later evaluation chapter will cover precision, recall, F1, thresholds, and other trade-offs; Chapter 3 intentionally stops at the foundation.

### Why accuracy is not enough

Suppose 99% of requests succeed. A classifier that always predicts success obtains:

```text
99% accuracy
0% of failures detected
```

A high accuracy number therefore does not automatically mean a useful model. Evaluation describes observed model behavior on a particular sample; it neither proves correctness nor guarantees production results.

## What the model learned

```text
WHAT WE GAVE THE MODEL

vendor_latency_ms
queue_depth
db_connections
retry_count

        ↓

WHAT WE ASKED IT TO LEARN

request_failed


WHAT IT DID NOT AUTOMATICALLY LEARN

root cause
business impact
member intent
correct remediation
```

The model learned statistical relationships between `X` and `y` within the supplied examples. It did not learn concepts such as “vendor outage” or “member frustration,” because the problem did not represent those concepts. Even a strong association does not establish causation.

## From offline training to online inference

```text
                  OFFLINE / TRAINING

Historical Harbor telemetry
            │
            ▼
      feature dataset
            │
            ▼
      training process
            │
            ▼
       trained model


                  ONLINE / INFERENCE

Harbor application
       │
       ▼
current telemetry
       │
       ▼
   ML service
       │
       ▼
 trained model
       │
       ▼
prediction → developer / monitoring system
```

Later chapters will build toward this architecture. We do **not** implement an ML service or persist a model here: serialization, artifact provenance, safe loading, version compatibility, and deployment lifecycle deserve focused treatment. Never load an untrusted serialized model.

## Limitations and responsible interpretation

The Harbor dataset is fictional, small, educational, and deliberately simplified. It is not representative of real credit-union production traffic, suitable for real financial decision-making, or evidence that these features predict failures in a real system. The few labels were constructed to teach mechanics, not statistical realism.

Production ML requires substantially more work, including data-quality analysis, representative sampling, class-imbalance handling, leakage prevention, temporal validation, monitoring, drift detection, security, privacy, and governance. A random split is useful here but can be misleading for time-dependent production traffic. These topics will be developed progressively.

## Exercises

### Exercise 1 — Trace the pipeline

Put these in the correct order, noting where features and target become `X` and `y`:

```text
historical observations
features
target
model
new observation
prediction
```

### Exercise 2 — Training or inference?

Classify each activity as data preparation, training, inference, or evaluation:

```text
model.fit(...)
model.predict(...)
constructing X_train
predicting a new request
evaluating held-out examples
```

### Exercise 3 — Find the mistake

A developer trains on all 100 examples, predicts those same 100 examples, and reports 100% accuracy. Why does this not establish generalization? Describe an appropriate correction.

### Exercise 4 — False positives and false negatives

For request-failure prediction, explain one operational consequence of each. Which might be more costly under an alert-only workflow, and what additional context would you need to decide?

### Coding exercise — Compare two predictions

Add a second fictional request to the executable example. Obtain its predicted class and probability, compare it with the supplied request, and explain which the model considers more failure-prone. Describe the result explicitly as a **model prediction**, not a root-cause diagnosis.

## Key takeaways

1. Historical examples become training data.
2. `X` contains features; `y` contains known targets.
3. Training learns from examples; inference applies the trained model to new observations.
4. Training and testing on the same examples does not demonstrate generalization.
5. Evaluation measures behavior; it does not prove correctness.
6. A pipeline keeps training-time transformations consistent with inference.
7. Predictions are evidence, not automatic engineering decisions.
8. The Harbor model is educational, not a production banking model.
9. ML becomes useful to a full-stack developer only inside a larger engineered system.

## Part I conclusion

```text
PART I

Chapter 0   Why might a full-stack developer use ML?
    │
    ▼
Chapter 1   What can the digital system observe?
    │
    ▼
Chapter 2   What should the model learn?
    │
    ▼
Chapter 3   How does data become a trained model?
```

Part I ends here: the reader has moved from deciding whether ML is appropriate to producing and honestly evaluating a prediction.

## Next: Part II — Machine Learning for Production Troubleshooting

**Chapter 4 — Detecting Abnormal Application Behavior** asks:

> What happens when Harbor does not already know what kind of failure to look for?

The supervised request-failure model had historical labels. Production incidents can instead involve behavior developers have never labeled. That leads naturally to **anomaly detection**. Continue to [Chapter 4](../part-02-production-troubleshooting/chapter-04-detecting-abnormal-application-behavior.md) for the first chapter of Part II and its executable laboratory.
