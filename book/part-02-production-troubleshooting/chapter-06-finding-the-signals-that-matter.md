# Chapter 6 — Finding the Signals That Matter

> Part II — Machine Learning for Production Troubleshooting

[Part II overview](README.md) · [Complete contents](../../CONTENTS.md) · [Previous: Chapter 5](chapter-05-incident-classification.md) · [Next: Chapter 7](chapter-07-predicting-integration-failures.md)

## Central question

> Which telemetry signals actually help Harbor distinguish one kind of incident from another?

The Chapter 5 classifier works reasonably well for its narrow, fictional fixture. During review, however, one developer asks, “Why is the model making these decisions?” Another asks, “Do we really need all six telemetry fields?” A third asks, “Could one noisy metric be making the model worse?” The **Harbor Federal Credit Union** team decides to investigate the signals themselves.

```text
TELEMETRY
   │
   ├── api_latency_ms
   ├── error_rate
   ├── db_connections
   ├── queue_depth
   ├── vendor_latency_ms
   └── requests_per_minute
            │
            ▼
      INCIDENT CLASSIFIER
            │
            ▼
        prediction
```

The new question is:

```text
Which inputs are contributing useful information?
```

This is an engineering investigation, not a search for one magical “feature importance” number. A feature is a deliberately selected, prediction-time representation supplied to a model. It might be useful, overlap another feature, contain noise, encode a shortcut, become misleading after a system change, or exist only after the prediction should have happened.

Everything here is deterministic, synthetic educational material. It contains no real member, vendor, transaction, or financial data and does not model an actual banking system.

## Learning objectives

By the end of this chapter, you should be able to:

1. explain what a feature represents;
2. distinguish useful, redundant, noisy, and misleading features;
3. inspect feature distributions by incident class;
4. calculate simple correlations between numerical signals;
5. explain why correlation is not causation;
6. inspect model coefficients carefully;
7. explain why coefficient magnitude depends on feature scale;
8. use standardized coefficients for comparison;
9. perform simple feature ablation;
10. compare model performance with and without selected features;
11. recognize that “important to this model” does not mean “causes the incident”; and
12. explain why feature usefulness can change over time.

## Begin with the data

> Before asking what the model thinks matters, inspect the underlying data.

Class summaries can expose broad patterns before a model is fitted. Chapter 6 calculates them directly from `data/harbor_incident_classes.csv`; it does not paste assumed averages into the program:

```python
observations = load_incident_dataset(DATASET)
summaries = summarize_features_by_class(observations)

for feature in INCIDENT_FEATURES:
    print(f"Feature: {feature}")
    for item in summaries:
        if item.feature == feature:
            print(item.incident_type, item.mean, item.minimum, item.maximum)
```

`summarize_features_by_class` groups observations by their historical `incident_type`. For every requested feature and group it computes the mean, minimum, and maximum from the fixture. This can suggest, for example, whether one class occupies a visibly different range. It is descriptive evidence, not a causal result.

The executable output has this shape:

```text
Feature: vendor_latency_ms
  normal                   mean=... min=... max=...
  vendor_degradation       mean=... min=... max=...
  database_pressure        mean=... min=... max=...
  traffic_spike            mean=... min=... max=...
  application_regression   mean=... min=... max=...
```

> If one feature has visibly different distributions across classes, it may contain useful predictive information.

Yet overlapping distributions do not mean a feature is useless. A signal may become useful only in combination with other signals, and min/max summaries hide the shape and density inside a range.

## Features can work together

An elevated API latency is compatible with several incident families. Context changes its meaning:

```text
api_latency alone
       ↓
ambiguous

vendor_latency alone
       ↓
somewhat informative

api_latency + vendor_latency + DB pressure
       ↓
much clearer incident pattern
```

Conceptually, the combinations could look like this:

```text
vendor_degradation:              application_regression:

api latency       high           api latency       high
vendor latency    high           vendor latency    normal
DB pressure       moderate       DB pressure       moderate
```

The important information can be the relationship among inputs. This is why checking one distribution at a time is a useful beginning rather than a complete explanation.

## Correlation

Pearson correlation summarizes the direction and strength of a *linear* association between two numerical variables:

```text
correlation ≈ +1       values tend to rise together
correlation ≈  0       no strong linear relationship
correlation ≈ -1       one tends to fall as the other rises
```

Chapter 6 uses NumPy, already installed through scikit-learn, rather than adding a dataframe or plotting dependency:

```python
correlations = calculate_correlations(X)
```

The result is a square matrix in exactly the same order as `ANALYSIS_FEATURES`. The laboratory prints every value with row and column labels. The matrix is computed with `numpy.corrcoef`; its diagonal is approximately `1.0`, and it is symmetric. A value near zero rules out neither a nonlinear relationship nor usefulness in combination with other features.

### Correlation is not causation

```text
queue depth ↑
api latency ↑
```

does not prove:

```text
queue depth caused API latency
```

Both could follow from a vendor slowdown, a traffic spike, or an application regression. One might genuinely contribute to the other, or the relationship might vary by incident type. Establishing an intervention and ruling out alternatives requires evidence this observational fixture does not provide.

```text
CORRELATION
These values move together.

CAUSATION
Changing one produces a change in the other.

These are not the same claim.
```

### Redundant features

`queue_depth` and a hypothetical `pending_request_count` might be strongly related. So might `api_latency_ms` and `request_duration_ms` if their definitions substantially overlap. Keeping both can create unnecessary complexity, collection cost, harder interpretation, and coefficient instability in some fitted models.

The rule is **not** “always remove correlated features.” They may cover different scopes, fail independently, or contribute complementary detail. Instead:

> Strong correlation is a reason to investigate whether both measurements are needed.

## A deliberately noisy feature

The laboratory appends `synthetic_noise` in memory. It is not written to the core fixture and is not a banking metric:

```python
rng = random.Random(42)
noise = [rng.random() for _ in range(number_of_rows)]
```

The generator never sees `incident_type`, so the noise has no intentionally designed target relationship. A fixed seed makes tests and demonstrations reproducible.

```text
USEFUL SIGNAL
contains information related to target patterns

NOISE
varies but has no intentionally designed relationship
to the target
```

That design does not guarantee a fitted model assigns the noise exactly zero weight. In 300 rows, accidental patterns can occur. A model can exploit those coincidences, especially when data is limited.

## Misleading features and leakage

Chapter 2 separated prediction-time inputs from facts learned later. An `incident_closed_reason` or `postmortem_category` might almost reveal the answer—but neither exists when an initial classifier must respond:

```text
prediction time
      │
      ▼
AVAILABLE FEATURES
      │
      ▼
MODEL
      │
      ▼
prediction
      │
      ▼
later incident investigation
      │
      ▼
postmortem category
```

The postmortem category cannot flow backward into prediction-time features. Such leakage can produce impressive offline scores and a useless production system. Availability includes timing, not merely whether a field exists somewhere in Harbor's data estate.

## Model coefficients

Chapter 5's pipeline standardizes each input and then fits multi-class logistic regression:

```text
StandardScaler
     │
     ▼
LogisticRegression
```

A coefficient describes how the fitted model uses a feature when calculating class scores. Each class has one coefficient per input. Chapter 6 deliberately obtains the classifier from the fitted pipeline and pairs `classifier.coef_` rows with `classifier.classes_`, rather than assuming taxonomy or alphabetical order. It pairs columns with the exact supplied feature order:

```python
coefficients = extract_model_coefficients(model, ANALYSIS_FEATURES)
for item in coefficients:
    print(item.incident_type, item.feature, item.coefficient)
```

The laboratory prints real fitted values:

```text
Class: vendor_degradation
  api_latency_ms          ...
  error_rate              ...
  db_connections          ...
  queue_depth             ...
  vendor_latency_ms       ...
  requests_per_minute     ...
  synthetic_noise         ...
```

The raw fields use incomparable units: milliseconds, proportions, counts, and requests per minute. Without scaling, a one-unit change means something radically different for each feature, so raw coefficient magnitudes cannot fairly be compared. `StandardScaler` expresses inputs in standard-deviation units learned from training data. That makes magnitude comparisons within this fitted model more meaningful—but not definitive.

Read the table cautiously:

- positive and negative directions concern this model's score for one class;
- magnitude suggests stronger or weaker use within this fitted linear model;
- correlated inputs can divide, trade, or destabilize coefficients;
- a coefficient is conditional on the other included features and training sample; and
- neither sign nor magnitude is a causal claim.

## Feature ablation

**Ablation** means removing one input or group of inputs, retraining the model, and observing what changes:

```text
BASELINE MODEL                 ABLATION MODEL
all features                   remove vendor_latency_ms
     │                              │
     ▼                              ▼
performance                     performance
          \                     /
           └──── difference ───┘
                     │
                     ▼
          evidence about usefulness
```

This feels familiar to full-stack developers:

```text
DEBUGGING                       FEATURE ANALYSIS
possible causes                 candidate features
   │                               │
   ▼                               ▼
remove / isolate variables      remove / isolate features
   │                               │
   ▼                               ▼
observe behavior                observe model behavior
```

`run_ablation` compares all seven inputs with variants that remove `vendor_latency_ms`, `db_connections`, `requests_per_minute`, or `synthetic_noise`. Every candidate uses the same stratified row split and is retrained from scratch. Selection occurs *after* splitting by selecting identical columns from the already split matrices:

```python
results = run_ablation(observations)
for result in results:
    print(result.name, result.features, result.accuracy)
```

Output is computed at run time:

```text
Ablation study
all_features                   accuracy: ...
without_vendor_latency_ms      accuracy: ...
without_db_connections         accuracy: ...
without_requests_per_minute    accuracy: ...
without_synthetic_noise        accuracy: ...
```

A drop is evidence that the omitted signal helped this model on this split. No change can mean redundancy, weak signal, or an evaluation too coarse to expose a difference. An improvement can happen when a distracting feature encourages overfitting—or simply by chance. None of those observations alone proves what will happen in production.

### Small datasets are unstable

The fixture has 300 rows and a 75-row test set. Therefore:

```text
75 test examples

1 additional mistake
=
1.33 percentage-point accuracy change
```

An apparent 1.33-point movement may be a single observation. One deterministic split is useful for a reproducible lesson but weak evidence for a production decision. Repeated evaluation or cross-validation would provide stronger evidence; Chapter 17 develops evaluation more deeply. Do not repeatedly choose features against this test set until it effectively becomes training data.

## There is no single universal meaning of “feature importance”

Different techniques answer different questions, and their answers are not interchangeable:

```text
QUESTION
"What does this feature look like across classes?"
→ distribution summary

QUESTION
"Does it move with another numerical feature?"
→ correlation

QUESTION
"How does this logistic model use it?"
→ coefficient

QUESTION
"What happens if we remove it?"
→ ablation
```

Class averages describe groups. Correlation describes pairwise linear association without using the target directly. Coefficients expose one fitted linear model's scoring rule. Ablation measures held-out behavior after removing and retraining. Agreement strengthens a hypothesis; disagreement tells engineers to investigate assumptions, interactions, split variance, and data quality.

## The reusable implementation

`src/harbor_ml/feature_analysis.py` defines frozen results:

```python
@dataclass(frozen=True)
class FeatureSummary:
    feature: str
    incident_type: str
    mean: float
    minimum: float
    maximum: float

@dataclass(frozen=True)
class AblationResult:
    name: str
    features: tuple[str, ...]
    accuracy: float
    test_observations: int
```

Frozen dataclasses give named, immutable results instead of positional tuples. `features` records what an experiment actually supplied, making accidental non-removal observable. Validation rejects unknown, empty, duplicate, mismatched, non-finite, and constant inputs where appropriate. `compare_feature_sets` is reusable for exercises; `run_ablation` supplies this chapter's standard candidates.

Run the complete laboratory from the repository root:

```bash
python examples/chapter_06_feature_analysis.py
```

Its interpretation is generated from the actual run: it identifies the largest observed ablation loss, compares the no-noise result with baseline, reports the one-error resolution, and explicitly denies a causal conclusion. It does not force the fixture to tell a predetermined story.

## Operational considerations

Collection is not free. Every production feature may add:

- instrumentation and processing complexity;
- storage and retention obligations;
- privacy and security review;
- dependency on a vendor field;
- schema and unit maintenance;
- changing semantics across deployments; and
- monitoring for absence, delay, and corruption.

> A feature should earn its place in the production system.

A tiny offline improvement may not justify an unreliable vendor dependency. Conversely, a modest but cheap, stable, prediction-time metric may be operationally valuable. Engineering judgment combines statistical evidence with availability, latency, resilience, ownership, and risk.

## Feature usefulness can drift

A feature useful today can become less useful when vendor behavior, architecture, traffic, queue configuration, application performance, or the incident taxonomy changes:

```text
TRAINING PERIOD
vendor latency strongly identifies one class

        ↓ system changes

FUTURE PERIOD
vendor latency behaves differently
```

This is **feature drift** at an introductory level: the feature's distribution or relationship to the target changes. Production monitoring must revisit assumptions rather than permanently trusting this analysis. Chapter 25 returns to model monitoring.

## Exercises

### Exercise 1 — Useful, redundant, noisy, or leaked?

Classify these candidates, and state what additional context you need:

```text
vendor_latency_ms
requests_per_minute
random_number
postmortem_category
request_duration_ms
api_latency_ms
```

`random_number` is intentionally noisy and `postmortem_category` is leaked if created after prediction. The telemetry fields may be useful. `request_duration_ms` may overlap API latency—or differ in scope enough to add information. The correct answer depends on definitions, timing, quality, and the model's task.

### Exercise 2 — Correlation

Suppose `queue_depth` and `api_latency_ms` have correlation `0.84`. What can legitimately be concluded?

You may conclude they have a strong positive linear association in the measured sample. You may not conclude that queue depth causes latency, that the relationship applies to future traffic, or that either is independently predictive of incident class.

### Exercise 3 — Coefficients

Consider this fictional standardized table:

```text
Class: vendor_degradation
vendor_latency_ms      +2.10
api_latency_ms         +0.75
queue_depth            +0.20
```

Which feature does this fitted model use most strongly in the positive direction for that class? `vendor_latency_ms`. Does that prove vendor latency causes the incident? **No.** It describes model scoring under a particular fit.

### Exercise 4 — Ablation

Suppose an experiment reports:

```text
all features              84%
without vendor latency    69%
without synthetic noise   85%
```

The results suggest hypotheses that vendor latency contributes useful predictive information and that noise is unnecessary or mildly harmful on this split. They are evidence, not final conclusions: examine error counts, repeat evaluation, check leakage and drift, and consider collection cost.

### Coding exercise — another synthetic signal

Add deterministic `synthetic_noise_2` with a different fixed seed. Then:

1. include it in the matrix and feature names;
2. retrain on the same deterministic split;
3. inspect its per-class coefficients;
4. ablate it;
5. compare held-out performance; and
6. write a short, non-causal interpretation.

Why can random noise occasionally appear useful? With a small sample, chance alignment with labels or other inputs can survive one split. Explain why changing seeds and repeated evaluation would test that apparent usefulness more honestly.

## Key takeaways

1. More features are not automatically better.
2. A feature must represent information available at prediction time.
3. Distribution summaries help reveal class patterns.
4. Correlation describes association, not causation.
5. Coefficients describe how a specific fitted model uses features.
6. Standardization makes coefficient magnitudes more comparable, not causal.
7. Correlated features can complicate interpretation.
8. Ablation measures what happens when inputs are removed and the model is retrained.
9. Noise can appear useful by chance on small datasets.
10. Feature usefulness can change as systems evolve.
11. Every collected feature carries operational cost.
12. Feature analysis should support engineering judgment rather than replace it.

## What comes next

# Chapter 7 — Predicting Integration Failures

So far, Harbor has used telemetry to:

```text
detect unusual behavior
        ↓
classify known incidents
        ↓
understand useful signals
```

Now return to a concrete full-stack responsibility: vendor and fintech API integrations. The next question is:

> Can Harbor predict that a specific vendor-backed request is likely to fail before the failure happens?

Chapter 7 will move from system-wide telemetry to request-level prediction, considering prediction-time features such as:

```text
vendor
endpoint
recent_vendor_latency
recent_error_rate
queue_depth
retry_count
request_size
time_of_day
```

and the target:

```text
request_failed
```

[Continue to Chapter 7 — Predicting Integration Failures](chapter-07-predicting-integration-failures.md)
