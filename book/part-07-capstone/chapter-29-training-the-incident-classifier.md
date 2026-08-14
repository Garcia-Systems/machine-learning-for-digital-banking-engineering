# Chapter 29 — Training the Incident Classifier

> **Part VII — Capstone: The Intelligent Digital Credit Union**

Chapter 28 successfully trained Harbor Federal Credit Union's anomaly detector. During the fictional capstone incident it begins reporting `anomaly = true`. That means only:

> Something about the current multivariate telemetry differs from the learned healthy baseline.

Engineers now ask whether the observation looks more like vendor degradation, database pressure, a traffic spike, or an application regression. Answering that question requires labeled history:

```text
historical telemetry + historical incident_type
                    │
                    ▼
           supervised classifier
                    │
                    ▼
       probabilities across known classes
```

The central question is:

> Once Harbor detects unusual system behavior, can it train a reproducible classifier that estimates which known incident pattern the current telemetry most resembles?

The answer is yes—with an important boundary:

```text
ANOMALY DETECTOR                         INCIDENT CLASSIFIER
"Current behavior differs from           "Current behavior most resembles
 the healthy baseline."                   this known historical pattern."
```

Neither establishes root cause.

## Learning objectives

By the end, you will be able to distinguish anomaly detection from classification; define and validate a target taxonomy and feature contract; justify a deterministic split; fit a multi-class classifier; map probabilities through `classes_`; calculate accuracy, a confusion matrix, per-class metrics, and macro F1; preserve ambiguity; describe unknown-class limits; persist a full pipeline with fingerprinted metadata; score the Chapter 26 timeline; and interpret every class as an engineering hypothesis signal.

## Taxonomy semantics come before fitting

The canonical target is `incident_type`:

```text
normal
vendor_degradation
database_pressure
traffic_spike
application_regression
```

Here it means **the synthetic historical incident category assigned to the scenario based on the primary initiating operational condition**. It does not necessarily mean the dominant current symptom, ticket category, or operator's final diagnosis. The loader rejects any unexpected value, such as `network_routing_failure`, rather than quietly creating a sixth class.

This distinction matters. Vendor degradation can trigger retries, queue growth, and database pressure. Late telemetry might resemble `database_pressure` even though a vendor condition initiated the scenario. A single label forces one answer onto compound reality.

Sometimes the problem is not the model; the categories themselves overlap. Production alternatives could include multi-label classification, an incident hierarchy, or separate cause and symptom taxonomies. We do not implement those here.

## Labeled history and class representation

Chapter 27 established Harbor's trustworthy prediction-time telemetry semantics. Its compact 17-row incident reconstruction is suitable for timeline evaluation but contains only three represented labels and cannot support a five-class train/test evaluation. Chapter 29 therefore reuses its six raw prediction-time fields while fitting on the already committed Chapter 5 labeled historical fixture. This is an explicit limitation, not a silent data substitution. A production capstone dataset would reconstruct the same rolling contract across many historical scenarios.

The committed synthetic history has 300 observations: 60 per class. The deterministic split leaves these actual training counts:

```text
normal                   45
vendor_degradation       45
database_pressure        45
traffic_spike            45
application_regression   45
```

No class is smaller, so this laboratory neither oversamples nor reweights. Balance is unusually clean because the data is synthetic. Real incident history is commonly dominated by normal observations; a small class then gives the model fewer patterns to learn and makes accuracy especially misleading.

## Feature contract and leakage discipline

The ordered contract is:

```text
api_latency_ms
error_rate
db_connections
queue_depth
vendor_latency_ms
requests_per_minute
```

These fields exist in Chapter 27's trustworthy raw telemetry and in the historical labeled fixture. Rolling fields such as `vendor_latency_mean_5m` are not fabricated for older observations that cannot reconstruct them consistently.

`incident_type` is only `y`; it never enters `X`. Also prohibited are `request_failed`, confirmed diagnosis/cause, final trace duration, future/final status, phase, trace diagnosis, and sensitive identity fields. Feature selection is an allowlist, so narrative fields cannot sneak in merely because a CSV gains a column.

```text
SAME TELEMETRY
     │
     ├── anomaly detector
     │      question: unusual or normal?
     │
     └── incident classifier
            question: which known category?
```

Sharing raw telemetry is acceptable: the fitted populations, targets, algorithms, and questions differ.

## Model and reproducibility

The pipeline is deliberately familiar:

```text
features → StandardScaler → LogisticRegression → multi-class probabilities
```

Logistic regression is understandable, produces probabilities, supports coefficient inspection, is sufficient for teaching data, and keeps attention on engineering workflow rather than algorithm novelty. Scikit-learn 1.9 handles multiple classes automatically; Chapter 29 does not pass the removed/deprecated `multi_class` option. The `lbfgs` solver, 1,000-iteration limit, and random state are explicit.

The scaler and classifier are fitted and saved as **one pipeline**. Saving only the classifier would lose the learned centering and scaling that define its inputs.

## Split choice

Earlier history should normally train and later history should test. Here, every class occurs throughout two short synthetic days, and a strict cutoff is fragile: small fixture changes could remove a class from one side. Chapter 29 therefore uses a deterministic, stratified 75/25 random split and records `deterministic_stratified_random_split` in metadata. Tests require all five classes on both sides.

This is not chosen to improve the score. It is a disclosed limitation: random splitting can make nearby synthetic observations more similar than genuinely future incidents would be. A larger production history should use incident-grouped chronological backtesting.

## Held-out evaluation

Run:

```bash
python examples/chapter_29_training_incident_classifier.py
```

With the committed dependency version and seed, the held-out results are:

```text
training rows:              225
test rows:                   75
majority baseline accuracy: 0.200
model accuracy:             0.907
macro F1:                   0.910
```

The majority baseline always predicts the most common training class. Because all classes tie, deterministic input order selects one, which correctly classifies 15 of 75 balanced test rows. Baseline accuracy alone remains weak: it says nothing about minority-class usefulness.

Macro F1 means: **compute F1 for each class and average them equally**. This prevents a majority class from dominating the summary.

```text
Class                    precision recall    F1 support
normal                       1.000  1.000 1.000      15
vendor_degradation           1.000  0.933 0.966      15
database_pressure            0.722  0.867 0.788      15
traffic_spike                1.000  0.867 0.929      15
application_regression       0.867  0.867 0.867      15
```

Precision asks how often predictions of a class are correct. Recall asks how much of that actual class is found. F1 balances the two.

### Confusion matrix

Rows are actual; columns are predicted, in canonical taxonomy order:

```text
                         normal vendor database traffic application
normal                       15      0        0       0           0
vendor_degradation            0     14        1       0           0
database_pressure             0      0       13       0           2
traffic_spike                 0      0        2      13           0
application_regression        0      0        2       0          13
```

The actual result shows database pressure as the principal competing prediction: it receives one vendor, two traffic, and two application examples. Database pressure also loses two examples to application regression. That visible overlap—not an invented narrative—explains its lower precision.

Error analysis prints up to three held-out mistakes with their features and probabilities. One application regression is assigned database pressure at 0.647. A confident error is valuable for dataset, taxonomy, feature, and model review. It is not evidence that database pressure caused that synthetic incident. Ask instead: **Which visible telemetry caused the current observation to resemble the competing class?**

## Correct probability mapping and ambiguity

`predict_proba` columns are not assumed to follow display or taxonomy order. The implementation zips scores to the fitted pipeline's actual `model.classes_`, then ranks the resulting mapping.

```text
vendor_degradation      0.45
database_pressure       0.41
traffic_spike           0.06
application_regression  0.05
normal                  0.03
```

The argmax is vendor degradation, but its top-two gap is only `0.45 - 0.41 = 0.04`. With this laboratory's configured teaching rule, `gap < 0.10` means ambiguous:

```text
Top class: vendor_degradation
Interpretation: ambiguous
```

The threshold is policy, not a universal truth. Changing it never changes model probabilities or argmax; it changes how Harbor communicates weak separation. Metadata records it.

## The Chapter 26 timeline

The executable lab constructs the same six-feature row at every timestamp, calculates real fitted probabilities, selects the top class, computes the gap, and marks ambiguity. Editorial phase and initiating class appear only after scoring as evaluation columns; neither is fed to the model.

Selected actual output:

```text
time  top class              probability gap   ambiguous editorial
10:00 normal                     0.943   0.917 no        normal/healthy
10:10 normal                     0.838   0.730 no        vendor_degradation/early_signal
10:12 vendor_degradation         0.572   0.191 no        vendor_degradation/degradation
10:18 vendor_degradation         1.000   1.000 no        vendor_degradation/degradation
10:30 vendor_degradation         1.000   1.000 no        vendor_degradation/confirmed_incident
10:34 database_pressure          0.531   0.141 no        normal/recovery
10:38 normal                     0.652   0.509 no        normal/recovery
```

Actual results do **not** become ambiguous under the 0.10 rule during peak compound pressure; they become extremely vendor-like. During recovery, however, the classifier temporarily favors database pressure. That is useful rather than automatically “wrong”: current residual queue/database telemetry resembles historical pressure while the editorial scenario label describes the initiating condition (or recovery). This is precisely why classification is resemblance, not diagnosis.

## Unknown conditions

```text
UNKNOWN CONDITION
       │
       ▼
classifier
       │
       ▼
closest known pattern
```

If reality is `network_routing_failure`, the closed-set classifier must still return one of its five known labels. The anomaly detector can say something is unusual, but neither model invents a new diagnosis.

```text
ANOMALY DETECTOR
unusual = yes
        │
        ▼
INCIDENT CLASSIFIER
top class = vendor_degradation
probability = model output
ambiguous = policy output
        │
        ▼
ENGINEER
investigate
```

This combination prioritizes investigation. Traces, logs, deployments, dependency evidence, and engineering judgment establish what happened.

## Artifact and metadata workflow

Run the modest training command:

```bash
python scripts/train_capstone_incident_classifier.py
python scripts/train_capstone_incident_classifier.py --ambiguity-gap 0.15
```

It validates, fingerprints, splits, trains, evaluates, scores the timeline, and writes:

```text
artifacts/capstone-incident-classifier/
├── model.joblib
└── metadata.json
```

`artifacts/` remains gitignored. Joblib loading is only for locally trusted artifacts; it is not safe for untrusted files. The model version is `harbor-capstone-incident-<dataset-hash-prefix>`. Metadata records model/type/version/time, dataset name/version/SHA-256, ordered features, target, taxonomy, train/test counts, all class counts, split strategy, seed, ambiguity threshold, held-out accuracy/macro and weighted F1/per-class metrics, Python, and scikit-learn.

The laboratory saves, reloads, and compares several full probability vectors with `numpy.allclose`. Repeated seeded runs produce the same split, metrics, timeline probabilities, and artifact behavior; only `trained_at` naturally changes.

## Exercises

1. **Anomaly versus classifier.** Which model asks “Does this look unlike healthy operation?” Which asks “Which known incident pattern does this resemble?”
2. **Argmax versus certainty.** For vendor `0.42`, database `0.40`, traffic `0.08`, application `0.06`, normal `0.04`, what is argmax? Is a `0.02` gap decisive?
3. **Unknown class.** What happens when `network_routing_failure` was absent from training?
4. **Macro F1.** Why is equal weighting of every class useful?
5. **Taxonomy.** Why can a vendor-originated event later resemble database pressure, and what does that imply for one label?
6. **Root cause.** Why does `predicted class = vendor_degradation` establish resemblance rather than cause?

### Coding exercise — ambiguity policy

Use `--ambiguity-gap` at two values. Confirm metadata records each threshold, count ambiguous timeline rows, and verify probabilities remain identical. Explain why interpretation changes but model output does not.

## Key takeaways

1. Incident classification is supervised learning over predefined categories.
2. Anomaly detection and classification answer different questions.
3. A closed-set classifier chooses only among classes it has seen.
4. Map probability columns with fitted `classes_`.
5. Confusion matrices expose overlapping patterns.
6. Macro F1 gives every class equal summary weight.
7. Argmax is not certainty; preserve ambiguity.
8. Evolving incidents can resemble secondary effects rather than the initiating category.
9. Dataset and taxonomy quality can limit a model as much as its algorithm.
10. Classification prioritizes investigation; it does not replace diagnosis.

## What comes next: Chapter 30 — Building the ML Service

Chapters 28 and 29 now produce a capstone anomaly detector and incident classifier. Chapter 30 will expose them behind one coherent, versioned Python service:

```text
HARBOR APPLICATION
       │
       ▼
CAPSTONE ML SERVICE
       ├── /score/telemetry-anomaly
       ├── /predict/incident
       └── /predict/integration-failure
       │
       ▼
versioned JSON contracts
```

It will cover trusted loading, multiple runtimes, health/readiness, request schemas, stable responses, ambiguity, partial unavailability, and no training in handlers. **Chapter 30 remains planned and is not implemented here.**

[Previous: Chapter 28 — Training the Anomaly Detector](chapter-28-training-the-anomaly-detector.md) · [Back to Part VII](README.md) · [Complete contents](../../CONTENTS.md) · Next: Chapter 30 — Building the ML Service *(planned)*
