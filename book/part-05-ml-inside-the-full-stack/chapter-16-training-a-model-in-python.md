# Chapter 16 — Training a Model in Python

> **Part V — Putting Machine Learning Inside the Full Stack**

[Part V overview](README.md) · [Complete contents](../../CONTENTS.md) · [Previous: Chapter 15](../part-04-banking-operations/chapter-15-database-performance-prediction.md)

A Harbor Federal Credit Union developer has Chapter 7's integration request-failure model working. The code can load data, train, print metrics, and predict. Then another developer asks: *How do I reproduce exactly what you trained? Which dataset version and features did you use? Which settings? Does the artifact on disk match this code? How can we train a replacement without editing Python each time?*

Those are software-engineering questions. This chapter asks:

> **How do we turn an educational model experiment into a repeatable, explicit training workflow that can be rerun, tested, inspected, and eventually integrated into a larger application?**

```text
DATA → VALIDATION → FEATURE CONTRACT → TRAINING CONFIGURATION
     → TRAIN → EVALUATE → MODEL ARTIFACT → METADATA / REPORT
```

The question is no longer whether scikit-learn can call `fit()`.

## Learning objectives

By the end, you can distinguish an experiment from a workflow; define and validate a feature/dataset contract; make configuration and randomness explicit; keep training separate from inference; reuse Harbor's pipeline; evaluate before persistence; save and load a trusted full pipeline; record UTC metadata, dependency versions, and a dataset fingerprint; test an artifact round trip; and explain the versioning, security, and compatibility limits of serialized models.

## Experiment versus training workflow

```text
EXPERIMENT                         TRAINING WORKFLOW
developer runs code               validated input
        │                                ↓
        ▼                         explicit configuration
model appears in memory                  ↓
                                 reproducible training
                                        ↓
                                 evaluation
                                        ↓
                                 controlled artifact
                                        ↓
                                 recorded metadata
```

An experiment answers a learning question quickly. A workflow makes inputs, decisions, outputs, and failure behavior explicit so another developer or automation can repeat it. It is intentionally smaller than an MLOps platform.

## Training is not application runtime

```text
TRAINING                            APPLICATION RUNTIME
historical dataset                 current request
       ↓                                  ↓
training script                    load trusted artifact
       ↓                                  ↓
model artifact                     predict
```

Do not train whenever an application request arrives. That is slow and wasteful, makes operational behavior harder to reproduce, audit, and version, and confuses a historical dataset with one current request. Runtime loads a previously approved artifact and applies its learned state.

## The feature contract

Chapter 16 reuses Chapter 7's constants rather than copying its model definition:

```text
NUMERICAL                         CATEGORICAL
recent_vendor_latency_ms          vendor
recent_vendor_error_rate          endpoint
queue_depth
retry_count                       TARGET
request_size_bytes                request_failed
hour_of_day
```

These values exist at prediction time. `final_http_status`, `failure_reason`, and `response_duration_ms` are prohibited post-outcome leakage. A declared feature contract lets dataset validation, training, metadata, and future inference agree on names and order.

## A focused dataset contract

`validate_training_dataset()` runs **before** splitting or fitting. It checks:

- the CSV is nonempty and has at least 20 teaching observations;
- required columns exist and headers are not duplicated;
- all numerical fields parse, are finite, nonnegative, and respect rate/hour constraints;
- categorical fields are nonempty;
- `request_failed` is binary and both classes occur;
- declared features contain no prohibited outcome fields.

This is deliberately not a universal validation framework. It is a clear boundary for one model. A missing column raises an actionable `ValueError` rather than failing deep inside a transformer.

## Explicit training configuration

```python
@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.25
    random_state: int = 42
    classification_threshold: float = 0.50
    max_iter: int = 1_000
```

The frozen value object validates its ranges. It exposes only consequential teaching choices, not every logistic-regression option. Important behavior should not be scattered as unexplained literals.

`random_state` fixes the stratified split and the estimator's random behavior. The fixture generator is also seeded. At the documented level:

```text
same data + same code + same configuration + same dependency behavior
= reproducible teaching run
```

That means the same held-out rows and metrics in the same controlled environment. It does **not** promise byte-for-byte identical joblib files across platforms or library versions. Exact reproducibility across environments requires environment control.

## Fingerprint the exact input bytes

```text
data/harbor_integration_requests.csv → SHA-256 → 3a91…
```

`calculate_file_sha256()` streams file bytes through `hashlib.sha256`. The 64-character digest identifies the exact bytes referenced by a run. It does not prove that the data is correct, representative, safe, or free of leakage; it only connects a report to a specific content version. Changing even formatting changes the digest.

The simple model version is `harbor-integration-failure-<first eight dataset hash characters>`. This is inspectable, not a model registry. Two configurations on the same dataset can share this identifier, so metadata remains essential.

## Split, train, and evaluate with discipline

```text
training data → pipeline.fit()
test data     → evaluation only
```

The existing pipeline contains the `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, and `LogisticRegression`. It is fitted only on `X_train`; consequently scaler statistics and encoder categories come only from training data. Fitting preprocessing on the full dataset before splitting would leak test-distribution knowledge.

**Training** fits parameters for the chosen configuration. **Tuning** compares configurations or hyperparameters. This workflow does not turn the test set into a tuning scoreboard.

After fitting, the workflow calculates accuracy, precision, recall, F1, and majority-class baseline accuracy at the configured threshold. Its modest evaluation gate requires finite, completed metrics. It reports honestly if the model does not beat baseline; `fit()` success alone is not evidence of deployability. A real team would define acceptance using operational costs and requirements rather than invent a high threshold to flatter synthetic data.

## Structured results and provenance

`TrainingResult` returns the fitted `Pipeline`, typed metrics, row counts, hash, and `TrainingMetadata`. The JSON record includes:

- model name/version and timezone-aware UTC `trained_at`;
- dataset path, SHA-256, and total/train/test row counts;
- numerical/categorical features and target;
- split, random state, threshold, and iteration configuration;
- estimator type, Python version, and scikit-learn version;
- real model and baseline metrics.

Tests inject a fixed aware timestamp; normal runs use current UTC. JSON is indented and key-sorted for reviewability. Metadata improves provenance—it does not approve the artifact by itself.

## Source code is not an artifact

```text
SOURCE CODE                         MODEL ARTIFACT
describes how to build a model      contains learned fitted state
```

Changing Python does not retroactively update an already deployed `model.joblib`. Conversely, replacing an artifact changes deployable state even if application source is unchanged. Treat each new artifact as a controlled deployment input.

The generated layout is:

```text
artifacts/
└── integration-failure/
    ├── model.joblib
    └── metadata.json
```

`artifacts/` is gitignored. Commit code, documentation, configuration, tests, and the repository's deterministic fictional CSV—not generated binary state.

## Persist the complete fitted pipeline

```text
fitted preprocessing + fitted classifier → model.joblib
```

`joblib.dump()` stores the full pipeline. Saving only logistic regression and reconstructing an encoder later risks different categories, column ordering, and scaling. The artifact must carry exactly the preprocessing learned with the estimator.

### Critical serialization security boundary

> [!CAUTION]
> `joblib` and pickle-style artifacts can execute code while loading.
>
> **ONLY LOAD MODEL ARTIFACTS FROM TRUSTED, CONTROLLED SOURCES.**

Never accept a user-supplied joblib file or download an arbitrary one from the internet. Production controls should cover provenance, filesystem permissions, deployment authorization, and hashes or signatures where appropriate. `load_trusted_model_artifact()` is named to make the caller's trust decision visible; its type check occurs *after* deserialization and therefore is not a sandbox or defense against a malicious file.

Serialized scikit-learn models are also not guaranteed portable across arbitrary scikit-learn versions. Align training and inference Python/dependency environments and record their versions. A hash verifies bytes, not safety or compatibility.

## The training command

From the repository root:

```bash
python scripts/train_integration_failure_model.py
```

The small `argparse` interface supports `--data`, `--output-dir`, `--threshold`, and `--random-state`. It validates, fingerprints, splits, trains, evaluates, then—and only then—creates `model.joblib` and `metadata.json`. Its report prints actual paths, digest, row counts, features, and computed metrics.

```text
TRAIN → EVALUATE → metrics finite and evaluation complete?
                         ├── no  → fail; write nothing
                         └── yes → write artifact and metadata
```

To inspect generated metadata, open `artifacts/integration-failure/metadata.json`. Do not edit it to make a run appear different; rerun training with explicit arguments.

## Executable laboratory and round-trip contract

Run:

```bash
python examples/chapter_16_training_workflow.py
```

The laboratory validates and hashes the committed fictional dataset, trains and evaluates, writes into a temporary directory, reloads the trusted local artifact, and compares one request's probability before and after serialization. Exact equality in this same-process round trip proves the saved fitted state reproduced that prediction. The temporary directory is removed, as are all test artifacts created with `tmp_path`.

The round trip expresses an important boundary:

```text
train → save → later invocation/process → load trusted artifact → predict
```

It does not establish that the model is useful, fair, secure, or production-ready.

## Exercises

### Exercise 1 — Experiment or workflow?

Compare `model.fit(...); print(model.score(...))` with `validate → fingerprint → train → evaluate → save → record metadata`. Why is the second easier to reproduce, test, inspect, automate, and audit? What extra failure states become explicit?

### Exercise 2 — Source versus artifact

Harbor changes the Python pipeline builder but leaves an old `model.joblib` deployed. Which behavior changes immediately? What must happen before runtime uses the new model definition?

### Exercise 3 — Dataset hash

Explain what SHA-256 can prove about exact file bytes. List three things it cannot prove, including correctness and fitness for purpose.

### Exercise 4 — Security

Why must Harbor never load an arbitrary `.joblib` downloaded from an unknown source? Why does checking that the result is a `Pipeline` not make loading safe?

### Exercise 5 — Preprocessing persistence

Explain why Harbor persists the entire pipeline instead of only logistic regression. Consider learned means, scales, categories, and output column order.

### Exercise 6 — Train/test discipline

Why must scaler and encoder fitting happen through `pipeline.fit(X_train, y_train)` rather than on all data before the split?

### Coding exercise — Reproducible split configuration

The command already accepts `--random-state`. Trace it into `TrainingConfig` and metadata. Train twice with the same value and compare metrics; then use a different value. Explain why the first pair is reproducible in the same environment, why the third can differ, and how explicit randomness improves debugging. Do not repeatedly choose the seed that looks best on the test set.

## Key takeaways

1. A working experiment is not yet a maintainable training system.
2. Workflows make data, features, configuration, and outputs explicit.
3. Dataset validation happens before fitting.
4. SHA-256 identifies exact input bytes but does not validate their quality.
5. Persist fitted preprocessing and estimator together.
6. Metadata makes provenance easier to inspect.
7. Artifacts are deployable learned state, not source code.
8. Load serialized Python artifacts only from trusted, controlled sources.
9. Training and inference need compatible dependency environments.
10. Reproducibility is a software-engineering concern as much as an ML concern.

## What comes next: Chapter 17 — Evaluating the Model

Earlier chapters introduced individual metrics. Chapter 17 will ask: **How do we determine whether a model is useful enough for its intended engineering role?**

```text
MODEL → held-out predictions → accuracy / precision / recall / F1
      → confusion matrix / threshold analysis / baseline comparison
      → engineering interpretation

GOOD METRIC ≠ SAFE OR USEFUL SYSTEM
```

Chapter 17 remains planned and is not implemented here.

[Previous: Chapter 15](../part-04-banking-operations/chapter-15-database-performance-prediction.md) · [Back to Part V](README.md) · [Complete contents](../../CONTENTS.md)
