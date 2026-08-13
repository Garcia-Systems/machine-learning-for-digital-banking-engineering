# Chapter 21 — Security and Sensitive Financial Data

> **Part VI — Responsible Machine Learning in Financial Systems**

Harbor Federal Credit Union is entirely fictional. Every name, payload, dataset, and result in this chapter is synthetic educational material.

**Central question:** How can Harbor build ML-assisted systems without unnecessarily exposing sensitive member, financial, authentication, operational, or model data?

```text
THE APPLICATION CAN ACCESS DATA

does not mean

THE ML SYSTEM SHOULD USE THAT DATA
```

The engineering principle is direct:

> Collect, transform, transmit, train on, log, and retain only the information required for the specific ML problem.

## Opening scenario: “more” is not automatically “better”

Harbor's system now spans an entire path:

```text
historical datasets
      │
      ▼
training workflow
      │
      ▼
model artifact
      │
      ▼
prediction API
      │
      ▼
PHP application
      │
      ▼
monitoring dashboard
```

A developer proposes adding `member_name`, `account_number`, `email`, a full vendor payload, `access_token`, the request body, and a recent account balance to the integration-failure model. “More data might make the model better.” But does this engineering problem actually require any of those fields? Generally, no.

```text
ENGINEERING QUESTION

Will this vendor-backed integration request fail?
        │
        ▼
RELEVANT OPERATIONAL SIGNALS

vendor
endpoint
latency
recent errors
queue
retry count
        │
        ▼
NO NEED FOR

name
account number
SSN
balance
access token
raw identity documents
```

The useful request-time context is `vendor`, `endpoint`, recent vendor latency and error rate, queue depth, retry count, request size, and hour. Member identity does not answer the question. Availability is not necessity, and a small metric improvement would not by itself justify collection.

## Learning objectives

By the end of this chapter, you should be able to:

1. explain data minimization and distinguish necessary features from available data;
2. identify sensitive financial and identity data and authentication secrets;
3. distinguish pseudonymization from anonymization;
4. explain how logs create secondary exposure;
5. validate schemas against explicit allowlists and design narrow API contracts;
6. protect training metadata and code-like model artifacts;
7. describe memorization, inversion, and membership-inference risks conceptually;
8. define access boundaries for training, inference, clients, and dashboards;
9. treat retention and debugging data as deliberate security decisions; and
10. automate teaching guards against prohibited dataset and payload fields.

## Security is a lifecycle property

```text
DATA SOURCE
    │
    ▼
COLLECTION
    │
    ▼
TRAINING DATA
    │
    ▼
FEATURE PIPELINE
    │
    ▼
MODEL ARTIFACT
    │
    ▼
PREDICTION API
    │
    ▼
APPLICATION
    │
    ▼
LOGS / DASHBOARD
```

> Sensitive information can leak at any stage.

TLS and endpoint authentication are important, but protecting only the API endpoint is insufficient. An over-broad export, notebook copy, serialized preprocessor, exception trace, application log, or dashboard can expose data before or after inference.

Security also involves three related properties:

```text
CONFIDENTIALITY   who can see the data?
INTEGRITY         can the data/model be modified unexpectedly?
AVAILABILITY      can the system operate when a component fails?
```

Harbor protects member data for confidentiality, fingerprints artifacts and controls deployment for integrity, and keeps deterministic banking operation available when the advisory ML service is unavailable. This is a focused engineering model, not a complete cybersecurity program.

## A practical teaching classification

### Public or non-sensitive educational data

Examples include a fictional model name, chapter number, generic endpoint category, and synthetic documentation.

### Internal operational data

API latency, queue depth, database connections, vendor response statistics, and model version may not identify a member. They can still reveal internal behavior and deserve controlled access.

### Member-related sensitive data

Account numbers, transaction details, balances, identity-verification information, contact information, and member communications can expose a member or their financial activity.

### Authentication and security secrets

Passwords, access and refresh tokens, session secrets, API keys, private keys, and authentication cookies enable access rather than merely describing it.

> Authentication secrets should not become ML features, dataset columns, logs, or dashboard fields.

Operational credentials needed by a future service belong in environment configuration, a secret-management system, or runtime injection—not source code, datasets, model features, model metadata, or Git.

## A small prohibited-name guard—and its limit

Chapter 21 implements this reusable teaching set in `harbor_ml.data_security`:

```python
PROHIBITED_SENSITIVE_FIELDS = frozenset(
    {
        "ssn",
        "social_security_number",
        "account_number",
        "routing_number",
        "card_number",
        "cvv",
        "password",
        "access_token",
        "refresh_token",
        "api_key",
        "private_key",
        "authentication_cookie",
        "full_name",
        "email_address",
    }
)
```

The implementation also names a few obvious fictional member-field variants used by the exercises. This is neither exhaustive nor a legal definition. Exact-name scanning will miss an unfamiliar name, nested field, encoded secret, or sensitive value under `notes`. It does not inspect arbitrary values and must not be described as data-loss prevention.

> An allowlist is stronger than relying only on a blacklist.

A blacklist asks developers to anticipate every dangerous name. An allowlist makes every new field require deliberate review.

## Allowlist-first feature design

```text
BAD DESIGN

take entire application object
        │
        ▼
remove fields we remember are sensitive
        │
        ▼
model

BETTER DESIGN

explicitly define approved ML features
        │
        ▼
construct feature object containing only those fields
        │
        ▼
model
```

The security policy declares the exact contract:

```python
APPROVED_INTEGRATION_FEATURES = (
    "vendor",
    "endpoint",
    "recent_vendor_latency_ms",
    "recent_vendor_error_rate",
    "queue_depth",
    "retry_count",
    "request_size_bytes",
    "hour_of_day",
)
```

`build_integration_feature_payload()` validates the keys, then constructs a new mapping in this order. The outcome `request_failed`, timestamps, and identifiers are not features. Existing model code likewise builds its matrix by iterating an explicit feature tuple rather than dumping the observation.

### Data minimization

> Data minimization means using the smallest amount of information necessary for the stated purpose.

Bad input for this problem:

```json
{
  "member_name": "Example Member",
  "account_number": "123456789",
  "balance": 10422.18,
  "vendor": "ClearVerify",
  "vendor_latency_ms": 1200
}
```

Better fictional input:

```json
{
  "vendor": "ClearVerify",
  "endpoint": "identity_verify",
  "recent_vendor_latency_ms": 1200,
  "recent_vendor_error_rate": 0.04,
  "queue_depth": 63,
  "retry_count": 2,
  "request_size_bytes": 2200,
  "hour_of_day": 14
}
```

Minimization must occur before transmission and logging, not as a cleanup job after everything has been copied.

## Identifiers and pseudonyms

`request_id`, `trace_id`, and `session_id` can support correlation, tracing, and debugging. That does not make them predictive inputs.

```text
request_id

useful for:
tracing

usually useless or dangerous as:
predictive feature
```

An arbitrary identifier can become a shortcut, allow rows to be joined unnecessarily, or encourage memorization. Keep tracing metadata separate from a model feature row unless the problem supplies a specific, reviewed reason.

Replacing:

```text
member_id = 874221
```

with:

```text
member_token = "7af1..."
```

reduces direct identifiability in some contexts, but the token may still be linkable to another table or repeated activity.

```text
PSEUDONYMOUS
≠
ANONYMOUS
```

Hashing an identifier does not erase linkage or contextual clues. This is an engineering warning, not a claim about a jurisdiction's legal anonymization standard.

## Do not retain raw payloads by default

Avoid storing full HTTP request bodies, full vendor responses, identity documents, authentication headers, or raw SOAP envelopes merely because middleware makes capture easy. A controlled adapter should extract the small, typed representation that downstream code needs:

```text
RAW VENDOR RESPONSE
        │
        ▼
CONTROLLED ADAPTER
        │
        ▼
SAFE INTERNAL FIELDS
        │
        ▼
TELEMETRY / ML
```

This preserves a normal integration boundary: vendor-specific complexity ends in the adapter rather than spreading into training data and observability.

## Logging is a secondary data surface

A correctly authenticated, narrow prediction API can still leak through logs, backups, log search, exception trackers, or dashboard ingestion. This is a fragile pattern:

```python
logger.info("Prediction request: %s", request.model_dump())
```

If a later API revision gains a sensitive field, the field enters logs automatically. Do not routinely log authorization headers, tokens, cookies, full bodies, or member identity.

Prefer an explicit context:

```python
logger.info(
    "integration_failure_prediction",
    extra={
        "vendor": request.vendor,
        "endpoint": request.endpoint,
        "model_version": runtime.model_version,
    },
)
```

`build_safe_log_context()` always constructs only those three keys. It intentionally ignores other input. The FastAPI success log uses this helper rather than serializing the request. A safe logging helper is not permission to retain logs forever: access and retention still matter.

## Fail closed on dataset schema drift

The integration training dataset permits exactly:

```text
timestamp
vendor
endpoint
recent_vendor_latency_ms
recent_vendor_error_rate
queue_depth
retry_count
request_size_bytes
hour_of_day
request_failed
```

Validation checks required columns, prohibited names, and unexpected columns. Imagine yesterday's export contained `vendor`, `endpoint`, latency, queue depth, and `request_failed`, but a changed exporter adds `member_email`. A permissive loader might silently keep it. Harbor's strict boundary stops before fitting:

```text
Unexpected dataset field: member_email
```

In the implemented guard, a known prohibited name produces the even more specific `prohibited field: member_email`. Either outcome demands review. Strictness is appropriate for this key training pipeline; a deliberately changed contract requires a code and test change.

Feature contract tests assert the exact list and separately prove that the target, identifiers, leakage fields, and prohibited names are absent. Tests also inspect committed CSV headers. These are repository-level teaching guards, not comprehensive content discovery.

Run the limited audit:

```bash
python scripts/audit_ml_data.py
```

It reports each committed CSV's columns and checks exact prohibited header names. It applies the complete strict schema only to `harbor_integration_requests.csv`; giant mappings for unrelated chapter fixtures would be brittle and add little value.

## Minimize application contracts

### FastAPI

Do not accept this contract:

```text
POST /predict
{
  "application_object": { ...everything... }
}
```

Chapter 18's Pydantic request accepts only the eight prediction fields and forbids extras. A narrow DTO reduces accidental leakage, coupling, logging risk, ambiguity, and future misuse. Validation rejects negative queue depth, `hour_of_day = 50`, `recent_vendor_error_rate = 8.2`, empty vendor names, and unexpected types. Validation protects model assumptions and service stability, but it does not make authorization, transport, provenance, and operational controls unnecessary.

### PHP

Never pass `$request->all()` or an entire domain object. This crosses too much state:

```php
$client->post('/predict', [
    'json' => $member->toArray(),
]);
```

Instead, Chapter 19 constructs its typed `IntegrationFailureRequest` from approved operational values:

```php
$predictionRequest = new IntegrationFailureRequest(
    vendor: $vendor,
    endpoint: $endpoint,
    recentVendorLatencyMs: $latency,
    recentVendorErrorRate: $errorRate,
    queueDepth: $queueDepth,
    retryCount: $retryCount,
    requestSizeBytes: $requestSize,
    hourOfDay: $hour,
);
```

Its serializer explicitly names each JSON key. It does not receive a member object.

### Output minimization

Do not reveal filesystem paths, dataset paths, coefficients, or stack traces:

```json
{
  "model_path": "/srv/models/private/model.joblib",
  "training_dataset": "/data/internal.csv",
  "classifier_coefficients": [],
  "python_stack_trace": "..."
}
```

The public response needs only:

```json
{
  "model": "harbor-integration-failure",
  "model_version": "...",
  "failure_probability": 0.61,
  "threshold": 0.5,
  "predicted_failure": true
}
```

The engineering dashboard similarly needs aggregated technical telemetry—not raw identity, secrets, complete payloads, or exact sensitive transaction details.

## Artifacts and metadata are sensitive operational assets

A serialized model can reveal model structure, learned parameters, preprocessing categories, and operational feature names. Pickle/joblib deserialization can execute code from a malicious artifact, so Harbor loads only local artifacts whose provenance it controls. Never download `model.joblib` from an unknown URL and load it.

Treat an artifact as a trusted code-like deployment asset with:

- controlled provenance and limited write access;
- controlled deployment and version tracking;
- integrity checks; and
- read access only where inference requires it.

Chapter 16 already provides `calculate_file_sha256()`. Chapter 21 places the reusable byte-hashing implementation in the security module:

```text
model.joblib
    │
    ▼
SHA-256
    │
    ▼
artifact fingerprint
```

A hash can detect changed bytes when compared with a trusted expected value. It does not establish who created a file or whether it is safe. The laboratory hashes a temporary fictional artifact without redesigning Chapter 16 metadata and avoiding the circular problem of embedding an artifact's own final hash within itself.

```text
training process   → allowed to write artifact
prediction service → usually needs read access only
```

## Least privilege and trust boundaries

```text
COMPONENT
gets only what it needs
```

| Component | Usually needs | Usually does not need |
|---|---|---|
| Training job | approved training dataset; artifact output directory | production banking credentials; member database write access |
| Prediction service | trusted model artifact; narrow request | training dataset; labels; arbitrary member database access |
| PHP application | narrow prediction client; advisory result | artifact internals; training export |
| Dashboard | summarized telemetry and prediction metadata | raw training data; passwords; identity documents |

```text
                 TRUST BOUNDARIES

PHP APPLICATION
      │
      │ validated HTTPS request
      ▼
ML SERVICE
      │
      │ trusted local artifact
      ▼
MODEL
```

```text
TRAINING ENVIRONMENT
      │
      ▼
MODEL ARTIFACT
      │
      ▼
DEPLOYMENT ENVIRONMENT
```

Every crossing calls for validation, authentication and authorization as appropriate, a controlled data contract, and disciplined logging. Training and deployment identities should have distinct permissions.

## Retention and copies

> Retention is part of security design.

“Store every prediction forever” is not a neutral default. Ask why the data is retained, how long it remains useful, whether it carries identifiers, whether aggregates suffice, who can access it, and how deletion works. This chapter deliberately does not prescribe jurisdiction-specific retention periods.

Exports multiply the surface:

```text
production system
      │
      ▼
training export 1
      │
      ├── local laptop copy
      ├── shared drive copy
      └── notebook copy
```

Every copy needs controls and eventually a disposition. This repository is different by design: all datasets are synthetic, fictional, and intentionally committed for education. That convention must remain true as new examples arrive.

Debugging convenience is not a purpose by itself. If detailed diagnostic data is temporarily justified, document scope, access, redaction, and deletion rather than converting an emergency capture into permanent telemetry.

## Predictive value is not justified use

Chapter 6 asked whether a field helps prediction. Chapter 21 adds a separate gate:

```text
PREDICTIVE VALUE
      ≠
JUSTIFIED DATA USE
```

Even if a sensitive field slightly improves a test score, Harbor must weigh necessity, sensitivity, operational risk, fairness, governance, and maintainability. First ask whether a less sensitive operational signal answers the engineering question.

Training data influences parameters. High-capacity model classes can sometimes memorize or expose details more readily than Harbor's small logistic-regression examples. We should not imply these teaching models store member records verbatim; the general risk still supports minimization.

At a high level, attackers may try **membership inference**—estimating whether a record participated in training—or **model inversion**, estimating properties reflected in model behavior. This chapter provides no attack procedure. Its defensive conclusion is sufficient:

> Do not train on sensitive data unless there is a legitimate need and appropriate controls.

## Lightweight threat modeling

For each component, ask:

```text
What sensitive data enters?
What leaves?
What is stored?
What is logged?
Who can access it?
What happens if compromised?
```

- **Training workflow:** inspect export columns, copies, labels, artifact output, metadata, and write identity.
- **Prediction API:** inspect request/response DTOs, validation, service authorization, error handling, and access logs.
- **PHP client:** inspect domain-to-DTO mapping, HTTP configuration, retries, and application logging.
- **Dashboard:** inspect telemetry aggregation, audience, caching, and drill-down data.

A short review at each boundary catches different failures than a single perimeter review.

## Security anti-patterns

### 1. API object equals feature object

```python
features = request.model_dump()
```

Passing everything couples API evolution to ML features. Construct from the feature allowlist instead.

### 2. Whole-object logging

```python
logger.info(request.model_dump())
```

Future sensitive fields automatically enter logs. Construct safe context explicitly.

### 3. Whole-domain-object transport

```php
'json' => $member->toArray()
```

An entire domain object crosses a service boundary. Use the narrow DTO.

### 4. Unknown model artifact

```text
model.joblib downloaded from unknown URL
```

Untrusted pickle/joblib loading can execute code. Require controlled provenance.

### 5. Permanent developer copy

```text
training.csv copied to developer desktop forever
```

Uncontrolled copies expand exposure. Minimize, control, and dispose of exports.

## Executable laboratory

Run:

```bash
python examples/chapter_21_data_security.py
```

It prints the approved feature contract, accepts a safe fictional payload, rejects `access_token` and `account_number`, constructs safe log context, validates the committed integration dataset header, hashes temporary artifact bytes, and summarizes least privilege. The temporary directory is removed automatically.

The core functions are deliberately small:

- `find_prohibited_fields()` provides limited exact-name detection;
- `validate_prediction_payload_fields()` requires an exact request allowlist;
- `validate_dataset_columns()` requires the exact training schema;
- `build_integration_feature_payload()` constructs only approved inputs;
- `build_safe_log_context()` constructs only approved log metadata; and
- `calculate_file_sha256()` fingerprints actual file bytes.

Explicit `SensitiveFieldError` and `UnexpectedFieldError` exceptions make failures testable. These helpers are education-scale policy boundaries, not a generic security or legal classification framework.

## Exercises

### Exercise 1 — Necessary or merely available?

Classify vendor latency, queue depth, member name, retry count, account balance, vendor error rate, and access token. For integration failure, the operational latency, queue, retry, and vendor error fields are relevant. Identity, balance, and tokens are not justified by this question; a token must never be a feature.

### Exercise 2 — Feature or tracing field?

Should `request_id` become a feature merely because it helps debugging? Explain why correlation metadata should normally remain separate and how an identifier can create shortcuts or memorization.

### Exercise 3 — Logging

Why is `logger.info(request.model_dump())` dangerous over time? Consider future schema evolution, log audiences, backups, and retention.

### Exercise 4 — Pseudonymization

Why does hashing `member_id` not automatically anonymize a dataset? Consider stable linkage, auxiliary tables, and surrounding observations.

### Exercise 5 — Artifact trust

Why must Harbor avoid `.joblib` artifacts from unknown sources? Separate integrity checking from provenance and recall that deserialization is code-like.

### Exercise 6 — Least privilege

Which component usually needs historical training data?

A. PHP application  
B. prediction API  
C. training workflow  
D. dashboard

Usually only **C**.

### Coding exercise — reject an unsafe addition

Add fictional `member_email` to a temporary payload—never to a committed dataset. Then:

1. prove the strict validator rejects it;
2. prove safe log context excludes it;
3. prove the allowlist builder cannot put it in the feature row;
4. add deterministic tests; and
5. explain why an allowlist is stronger than developers remembering sensitive names.

## Key takeaways

1. Available data is not automatically justified ML data.
2. Feature contracts should be explicit allowlists.
3. Authentication secrets should never become model features or routine logs.
4. Pseudonymous data can still be sensitive.
5. Raw request and vendor payloads should not be copied into ML systems unnecessarily.
6. Prediction APIs should accept and return only their contract requirements.
7. Training datasets, artifacts, logs, and dashboards are all security surfaces.
8. Serialized Python artifacts are trusted code-like assets.
9. Least privilege and retention limits reduce exposure.
10. Data minimization is an engineering design choice, not merely cleanup.

## What comes next: Chapter 22 — Explainability

Chapter 21 asks:

> What information should the model be allowed to use?

Chapter 22 will ask:

> Once a model produces a prediction, what can developers legitimately say about why the model produced it?

```text
PREDICTION
    │
    ▼
model internals
    │
    ▼
feature contribution / model explanation
    │
    ▼
engineering interpretation
```

It will distinguish:

```text
MODEL EXPLANATION
how the fitted model used its inputs

from

CAUSAL EXPLANATION
what actually caused the real-world outcome
```

Possible tools include logistic-regression coefficients, transformed per-request features, simple local linear contributions, and—where appropriate—permutation importance. Chapter 22 remains planned and is not implemented here.

[Previous: Chapter 20](../part-05-ml-inside-the-full-stack/chapter-20-building-an-ml-assisted-monitoring-dashboard.md) · [Back to Part VI](README.md) · [Complete contents](../../CONTENTS.md)
