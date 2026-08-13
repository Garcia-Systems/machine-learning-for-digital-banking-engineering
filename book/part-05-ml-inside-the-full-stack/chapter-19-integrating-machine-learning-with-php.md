# Chapter 19 — Integrating Machine Learning with PHP

> **Central question:** How can a PHP full-stack application consume Harbor's Python ML prediction service safely without making the machine-learning service a critical authority for core banking behavior?

[Previous: Chapter 18](chapter-18-serving-a-model-through-an-api.md) · [Back to Part V](README.md) · [Complete contents](../../CONTENTS.md)

Chapter 18 left **Harbor Federal Credit Union** with a working Python service. A developer can send `POST /api/v1/predict/integration-failure` and receive:

```json
{"model":"harbor-integration-failure","model_version":"...","failure_probability":0.67,"threshold":0.5,"predicted_failure":true}
```

Now PHP must use that signal. `$response = file_get_contents($url);` is not an application architecture. Nor is a large Guzzle call inside a controller. Both mix transport, configuration, decoding, validation, failure policy, and business flow. They obscure timeouts, spread array keys, resist fakes, and tempt a controller to turn an advisory result into authority.

```text
CONTROLLER / SERVICE → IntegrationFailurePredictor interface → HTTP implementation → ML service
```

The full path is `PHP APPLICATION → ML CLIENT ADAPTER → HTTP / JSON → PYTHON ML SERVICE → prediction response → PHP DTO → observability / engineering logic`.

## Learning objectives

By the end, you can explain the adapter boundary; define typed request and response DTOs; call JSON with Guzzle; configure connection and whole-request timeouts; distinguish ML failure from business failure; validate unavailable or malformed responses; inject an interface and use fakes; reason about retries, fallback, circuit breaking, and latency; keep controllers free of transport; propagate model metadata into observability; and preserve deterministic banking behavior when ML is absent.

## The focused PHP package

There was no PHP application in the repository, so `php/` is deliberately a small Laravel-style library—not a pretend Laravel installation. It uses PHP 8.2+, strict types, readonly DTOs, constructor promotion, Guzzle, PSR-3 logging, and PHPUnit.

```bash
cd php
composer install
composer test
composer lint
php examples/chapter_19_ml_client.php
```

The laboratory uses deterministic Guzzle mock responses and needs no Python server. `HARBOR_ML_BASE_URL` supplies deployment configuration; `http://127.0.0.1:8000` is only a laboratory development default. Never embed production URLs or future credentials in classes. Credentials belong in configuration and secrets management.

```text
php/
├── composer.json
├── examples/chapter_19_ml_client.php
├── src/Application/VendorRequestObserver.php
├── src/Ml/...
└── tests/Chapter19Test.php
```

## Dependency inversion: the application owns the need

```php
interface IntegrationFailurePredictor
{
    public function predict(IntegrationFailureRequest $request): IntegrationFailurePrediction;
}
```

```text
HIGH-LEVEL APPLICATION LOGIC
        │
        ▼
IntegrationFailurePredictor
        ▲
        │
HTTP IMPLEMENTATION
```

The interface belongs to the application need; the HTTP adapter implements it. This is ordinary dependency inversion:

```text
APPLICATION → INTERFACE ─┬─ HTTP ML CLIENT
                         └─ FAKE / FALLBACK
```

Constructor injection lets production wiring select HTTP while unit tests select a fake. Identity vendors, payment providers, document services, and ML prediction services are all external dependencies. Each needs a contract, timeout, failure handling, observability, version management, and tests.

## Typed request and explicit serialization

Chapter 18's v1 request has exactly eight fields—no `request_id` was implemented—so PHP must not invent a ninth:

```php
final readonly class IntegrationFailureRequest
{
    public function __construct(
        public string $vendor,
        public string $endpoint,
        public float $recentVendorLatencyMs,
        public float $recentVendorErrorRate,
        public int $queueDepth,
        public int $retryCount,
        public int $requestSizeBytes,
        public int $hourOfDay,
    ) {}

    public function toApiPayload(): array
    {
        return [
            'vendor' => $this->vendor,
            'endpoint' => $this->endpoint,
            'recent_vendor_latency_ms' => $this->recentVendorLatencyMs,
            'recent_vendor_error_rate' => $this->recentVendorErrorRate,
            'queue_depth' => $this->queueDepth,
            'retry_count' => $this->retryCount,
            'request_size_bytes' => $this->requestSizeBytes,
            'hour_of_day' => $this->hourOfDay,
        ];
    }
}
```

PHP object naming and wire naming need not match. The serializer owns `recentVendorLatencyMs` → `recent_vendor_latency_ms`. A contract test compares the complete payload, catching accidental renamed or extra keys.

## A validated response DTO, not a roaming array

```php
final readonly class IntegrationFailurePrediction
{
    public function __construct(
        public string $model,
        public string $modelVersion,
        public float $failureProbability,
        public float $threshold,
        public bool $predictedFailure,
    ) {}
}
```

Again, v1 has no `request_id`. `$result['failure_probability']` is weaker than `$prediction->failureProbability`: a DTO supplies explicit types, stable contracts, centralized validation, safer refactoring, and better tests.

`fromApiPayload()` distrusts downstream JSON. It requires `model`, `model_version`, `failure_probability`, `threshold`, and `predicted_failure`; nonempty model strings; finite numeric values in `[0, 1]`; and an actual boolean. It also verifies Chapter 18's invariant:

```text
predicted_failure == (failure_probability >= threshold)
```

Missing, malformed, or inconsistent data is not a prediction.

## The Guzzle HTTP adapter

`HttpIntegrationFailurePredictor` accepts `ClientInterface`, a base URL, and timeout configuration:

```php
$response = $this->client->request(
    'POST',
    rtrim($this->baseUrl, '/') . '/api/v1/predict/integration-failure',
    [
        'json' => $request->toApiPayload(),
        'connect_timeout' => $this->connectTimeoutSeconds,
        'timeout' => $this->timeoutSeconds,
        'http_errors' => false,
    ],
);
```

Guzzle's `json` option safely encodes and sets content type; do not encode manually. The adapter targets `/api/v1`. An incompatible future contract should use `/api/v2` or a coordinated migration.

The 0.2-second connect and 0.5-second total defaults are teaching examples, **not universal production values**:

```text
connect timeout → time allowed to establish a connection
request timeout → time allowed for the entire request
```

Choose actual values from measured behavior, the caller's latency budget, and product requirements.

### Status and exception policy

`MlPredictionUnavailable` is the intentional application abstraction. The adapter maps connection failure, timeout, non-200 response, invalid JSON, missing fields, invalid types/ranges, and inconsistency to it. `500`/`503` represent availability trouble. `422` probably reveals a v1 contract mismatch or bad telemetry and must be conspicuously observable as an integration defect. Never expose an internal ML response body to members.

The application decides how to handle this exception. Do not catch every `Throwable`, conceal programming faults, or leak generic transport exceptions across the boundary.

## The core rule: ML failure is not banking failure

```text
CORE HARBOR OPERATION
        ├── authentication
        ├── authorization
        ├── deterministic validation
        ├── vendor request
        └── financial/business rules

ML prediction → observability enhancement
```

If ML fails, deterministic operation continues. Here ML **fails open with respect to the core banking workflow**: ignore the optional advisory signal and continue established deterministic application behavior. This does **not** bypass or weaken authentication, authorization, fraud controls, validation, accounting, or security.

| Vendor API | ML API | Core request behavior |
| --- | --- | --- |
| available | available | normal operation + ML signal |
| available | unavailable | normal operation, no ML signal |
| unavailable | available | vendor failure handled by deterministic integration logic |
| unavailable | unavailable | vendor failure handled normally; ML signal unavailable |

`UnavailableIntegrationFailurePredictor` throws explicit unavailability. It never fabricates `failure_probability = 0`, which would falsely report confident success. `FakeIntegrationFailurePredictor` returns a deliberately supplied DTO for tests.

## Application service: observe, never approve or deny

`VendorRequestObserver` depends only on the interface and `LoggerInterface`:

```php
try {
    $prediction = $this->predictor->predict($request);
    $this->logger->info('ml.integration_failure_prediction', [
        'model' => $prediction->model,
        'model_version' => $prediction->modelVersion,
        'failure_probability' => $prediction->failureProbability,
        'threshold' => $prediction->threshold,
        'predicted_failure' => $prediction->predictedFailure,
        'vendor' => $request->vendor,
        'endpoint' => $request->endpoint,
    ]);
} catch (MlPredictionUnavailable $error) {
    $this->logger->warning('ml.integration_failure_unavailable', [
        'vendor' => $request->vendor,
        'endpoint' => $request->endpoint,
        'reason' => $error->getMessage(),
    ]);
}
```

Model, version, probability, threshold, result, vendor, endpoint, and—if a future contract adds it—request ID are useful safe technical context. Do not log credentials, tokens, account numbers, raw member data, or full sensitive payloads. This service emits engineering evidence; it approves and denies nothing.

## Anti-pattern: ML inside the controller

```php
public function transfer(Request $request)
{
    $client = new Client();
    $ml = $client->post(...);
    if ($ml['predicted_failure']) {
        abort(503);
    }
    // ...core behavior...
}
```

This couples transport to the controller, makes advisory ML authoritative, impairs testing, leaves timeout and fallback unclear, avoids centralized validation, and binds core behavior to a prediction. Prefer `controller/service → injected interface → HTTP adapter → ML service`; the observer consumes a typed result without changing the operation.

## Testing the seam

```text
UNIT TEST                 INTEGRATION TEST
PHP application           PHP HTTP predictor
      │                          │
      ▼                          ▼
fake predictor             FastAPI test/server boundary
```

The `+` characters above emphasize two separate paths, not addition. PHPUnit uses Guzzle `MockHandler`, so normal tests require no network or Python server. Tests cover exact serialization, valid parsing, malformed JSON, missing fields, out-of-range values, threshold inconsistency, connection/timeout and HTTP 500 mapping, fake behavior, model-version logging, graceful unavailability, and unchanged deterministic outcomes. A separately started Uvicorn server can support an optional cross-language smoke test, but background-process management is intentionally not part of the stable suite.

## Retries, circuit breaking, and latency

Prediction is read-only, so repeating the same request is conceptually safe. It is not operationally free: retries add latency and load, worsen outages, and can cause cascading failure. This client prefers **timeout + graceful failure** and performs no automatic retry. A measured, budgeted single retry might be justified; an unbounded retry loop is not.

```text
ML service repeatedly fails → stop calling temporarily → use fallback
```

That circuit-breaker idea avoids repeatedly paying a slow timeout and permits recovery. State, concurrency, probes, and metrics make it nontrivial, so implementation is deferred.

Synchronous scoring is `PHP request → ML service → response`, but total latency is `application work + vendor call + database + ML call`. Ask: **Does the application need the prediction before responding?** Pure observability often does not. `application event → queue → ML scoring → monitoring` reduces member-response coupling, at the cost of delay and queue operations. Queues are not implemented here.

## Security and deployment boundaries

- Use TLS in production and restrict internal ML service exposure.
- Authenticate service-to-service traffic when appropriate; load credentials from secrets management.
- Validate response schemas; never trust arbitrary downstream JSON.
- Configure strict, measured timeouts.
- Avoid secrets and member data in logs.
- Treat model version as observability metadata, not authority.
- Never expose internal bodies or stack traces to members.

Production TLS, authentication, service discovery, queues, and circuit breakers are outside this focused laboratory.

## Coding exercise — observability decorator

`LoggingIntegrationFailurePredictor` is included and tested. Recreate it without looking. It must call its inner predictor, measure elapsed time, log model version and latency, return the original prediction unchanged, log unavailability and rethrow the same exception, and never alter probability or business behavior. A decorator adds cross-cutting observability without changing prediction semantics.

## Exercises

### Exercise 1 — Where should HTTP logic live?

Choose controller, dedicated ML adapter, or database model. **Answer:** the adapter owns one external contract and remains replaceable without mixing transport into presentation or persistence.

### Exercise 2 — DTO or array?

Why convert JSON to a DTO? **Answer:** types, centralized validation, stable naming translation, refactorability, and focused tests prevent wire details from spreading.

### Exercise 3 — Timeout

What if advisory ML takes seconds? **Answer:** stop waiting at a measured deadline, record unavailability, continue deterministically, and consider asynchronous scoring.

### Exercise 4 — Fallback

Why is this dangerous?

```php
catch (...) {
    return new IntegrationFailurePrediction(failureProbability: 0.0, predictedFailure: false, ...);
}
```

It misrepresents absence of evidence as high-confidence success. Model unavailability explicitly.

### Exercise 5 — Business authority

Should `predicted_failure = true` automatically prevent a vendor call? **No.** It is advisory engineering evidence, not authorization or a deterministic integration rule.

### Exercise 6 — API version

Why target `/api/v1/...`? **Answer:** deployed client and server can coordinate incompatible migrations; silent unversioned changes are unsafe.

## Key takeaways

1. An ML service is another external dependency.
2. PHP depends on an interface, not raw HTTP.
3. DTOs create explicit contracts.
4. Timeouts and failure behavior are architecture.
5. Unavailability must not masquerade as low risk.
6. Advisory ML is not authoritative for core banking behavior.
7. Safe read-only retries can still worsen outages.
8. Observability records model version and prediction context safely.
9. Dependency injection makes integration testable.
10. Cross-language ML integration is normal API engineering.

## What comes next: Chapter 20 — Building an ML-Assisted Monitoring Dashboard

The application can now obtain a signal. Next: **How should engineers see and interpret predictions alongside normal telemetry?**

```text
API latency
vendor latency
error rate
queue depth
model probability
model version
recent predictions
        │
        ▼
ENGINEERING DASHBOARD
```

Chapter 20 combines deterministic metrics, predictions, anomaly signals, model version, clear uncertainty language, and no automatic root-cause claims.

[Previous: Chapter 18](chapter-18-serving-a-model-through-an-api.md) · [Next: Chapter 20](chapter-20-building-an-ml-assisted-monitoring-dashboard.md) · [Back to Part V](README.md) · [Complete contents](../../CONTENTS.md)
