# Chapter 31 — Integrating the Banking Application

> **Central question:** How can Harbor's PHP application consume several independent ML signals through one maintainable application boundary while keeping all core banking behavior deterministic?

[Previous: Chapter 30 — Building the ML Service](chapter-30-building-the-ml-service.md) · [Back to Part VII](README.md) · [Complete contents](../../CONTENTS.md)

## Opening scenario

Chapter 30 gives **Harbor Federal Credit Union** one Python service with several endpoints. It would be easy to scatter calls through controllers:

```php
$anomaly = $client->post('/api/v1/score/telemetry-anomaly', ...);
$incident = $client->post('/api/v1/predict/incident', ...);
$failure = $client->post('/api/v1/predict/integration-failure', ...);
```

That leaks transport concerns into application code, manually handles three contracts, produces inconsistent timeouts, makes partial failure difficult, spreads raw arrays, and couples business behavior to HTTP. Chapter 31 instead extends Chapter 19's PHP package:

```text
APPLICATION SERVICE
       │
       ▼
MachineLearningGateway
       │
       ▼
HTTP IMPLEMENTATION
       │
       ▼
CAPSTONE ML SERVICE
```

The complete flow is:

```text
                      HARBOR PHP APPLICATION
                               │
                               ▼
                        ML GATEWAY INTERFACE
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       anomaly endpoint   incident endpoint   request-failure endpoint
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                       typed ML observations
                               │
                               ▼
                 APPLICATION OBSERVABILITY
```

Every signal is advisory. It must not determine authentication, authorization, transaction validity, transfer approval, member eligibility, pricing, account access, vendor-contract correctness, or idempotency.

## Learning objectives

By the end, you can define a gateway; model anomaly, incident, and existing integration-failure results with typed DTOs; distinguish unavailable from low risk; safely call independent endpoints; reason about sequential and concurrent calls; centralize timeouts; retain partial results, uncertainty, and independent versions; avoid raw JSON in business services; test with fakes and mocked HTTP; verify the Python/PHP contract; and keep banking behavior deterministic during complete ML outage.

## One application boundary, three meanings

`MachineLearningGateway` represents Harbor's **application need for ML capabilities**. It does not represent one giant model:

```text
ONE APPLICATION INTERFACE
        ≠
ONE MODEL SEMANTIC
```

```php
interface MachineLearningGateway
{
    public function scoreTelemetryAnomaly(TelemetrySnapshot $telemetry): AnomalyPrediction;
    public function predictIncident(TelemetrySnapshot $telemetry): IncidentPrediction;
    public function predictIntegrationFailure(IntegrationFailureRequest $request): IntegrationFailurePrediction;
}
```

Endpoint paths are infrastructure knowledge in `HttpMachineLearningGateway`, never application-service knowledge. An anomaly score measures unusualness and is **not a probability**. An incident distribution describes resemblance among canonical patterns. A failure probability estimates a request outcome before a vendor call. A generic `$mlSaysBad` boolean destroys all three meanings.

## Exact request DTOs

`TelemetrySnapshot` contains exactly Chapter 30's seven accepted fields: API latency, error rate, database connections, queue depth, vendor latency, requests per minute, and retry count. It explicitly maps PHP names onto the snake-case wire contract:

```text
apiLatencyMs → api_latency_ms
```

No rolling five-minute fields are added: Chapter 30's public schema does not accept them. Likewise, Chapter 19's `IntegrationFailureRequest` remains unchanged. An allowlist serializer means a member domain object, credential, or arbitrary request body cannot accidentally cross the PHP-to-Python boundary.

## Typed responses retain semantics

`AnomalyPrediction` retains `model`, `modelVersion`, `anomalyScore`, and `isAnomaly`. Its parser requires a finite score but deliberately permits values outside `[0,1]`; Chapter 28 did not define that score as a probability.

`IncidentPrediction` uses the closed `IncidentClass` enum:

```php
enum IncidentClass: string
{
    case Normal = 'normal';
    case VendorDegradation = 'vendor_degradation';
    case DatabasePressure = 'database_pressure';
    case TrafficSpike = 'traffic_spike';
    case ApplicationRegression = 'application_regression';
}
```

It retains the entire probability map, top and second probabilities, their gap, and `ambiguous`. Endpoint-specific validation requires numeric finite probabilities in `[0,1]`, a sum approximately equal to one, a known predicted class present in the map, and ranking fields consistent with the map. The ambiguity flag is retained as API output rather than recomputed using a PHP-only policy.

Chapter 19's `IntegrationFailurePrediction` is reused, including its `failureProbability`, `threshold`, and `predictedFailure` consistency checks. Symmetry is not a reason to redesign a stable contract.

Versions also stay independent:

```text
anomaly model:     abc123
incident model:    def456
integration model: ghi789
```

There is no invented combined model version. The service API version (`v1`) is a wire-contract version, not a fitted-artifact version.

## Transport adapter and timeout budget

`MlClientConfig` centralizes base URL, connect timeout, and request timeout. A single teaching budget is clearer today; endpoints can receive separate evidence-based budgets later. `HttpMachineLearningGateway::postJson()` centralizes URL assembly, JSON encoding, timeouts, status handling, and JSON decoding. Semantic parsers remain separate—there is no magical `parsePrediction()`.

A `422` for a payload PHP believed valid is a contract defect: perhaps schema drift, incompatible deployments, or serialization. `MlContractViolation` distinguishes it, malformed JSON, and invalid response semantics from ordinary `MlPredictionUnavailable` transport failures such as timeout, connection refusal, or `503`. Both make that prediction unavailable, but operators investigate them differently. Do not log full bodies.

## Missing is not normal

Never substitute a zero:

```php
// Wrong: invents evidence
$failureProbability = $unavailable ? 0.0 : $prediction->failureProbability;
```

`MlObservation` instead holds nullable typed results and exposes explicit per-signal `available`/`unavailable` status. Null means no prediction exists; it does not mean normal. `MlCapabilityStatus` summarizes the three capabilities as `all_available`, `partially_available`, or `unavailable`. It is deliberately not called “healthy”: ML capability status is not Harbor system health.

`MlObservationService` calls each endpoint separately, catches endpoint-level unavailability, preserves every success, and emits `harbor.ml_observation`. If the incident endpoint fails while anomaly and integration failure succeed, the aggregate remains:

```text
anomaly:             available
incident:            unavailable
integration failure: available
ML capability:       partially_available
```

The event contains availability, each independent model version, selected scores/flags, incident class and ambiguity, and capability status. The full incident map stays in the DTO for a dashboard but is not logged. Warning logs identify the failed signal and distinguish a contract violation. No unavailable value is serialized as zero or “normal.”

## Sequential first, concurrency later

The executable service calls anomaly, then incident, then optional integration failure. This is easy to understand, handle, and test. If calls take 20, 30, and 25 ms, ML latency can be roughly 75 ms plus HTTP overhead.

Anomaly and incident scoring are independent:

```text
anomaly ─────┐
             ├── concurrent
incident ────┘
```

Guzzle promises could reduce wall-clock latency, but concurrency complicates per-call error handling, resource use, cancellation, and debugging. It is an optional optimization, not this chapter's essential architecture. Measure before adding it.

More importantly, ask: **Does Harbor need every result before responding to the member?** Monitoring predictions probably need not block a response. A future design could use:

```text
application event → queue/event stream → ML scoring → dashboard
```

This chapter stays synchronous and implements no queue. Strict timeouts keep an advisory dependency from exhausting PHP workers. Production designs may add a circuit breaker, asynchronous scoring, and an isolated observability path.

## Deterministic core behavior

`IdentityVerificationService` demonstrates the boundary. It collects an observation but returns the deterministic fictional vendor result. ML timeout cannot turn vendor success into failure. Integration-failure prediction uses prediction-time fields **before** the vendor request; the actual vendor outcome is logged later as a separate observation. It is never fed backward into the already-issued prediction.

```text
ALL ML ENDPOINTS UNAVAILABLE
        │
        ▼
core Harbor workflow continues
        │
        ▼
normal deterministic telemetry/logging continues
        │
        ▼
ML observation reports unavailable signals
```

The critical test runs the same vendor outcome once with every model available and once with every model unavailable. The `VerificationResult` is identical; only observability differs. This is dependency inversion:

```text
HIGH-LEVEL APPLICATION
       │
       ▼
MachineLearningGateway
       ▲
       │
HTTP adapter
```

Controllers delegate rather than orchestrating three calls, parsing JSON, handling timeout, calling a vendor, and deciding a response. The Python service is ordinary infrastructure.

## Testing the boundary

`FakeMachineLearningGateway` accepts exact DTOs or endpoint-specific exceptions. Application tests therefore express partial degradation without generic arrays. Adapter tests use Guzzle `MockHandler`, assert all three exact routes, inspect the serialized telemetry body, and exercise response validation without a server. The telemetry serialization assertion is a small, committed cross-language contract check against `CapstoneTelemetryRequest`; the Python service tests independently protect the same schema.

Run the deterministic laboratory:

```bash
cd php
composer install
composer test
php examples/chapter_31_banking_application_integration.php
```

It prints all capabilities, an incident-only outage, and a complete outage, while the core workflow continues each time. To exercise the real service optionally, first train the Chapter 30 artifacts, then:

```bash
# terminal 1, repository root
uvicorn harbor_ml.service.app:create_app --factory --host 127.0.0.1 --port 8000

# terminal 2
# Configure MlClientConfig('http://127.0.0.1:8000') in a local client harness.
```

The mocked laboratory remains the default because it is deterministic and needs no persistent server.

## Security checklist

- Serialize only approved telemetry fields; never serialize a member domain object.
- Send no credentials or member data in prediction payloads.
- Obtain the base URL from deployment configuration and use protected service-to-service transport in production.
- Avoid full request/response bodies in logs.
- Treat downstream JSON as untrusted and validate it.
- Retain model versions as safe internal observability metadata.
- Keep authentication, authorization, correctness, eligibility, pricing, access, and idempotency deterministic.

## Exercises

### Exercise 1 — Which DTO?

Match `anomaly_score` to `AnomalyPrediction`, `failure_probability` to `IntegrationFailurePrediction`, and incident probabilities to `IncidentPrediction`. Why would a generic score erase units, range, taxonomy, threshold, and uncertainty?

### Exercise 2 — Partial outage

Anomaly and request-failure work, but incident returns `503`. Harbor should preserve both successes, mark incident unavailable, log the endpoint failure, report `partially_available`, and continue its deterministic workflow.

### Exercise 3 — Semantic validation

```json
{"predicted_class":"vendor_degradation","probabilities":{"vendor_degradation":0.30,"database_pressure":0.60}}
```

The probabilities sum to only `0.90`, omit canonical classes, and contradict the predicted class because database pressure has the highest value. A complete response also needs metadata and uncertainty fields.

### Exercise 4 — Model versions

Why retain separate versions? Consider independent deployments, rollback, incident reconstruction, and comparing the exact artifacts that produced each observation.

### Exercise 5 — Business independence

Should identity verification fail because the anomaly model timed out? **No.** The model is advisory and unrelated to deterministic vendor outcome correctness.

### Exercise 6 — Parallel requests

What wall-clock latency can concurrency save? What does it add in cancellation, exception aggregation, worker/resource pressure, and debugging?

### Coding exercise — capability status

The solution includes `MlCapabilityStatus` values `all_available`, `partially_available`, and `unavailable`. Extend its tests to cover every combination. Confirm it appears in observation logging but never changes deterministic business results. Explain why capability availability differs from system health.

## Key takeaways

1. Depend on a stable ML gateway, not raw HTTP.
2. Preserve each capability's typed semantics.
3. Unavailable is a first-class state, never a fake zero.
4. Preserve successes during partial degradation.
5. Keep model versions independently observable.
6. Transport failure and contract violation require different investigations.
7. Strict timeouts contain advisory latency.
8. Fakes, mocks, and dependency injection make the language boundary testable.
9. Core banking behavior remains deterministic when ML disappears.
10. ML integration is distributed-systems engineering with unusually important semantic discipline.

## What comes next: Chapter 32 — Building the Engineering Dashboard

Chapter 31 produces `MlObservation`: anomaly signal, incident probabilities, request-failure prediction, independent versions, and availability. Chapter 32 will combine—not replace—application, database, and vendor telemetry with that observation:

```text
APPLICATION TELEMETRY
DATABASE TELEMETRY
VENDOR TELEMETRY
       +
ML OBSERVATION
       +
MODEL AVAILABILITY
       +
MODEL VERSIONS
       │
       ▼
FINAL CAPSTONE DASHBOARD
```

It addresses current state and history, direct observations versus suggestions, ambiguity, availability, versions, explanations, investigation guidance, stale predictions, and incident playback without claiming root-cause certainty.

[Previous: Chapter 30 — Building the ML Service](chapter-30-building-the-ml-service.md) · [Back to Part VII](README.md) · [Complete contents](../../CONTENTS.md) · [Next: Chapter 32 — Building the Engineering Dashboard](chapter-32-building-the-engineering-dashboard.md)
