<?php

declare(strict_types=1);

namespace Harbor\Ml;

final readonly class AnomalyPrediction
{
    public function __construct(public string $model, public string $modelVersion, public float $anomalyScore, public bool $isAnomaly) {}

    /** @param array<string, mixed> $payload */
    public static function fromApiPayload(array $payload): self
    {
        foreach (['model', 'model_version', 'anomaly_score', 'is_anomaly'] as $field) {
            if (!array_key_exists($field, $payload)) throw new MlContractViolation("Anomaly response missing {$field}");
        }
        if (!is_string($payload['model']) || trim($payload['model']) === '' || !is_string($payload['model_version']) || trim($payload['model_version']) === '') throw new MlContractViolation('Invalid anomaly model metadata');
        if (is_bool($payload['anomaly_score']) || !is_numeric($payload['anomaly_score']) || !is_finite((float) $payload['anomaly_score'])) throw new MlContractViolation('Anomaly score must be finite');
        if (!is_bool($payload['is_anomaly'])) throw new MlContractViolation('is_anomaly must be boolean');
        return new self($payload['model'], $payload['model_version'], (float) $payload['anomaly_score'], $payload['is_anomaly']);
    }
}
