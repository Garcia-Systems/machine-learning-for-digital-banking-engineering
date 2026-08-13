<?php

declare(strict_types=1);

namespace Harbor\Ml;

final readonly class IntegrationFailurePrediction
{
    public function __construct(
        public string $model,
        public string $modelVersion,
        public float $failureProbability,
        public float $threshold,
        public bool $predictedFailure,
    ) {
    }

    /** @param array<string, mixed> $payload */
    public static function fromApiPayload(array $payload): self
    {
        foreach (['model', 'model_version', 'failure_probability', 'threshold', 'predicted_failure'] as $field) {
            if (!array_key_exists($field, $payload)) {
                throw new MlPredictionUnavailable("ML response is missing required field: {$field}");
            }
        }
        if (!is_string($payload['model']) || trim($payload['model']) === ''
            || !is_string($payload['model_version']) || trim($payload['model_version']) === '') {
            throw new MlPredictionUnavailable('ML response has invalid model metadata');
        }
        if (is_bool($payload['failure_probability']) || !is_int($payload['failure_probability']) && !is_float($payload['failure_probability'])
            || is_bool($payload['threshold']) || !is_int($payload['threshold']) && !is_float($payload['threshold'])) {
            throw new MlPredictionUnavailable('ML response probabilities must be numeric');
        }
        $probability = (float) $payload['failure_probability'];
        $threshold = (float) $payload['threshold'];
        if (!is_finite($probability) || $probability < 0.0 || $probability > 1.0
            || !is_finite($threshold) || $threshold < 0.0 || $threshold > 1.0) {
            throw new MlPredictionUnavailable('ML response probability or threshold is out of range');
        }
        if (!is_bool($payload['predicted_failure'])) {
            throw new MlPredictionUnavailable('ML response predicted_failure must be boolean');
        }
        if ($payload['predicted_failure'] !== ($probability >= $threshold)) {
            throw new MlPredictionUnavailable('ML response prediction is inconsistent with its threshold');
        }

        return new self($payload['model'], $payload['model_version'], $probability, $threshold, $payload['predicted_failure']);
    }
}
