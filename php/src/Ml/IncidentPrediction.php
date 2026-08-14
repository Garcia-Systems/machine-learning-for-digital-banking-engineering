<?php

declare(strict_types=1);

namespace Harbor\Ml;

final readonly class IncidentPrediction
{
    /** @param array<string, float> $probabilities */
    public function __construct(public string $model, public string $modelVersion, public IncidentClass $predictedClass, public array $probabilities, public float $topProbability, public float $secondProbability, public float $probabilityGap, public bool $ambiguous) {}

    /** @param array<string, mixed> $payload */
    public static function fromApiPayload(array $payload): self
    {
        foreach (['model','model_version','predicted_class','probabilities','top_probability','second_probability','probability_gap','ambiguous'] as $field) if (!array_key_exists($field, $payload)) throw new MlContractViolation("Incident response missing {$field}");
        if (!is_string($payload['model']) || trim($payload['model']) === '' || !is_string($payload['model_version']) || trim($payload['model_version']) === '' || !is_string($payload['predicted_class'])) throw new MlContractViolation('Invalid incident metadata');
        $class = IncidentClass::tryFrom($payload['predicted_class']);
        if ($class === null || !is_array($payload['probabilities']) || $payload['probabilities'] === []) throw new MlContractViolation('Invalid incident taxonomy or probability map');
        $probabilities = [];
        foreach ($payload['probabilities'] as $name => $value) {
            if (!is_string($name) || IncidentClass::tryFrom($name) === null || is_bool($value) || !is_numeric($value) || !is_finite((float)$value) || $value < 0 || $value > 1) throw new MlContractViolation('Invalid incident probability');
            $probabilities[$name] = (float)$value;
        }
        if (abs(array_sum($probabilities) - 1.0) > 1e-6 || !isset($probabilities[$class->value])) throw new MlContractViolation('Incident probabilities must sum to one and contain predicted class');
        foreach (['top_probability','second_probability','probability_gap'] as $field) if (is_bool($payload[$field]) || !is_numeric($payload[$field]) || !is_finite((float)$payload[$field])) throw new MlContractViolation("Invalid {$field}");
        if (!is_bool($payload['ambiguous'])) throw new MlContractViolation('ambiguous must be boolean');
        $sorted = array_values($probabilities); rsort($sorted, SORT_NUMERIC);
        $top=(float)$payload['top_probability']; $second=(float)$payload['second_probability']; $gap=(float)$payload['probability_gap'];
        if (abs($top-$sorted[0]) > 1e-6 || abs($second-($sorted[1] ?? 0.0)) > 1e-6 || abs($gap-($top-$second)) > 1e-6 || abs($probabilities[$class->value]-$top)>1e-6 || $payload['ambiguous'] !== ($gap < 0.10)) throw new MlContractViolation('Incident ranking or ambiguity fields are inconsistent');
        return new self($payload['model'],$payload['model_version'],$class,$probabilities,$top,$second,$gap,$payload['ambiguous']);
    }
}
