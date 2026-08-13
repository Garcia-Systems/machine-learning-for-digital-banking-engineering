<?php

declare(strict_types=1);

namespace Harbor\Ml;

final readonly class FakeIntegrationFailurePredictor implements IntegrationFailurePredictor
{
    public function __construct(private IntegrationFailurePrediction $prediction)
    {
    }

    public function predict(IntegrationFailureRequest $request): IntegrationFailurePrediction
    {
        return $this->prediction;
    }
}
