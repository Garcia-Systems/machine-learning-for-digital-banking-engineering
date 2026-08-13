<?php

declare(strict_types=1);

namespace Harbor\Ml;

interface IntegrationFailurePredictor
{
    public function predict(IntegrationFailureRequest $request): IntegrationFailurePrediction;
}
