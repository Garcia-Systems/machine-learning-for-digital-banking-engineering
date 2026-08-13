<?php

declare(strict_types=1);

namespace Harbor\Ml;

final class UnavailableIntegrationFailurePredictor implements IntegrationFailurePredictor
{
    public function predict(IntegrationFailureRequest $request): IntegrationFailurePrediction
    {
        throw new MlPredictionUnavailable('ML prediction is explicitly unavailable');
    }
}
