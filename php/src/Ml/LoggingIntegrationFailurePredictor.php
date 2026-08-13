<?php

declare(strict_types=1);

namespace Harbor\Ml;

use Psr\Log\LoggerInterface;

final readonly class LoggingIntegrationFailurePredictor implements IntegrationFailurePredictor
{
    public function __construct(private IntegrationFailurePredictor $inner, private LoggerInterface $logger)
    {
    }

    public function predict(IntegrationFailureRequest $request): IntegrationFailurePrediction
    {
        $started = hrtime(true);
        try {
            $prediction = $this->inner->predict($request);
            $this->logger->info('ml.integration_failure_client', [
                'model_version' => $prediction->modelVersion,
                'latency_ms' => (hrtime(true) - $started) / 1_000_000,
            ]);
            return $prediction;
        } catch (MlPredictionUnavailable $error) {
            $this->logger->warning('ml.integration_failure_client_unavailable', [
                'latency_ms' => (hrtime(true) - $started) / 1_000_000,
            ]);
            throw $error;
        }
    }
}
