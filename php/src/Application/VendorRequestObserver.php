<?php

declare(strict_types=1);

namespace Harbor\Application;

use Harbor\Ml\IntegrationFailurePredictor;
use Harbor\Ml\IntegrationFailureRequest;
use Harbor\Ml\MlPredictionUnavailable;
use Psr\Log\LoggerInterface;

final readonly class VendorRequestObserver
{
    public function __construct(private IntegrationFailurePredictor $predictor, private LoggerInterface $logger)
    {
    }

    public function observe(IntegrationFailureRequest $request): void
    {
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
    }
}
