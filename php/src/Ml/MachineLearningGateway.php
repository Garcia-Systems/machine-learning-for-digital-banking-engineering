<?php

declare(strict_types=1);

namespace Harbor\Ml;

interface MachineLearningGateway
{
    public function scoreTelemetryAnomaly(TelemetrySnapshot $telemetry): AnomalyPrediction;
    public function predictIncident(TelemetrySnapshot $telemetry): IncidentPrediction;
    public function predictIntegrationFailure(IntegrationFailureRequest $request): IntegrationFailurePrediction;
}
