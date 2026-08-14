<?php
declare(strict_types=1);
namespace Harbor\Ml;
final readonly class FakeMachineLearningGateway implements MachineLearningGateway
{
    public function __construct(private AnomalyPrediction|MlPredictionUnavailable $anomaly, private IncidentPrediction|MlPredictionUnavailable $incident, private IntegrationFailurePrediction|MlPredictionUnavailable $integrationFailure) {}
    public function scoreTelemetryAnomaly(TelemetrySnapshot $telemetry): AnomalyPrediction { if($this->anomaly instanceof MlPredictionUnavailable) throw $this->anomaly; return $this->anomaly; }
    public function predictIncident(TelemetrySnapshot $telemetry): IncidentPrediction { if($this->incident instanceof MlPredictionUnavailable) throw $this->incident; return $this->incident; }
    public function predictIntegrationFailure(IntegrationFailureRequest $request): IntegrationFailurePrediction { if($this->integrationFailure instanceof MlPredictionUnavailable) throw $this->integrationFailure; return $this->integrationFailure; }
}
