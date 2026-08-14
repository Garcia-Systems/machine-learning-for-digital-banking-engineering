<?php
declare(strict_types=1);
namespace Harbor\Application;
use Harbor\Ml\{AnomalyPrediction,IncidentPrediction,IntegrationFailurePrediction,IntegrationFailureRequest,MachineLearningGateway,MlObservation,MlPredictionUnavailable,TelemetrySnapshot};
use Psr\Log\LoggerInterface;
final readonly class MlObservationService
{
    public function __construct(private MachineLearningGateway $gateway,private LoggerInterface $logger) {}
    public function observe(TelemetrySnapshot $telemetry,?IntegrationFailureRequest $request=null): MlObservation
    {
        $anomaly=$this->attempt(fn():AnomalyPrediction=>$this->gateway->scoreTelemetryAnomaly($telemetry),'anomaly');
        $incident=$this->attempt(fn():IncidentPrediction=>$this->gateway->predictIncident($telemetry),'incident');
        $integration=$request===null?null:$this->attempt(fn():IntegrationFailurePrediction=>$this->gateway->predictIntegrationFailure($request),'integration_failure');
        $observation=new MlObservation($anomaly,$incident,$integration);
        $this->logger->info('harbor.ml_observation',['ml_capability_status'=>$observation->capabilityStatus()->value,'anomaly_available'=>$anomaly!==null,'anomaly_model_version'=>$anomaly?->modelVersion,'anomaly_score'=>$anomaly?->anomalyScore,'is_anomaly'=>$anomaly?->isAnomaly,'incident_available'=>$incident!==null,'incident_model_version'=>$incident?->modelVersion,'predicted_class'=>$incident?->predictedClass->value,'incident_ambiguous'=>$incident?->ambiguous,'integration_failure_available'=>$integration!==null,'integration_model_version'=>$integration?->modelVersion,'failure_probability'=>$integration?->failureProbability,'predicted_failure'=>$integration?->predictedFailure]);
        return $observation;
    }
    private function attempt(callable $call,string $signal): mixed { try{return $call();}catch(MlPredictionUnavailable $e){$this->logger->warning('harbor.ml_signal_unavailable',['signal'=>$signal,'reason'=>$e->getMessage(),'contract_violation'=>$e instanceof \Harbor\Ml\MlContractViolation]);return null;} }
}
