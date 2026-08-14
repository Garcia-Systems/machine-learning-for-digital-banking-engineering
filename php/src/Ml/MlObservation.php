<?php
declare(strict_types=1);
namespace Harbor\Ml;
final readonly class MlObservation
{
    public function __construct(public ?AnomalyPrediction $anomaly, public ?IncidentPrediction $incident, public ?IntegrationFailurePrediction $integrationFailure) {}
    public function anomalyStatus(): MlSignalStatus { return $this->anomaly === null ? MlSignalStatus::Unavailable : MlSignalStatus::Available; }
    public function incidentStatus(): MlSignalStatus { return $this->incident === null ? MlSignalStatus::Unavailable : MlSignalStatus::Available; }
    public function integrationFailureStatus(): MlSignalStatus { return $this->integrationFailure === null ? MlSignalStatus::Unavailable : MlSignalStatus::Available; }
    public function capabilityStatus(): MlCapabilityStatus { $n=count(array_filter([$this->anomaly,$this->incident,$this->integrationFailure])); return $n===3?MlCapabilityStatus::AllAvailable:($n===0?MlCapabilityStatus::Unavailable:MlCapabilityStatus::PartiallyAvailable); }
}
