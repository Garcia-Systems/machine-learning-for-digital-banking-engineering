<?php
declare(strict_types=1);
require dirname(__DIR__).'/vendor/autoload.php';
use Harbor\Application\MlObservationService;
use Harbor\Ml\{AnomalyPrediction,FakeMachineLearningGateway,IncidentClass,IncidentPrediction,IntegrationFailurePrediction,IntegrationFailureRequest,MlObservation,MlPredictionUnavailable,TelemetrySnapshot};
use Psr\Log\NullLogger;
$telemetry=new TelemetrySnapshot(180,.04,34,18,720,1250,2);
$request=new IntegrationFailureRequest('FictionalIdentityVendor','verify',720,.04,18,2,640,14);
$anomaly=new AnomalyPrediction('harbor-capstone-anomaly','anomaly-abc123',-.61,true);
$incident=new IncidentPrediction('harbor-capstone-incident','incident-def456',IncidentClass::VendorDegradation,['normal'=>.08,'vendor_degradation'=>.72,'database_pressure'=>.08,'traffic_spike'=>.06,'application_regression'=>.06],.72,.08,.64,false);
$failure=new IntegrationFailurePrediction('harbor-integration-failure','integration-ghi789',.67,.60,true);
$show=function(string $heading,MlObservation $o):void { echo "\n{$heading}\n";foreach(['Anomaly'=>$o->anomalyStatus(),'Incident'=>$o->incidentStatus(),'Integration failure'=>$o->integrationFailureStatus()] as $name=>$status)echo "{$name}: {$status->value}\n";if($o->anomaly)echo "score: {$o->anomaly->anomalyScore}; anomaly: ".($o->anomaly->isAnomaly?'yes':'no')."; model version: {$o->anomaly->modelVersion}\n";if($o->incident)echo "top class: {$o->incident->predictedClass->value}; ambiguous: ".($o->incident->ambiguous?'yes':'no')."; model version: {$o->incident->modelVersion}\n";if($o->integrationFailure)echo "probability: {$o->integrationFailure->failureProbability}; predicted failure: ".($o->integrationFailure->predictedFailure?'yes':'no')."; model version: {$o->integrationFailure->modelVersion}\n";echo "ML capability: {$o->capabilityStatus()->value}\nCore verification workflow: continued normally.\n";};
echo "Harbor Federal Credit Union\nBanking Application ML Integration Laboratory\n";
$show('All models available',(new MlObservationService(new FakeMachineLearningGateway($anomaly,$incident,$failure),new NullLogger()))->observe($telemetry,$request));
$show('Simulating incident classifier outage',(new MlObservationService(new FakeMachineLearningGateway($anomaly,new MlPredictionUnavailable('incident unavailable'),$failure),new NullLogger()))->observe($telemetry,$request));
$outage=new MlPredictionUnavailable('service unavailable');
$show('Simulating complete ML outage',(new MlObservationService(new FakeMachineLearningGateway($outage,$outage,$outage),new NullLogger()))->observe($telemetry,$request));
