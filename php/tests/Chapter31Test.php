<?php
declare(strict_types=1);
namespace Harbor\Tests;
use GuzzleHttp\Client;
use GuzzleHttp\Handler\MockHandler;
use GuzzleHttp\HandlerStack;
use GuzzleHttp\Middleware;
use GuzzleHttp\Psr7\Response;
use Harbor\Application\{IdentityVerificationService,MlObservationService,VerificationResult,VerificationVendor};
use Harbor\Ml\{AnomalyPrediction,FakeMachineLearningGateway,HttpMachineLearningGateway,IncidentClass,IncidentPrediction,IntegrationFailurePrediction,IntegrationFailureRequest,MlCapabilityStatus,MlClientConfig,MlContractViolation,MlPredictionUnavailable,TelemetrySnapshot};
use PHPUnit\Framework\TestCase;
use Psr\Log\AbstractLogger;

final class Chapter31Test extends TestCase
{
    public function testTelemetryUsesExactPythonWireContract(): void { self::assertSame(['api_latency_ms'=>120.0,'error_rate'=>0.02,'db_connections'=>12,'queue_depth'=>3,'vendor_latency_ms'=>240.0,'requests_per_minute'=>900.0,'retry_count'=>1],self::telemetry()->toApiPayload()); }
    public function testHttpGatewayUsesAllExactRoutesAndPreservesVersions(): void
    {
        $history=[];$mock=new MockHandler([new Response(200,[],json_encode(self::anomalyPayload(),JSON_THROW_ON_ERROR)),new Response(200,[],json_encode(self::incidentPayload(),JSON_THROW_ON_ERROR)),new Response(200,[],json_encode(self::integrationPayload(),JSON_THROW_ON_ERROR))]);
        $stack=HandlerStack::create($mock);$stack->push(Middleware::history($history));$gateway=new HttpMachineLearningGateway(new Client(['handler'=>$stack]),new MlClientConfig('http://ml.test'));
        self::assertSame('anomaly-v1',$gateway->scoreTelemetryAnomaly(self::telemetry())->modelVersion);self::assertSame('incident-v2',$gateway->predictIncident(self::telemetry())->modelVersion);self::assertSame('integration-v3',$gateway->predictIntegrationFailure(self::request())->modelVersion);
        self::assertSame(['/api/v1/score/telemetry-anomaly','/api/v1/predict/incident','/api/v1/predict/integration-failure'],array_map(fn($x)=>$x['request']->getUri()->getPath(),$history));
        self::assertSame(array_keys(self::telemetry()->toApiPayload()),array_keys(json_decode((string)$history[0]['request']->getBody(),true,512,JSON_THROW_ON_ERROR)));
    }
    /** @dataProvider invalidIncidents */
    public function testIncidentSemanticValidation(array $change): void { $this->expectException(MlContractViolation::class);IncidentPrediction::fromApiPayload(array_replace(self::incidentPayload(),$change)); }
    public static function invalidIncidents(): array { return ['unknown taxonomy'=>[['predicted_class'=>'other']],'bad sum'=>[['probabilities'=>['normal'=>.4,'vendor_degradation'=>.4]]],'malformed map'=>[['probabilities'=>['normal'=>'high']]],'top mismatch'=>[['top_probability'=>.6]],'class mismatch'=>[['predicted_class'=>'normal']]]; }
    public function testAnomalyScoreIsFiniteButNotRestrictedToProbabilityRange(): void { $p=AnomalyPrediction::fromApiPayload(array_replace(self::anomalyPayload(),['anomaly_score'=>-2.4]));self::assertSame(-2.4,$p->anomalyScore); }
    public function testPartialAndCompleteOutageHaveNoFakeZeros(): void
    {
        $logger=new MemoryLogger();$partial=(new MlObservationService(new FakeMachineLearningGateway(self::anomaly(),new MlPredictionUnavailable('down'),self::integration()),$logger))->observe(self::telemetry(),self::request());
        self::assertSame(MlCapabilityStatus::PartiallyAvailable,$partial->capabilityStatus());self::assertNull($partial->incident);self::assertSame(-0.42,$partial->anomaly?->anomalyScore);
        $down=new MlPredictionUnavailable('down');$none=(new MlObservationService(new FakeMachineLearningGateway($down,$down,$down),$logger))->observe(self::telemetry(),self::request());self::assertSame(MlCapabilityStatus::Unavailable,$none->capabilityStatus());self::assertNull($none->anomaly);self::assertNull($none->integrationFailure);
        self::assertTrue((bool)array_filter($logger->records,fn($r)=>$r[1]==='harbor.ml_observation' && $r[2]['ml_capability_status']==='unavailable'));
    }
    public function testCoreResultIsIdenticalWhenEveryMlSignalFails(): void
    {
        $vendor=new class implements VerificationVendor { public function verify(string $verificationReference):VerificationResult{return new VerificationResult(true,'vendor_verified');}};$logger=new MemoryLogger();
        $available=new IdentityVerificationService($vendor,new MlObservationService(new FakeMachineLearningGateway(self::anomaly(),self::incident(),self::integration()),$logger));$e=new MlPredictionUnavailable('outage');$unavailable=new IdentityVerificationService($vendor,new MlObservationService(new FakeMachineLearningGateway($e,$e,$e),$logger));
        self::assertEquals($available->verify('fictional-reference',self::telemetry(),self::request()),$unavailable->verify('fictional-reference',self::telemetry(),self::request()));
    }
    private static function telemetry():TelemetrySnapshot{return new TelemetrySnapshot(120,.02,12,3,240,900,1);} private static function request():IntegrationFailureRequest{return new IntegrationFailureRequest('FictionalVendor','identity',240,.02,3,1,512,10);}
    private static function anomalyPayload():array{return ['model'=>'anomaly','model_version'=>'anomaly-v1','anomaly_score'=>-.42,'is_anomaly'=>true];} private static function incidentPayload():array{return ['model'=>'incident','model_version'=>'incident-v2','predicted_class'=>'vendor_degradation','probabilities'=>['normal'=>.1,'vendor_degradation'=>.7,'database_pressure'=>.1,'traffic_spike'=>.05,'application_regression'=>.05],'top_probability'=>.7,'second_probability'=>.1,'probability_gap'=>.6,'ambiguous'=>false];} private static function integrationPayload():array{return ['model'=>'integration','model_version'=>'integration-v3','failure_probability'=>.7,'threshold'=>.6,'predicted_failure'=>true];}
    private static function anomaly():AnomalyPrediction{return AnomalyPrediction::fromApiPayload(self::anomalyPayload());} private static function incident():IncidentPrediction{return IncidentPrediction::fromApiPayload(self::incidentPayload());} private static function integration():IntegrationFailurePrediction{return IntegrationFailurePrediction::fromApiPayload(self::integrationPayload());}
}
final class MemoryLogger extends AbstractLogger { public array $records=[];public function log($level,string|\Stringable $message,array $context=[]):void{$this->records[]=[$level,(string)$message,$context];} }
