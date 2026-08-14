<?php
declare(strict_types=1);
namespace Harbor\Ml;
use GuzzleHttp\ClientInterface;
use GuzzleHttp\Exception\GuzzleException;
use JsonException;
final readonly class HttpMachineLearningGateway implements MachineLearningGateway
{
    public function __construct(private ClientInterface $client, private MlClientConfig $config) {}
    public function scoreTelemetryAnomaly(TelemetrySnapshot $telemetry): AnomalyPrediction { return AnomalyPrediction::fromApiPayload($this->postJson('/api/v1/score/telemetry-anomaly',$telemetry->toApiPayload())); }
    public function predictIncident(TelemetrySnapshot $telemetry): IncidentPrediction { return IncidentPrediction::fromApiPayload($this->postJson('/api/v1/predict/incident',$telemetry->toApiPayload())); }
    public function predictIntegrationFailure(IntegrationFailureRequest $request): IntegrationFailurePrediction { return IntegrationFailurePrediction::fromApiPayload($this->postJson('/api/v1/predict/integration-failure',$request->toApiPayload())); }
    /** @param array<string,mixed> $payload @return array<string,mixed> */
    private function postJson(string $path,array $payload): array
    {
        try { $response=$this->client->request('POST',rtrim($this->config->baseUrl,'/').$path,['json'=>$payload,'connect_timeout'=>$this->config->connectTimeoutSeconds,'timeout'=>$this->config->requestTimeoutSeconds,'http_errors'=>false]); }
        catch (GuzzleException $e) { throw new MlPredictionUnavailable('ML service could not be reached',0,$e); }
        $status=$response->getStatusCode();
        if ($status!==200) { if ($status===422) throw new MlContractViolation('ML service rejected the v1 request contract'); throw new MlPredictionUnavailable("ML service returned HTTP {$status}"); }
        try { $decoded=json_decode((string)$response->getBody(),true,512,JSON_THROW_ON_ERROR); } catch (JsonException $e) { throw new MlContractViolation('ML response is invalid JSON',0,$e); }
        if (!is_array($decoded)) throw new MlContractViolation('ML response must be an object');
        return $decoded;
    }
}
