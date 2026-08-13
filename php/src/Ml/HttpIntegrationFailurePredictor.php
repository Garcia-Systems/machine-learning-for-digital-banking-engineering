<?php

declare(strict_types=1);

namespace Harbor\Ml;

use GuzzleHttp\ClientInterface;
use GuzzleHttp\Exception\GuzzleException;
use JsonException;

final readonly class HttpIntegrationFailurePredictor implements IntegrationFailurePredictor
{
    public function __construct(
        private ClientInterface $client,
        private string $baseUrl,
        private float $connectTimeoutSeconds = 0.2,
        private float $timeoutSeconds = 0.5,
    ) {
    }

    public function predict(IntegrationFailureRequest $request): IntegrationFailurePrediction
    {
        try {
            $response = $this->client->request('POST', rtrim($this->baseUrl, '/') . '/api/v1/predict/integration-failure', [
                'json' => $request->toApiPayload(),
                'connect_timeout' => $this->connectTimeoutSeconds,
                'timeout' => $this->timeoutSeconds,
                'http_errors' => false,
            ]);
        } catch (GuzzleException $error) {
            throw new MlPredictionUnavailable('ML prediction service could not be reached', 0, $error);
        }

        $status = $response->getStatusCode();
        if ($status !== 200) {
            $kind = $status === 422 ? 'rejected the v1 request contract' : "returned HTTP {$status}";
            throw new MlPredictionUnavailable("ML prediction service {$kind}");
        }
        try {
            $payload = json_decode((string) $response->getBody(), true, 512, JSON_THROW_ON_ERROR);
        } catch (JsonException $error) {
            throw new MlPredictionUnavailable('ML response is not valid JSON', 0, $error);
        }
        if (!is_array($payload)) {
            throw new MlPredictionUnavailable('ML response must be a JSON object');
        }

        return IntegrationFailurePrediction::fromApiPayload($payload);
    }
}
