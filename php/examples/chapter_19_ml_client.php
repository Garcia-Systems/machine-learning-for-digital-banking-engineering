<?php

declare(strict_types=1);

use GuzzleHttp\Client;
use GuzzleHttp\Handler\MockHandler;
use GuzzleHttp\HandlerStack;
use GuzzleHttp\Psr7\Response;
use Harbor\Ml\HttpIntegrationFailurePredictor;
use Harbor\Ml\IntegrationFailureRequest;
use Harbor\Ml\MlPredictionUnavailable;

require dirname(__DIR__) . '/vendor/autoload.php';

$request = new IntegrationFailureRequest('ClearVerify', 'identity_verify', 940.0, 0.031, 42, 1, 2400, 14);
$success = json_encode(['model' => 'harbor-integration-failure', 'model_version' => '2026.08.1',
    'failure_probability' => 0.67, 'threshold' => 0.5, 'predicted_failure' => true], JSON_THROW_ON_ERROR);
$client = new Client(['handler' => HandlerStack::create(new MockHandler([
    new Response(200, [], $success), new Response(503, [], 'not exposed'),
]))]);
$predictor = new HttpIntegrationFailurePredictor($client, getenv('HARBOR_ML_BASE_URL') ?: 'http://127.0.0.1:8000');

echo "Harbor Federal Credit Union\nPHP ML Integration Laboratory\n\nSending prediction request...\n\n";
$prediction = $predictor->predict($request);
printf("Model:\n%s\n\nVersion:\n%s\n\nFailure probability:\n%.2f\n\nPredicted failure:\n%s\n\nObservability event recorded.\n",
    $prediction->model, $prediction->modelVersion, $prediction->failureProbability, $prediction->predictedFailure ? 'yes' : 'no');

echo "\nNow simulating ML-service timeout...\n\n";
try {
    $predictor->predict($request);
} catch (MlPredictionUnavailable) {
    echo "Prediction unavailable.\n\nCore Harbor workflow:\ncontinues normally.\n";
}
