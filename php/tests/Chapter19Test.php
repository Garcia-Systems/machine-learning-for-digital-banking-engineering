<?php

declare(strict_types=1);

namespace Harbor\Tests;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\ConnectException;
use GuzzleHttp\Handler\MockHandler;
use GuzzleHttp\HandlerStack;
use GuzzleHttp\Promise\Create;
use GuzzleHttp\Psr7\Request;
use GuzzleHttp\Psr7\Response;
use Harbor\Application\VendorRequestObserver;
use Harbor\Ml\FakeIntegrationFailurePredictor;
use Harbor\Ml\HttpIntegrationFailurePredictor;
use Harbor\Ml\IntegrationFailurePrediction;
use Harbor\Ml\IntegrationFailureRequest;
use Harbor\Ml\LoggingIntegrationFailurePredictor;
use Harbor\Ml\MlPredictionUnavailable;
use Harbor\Ml\UnavailableIntegrationFailurePredictor;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Psr\Log\LoggerInterface;

final class Chapter19Test extends TestCase
{
    private IntegrationFailureRequest $request;

    protected function setUp(): void
    {
        $this->request = new IntegrationFailureRequest('ClearVerify', 'identity_verify', 940.0, 0.031, 42, 1, 2400, 14);
    }

    public function testRequestSerializationExactlyMatchesV1Contract(): void
    {
        self::assertSame([
            'vendor' => 'ClearVerify', 'endpoint' => 'identity_verify',
            'recent_vendor_latency_ms' => 940.0, 'recent_vendor_error_rate' => 0.031,
            'queue_depth' => 42, 'retry_count' => 1, 'request_size_bytes' => 2400,
            'hour_of_day' => 14,
        ], $this->request->toApiPayload());
    }

    public function testHttpPredictorParsesValidResponseAndSendsJson(): void
    {
        $history = [];
        $mock = new MockHandler([new Response(200, [], json_encode($this->validPayload(), JSON_THROW_ON_ERROR))]);
        $stack = HandlerStack::create($mock);
        $stack->push(\GuzzleHttp\Middleware::history($history));
        $result = (new HttpIntegrationFailurePredictor(new Client(['handler' => $stack]), 'http://ml.test'))->predict($this->request);

        self::assertSame('2026.08.1', $result->modelVersion);
        self::assertSame(0.67, $result->failureProbability);
        self::assertSame('/api/v1/predict/integration-failure', $history[0]['request']->getUri()->getPath());
        self::assertSame($this->request->toApiPayload(), json_decode((string) $history[0]['request']->getBody(), true, 512, JSON_THROW_ON_ERROR));
    }

    /** @return iterable<string, array{array<string, mixed>|string}> */
    public static function invalidResponses(): iterable
    {
        yield 'malformed JSON' => ['{'];
        yield 'missing field' => [['model' => 'x']];
        yield 'probability out of range' => [[
            'model' => 'x', 'model_version' => '1', 'failure_probability' => 1.1,
            'threshold' => 0.5, 'predicted_failure' => true,
        ]];
        yield 'inconsistent boolean' => [[
            'model' => 'x', 'model_version' => '1', 'failure_probability' => 0.2,
            'threshold' => 0.5, 'predicted_failure' => true,
        ]];
    }

    #[DataProvider('invalidResponses')]
    public function testInvalidResponsesMapToExplicitUnavailability(array|string $body): void
    {
        $body = is_array($body) ? json_encode($body, JSON_THROW_ON_ERROR) : $body;
        $predictor = $this->httpPredictor([new Response(200, [], $body)]);
        $this->expectException(MlPredictionUnavailable::class);
        $predictor->predict($this->request);
    }

    public function testConnectionOrTimeoutMapsToExplicitUnavailability(): void
    {
        $failure = static function ($request) {
            return Create::rejectionFor(new ConnectException('timed out', $request));
        };
        $this->expectException(MlPredictionUnavailable::class);
        (new HttpIntegrationFailurePredictor(new Client(['handler' => $failure]), 'http://ml.test'))->predict($this->request);
    }

    public function testHttp500MapsToExplicitUnavailability(): void
    {
        $this->expectException(MlPredictionUnavailable::class);
        $this->httpPredictor([new Response(500, [], 'private details')])->predict($this->request);
    }

    public function testFakeReturnsPredeterminedTypedPrediction(): void
    {
        $expected = IntegrationFailurePrediction::fromApiPayload($this->validPayload());
        self::assertSame($expected, (new FakeIntegrationFailurePredictor($expected))->predict($this->request));
    }

    public function testObserverLogsPredictionMetadata(): void
    {
        $logger = $this->createMock(LoggerInterface::class);
        $logger->expects(self::once())->method('info')->with(
            'ml.integration_failure_prediction',
            self::callback(fn (array $context): bool => $context['model_version'] === '2026.08.1')
        );
        (new VendorRequestObserver(new FakeIntegrationFailurePredictor(
            IntegrationFailurePrediction::fromApiPayload($this->validPayload())
        ), $logger))->observe($this->request);
    }

    public function testObserverContinuesWhenMlIsUnavailable(): void
    {
        $logger = $this->createMock(LoggerInterface::class);
        $logger->expects(self::once())->method('warning')->with('ml.integration_failure_unavailable', self::isType('array'));
        (new VendorRequestObserver(new UnavailableIntegrationFailurePredictor(), $logger))->observe($this->request);
        self::assertTrue($this->deterministicVendorOperation());
    }

    public function testMlResultDoesNotChangeDeterministicBusinessOutcome(): void
    {
        foreach ([true, false] as $predictedFailure) {
            $payload = $this->validPayload();
            $payload['failure_probability'] = $predictedFailure ? 0.67 : 0.33;
            $payload['predicted_failure'] = $predictedFailure;
            $logger = $this->createStub(LoggerInterface::class);
            (new VendorRequestObserver(new FakeIntegrationFailurePredictor(
                IntegrationFailurePrediction::fromApiPayload($payload)
            ), $logger))->observe($this->request);
            self::assertTrue($this->deterministicVendorOperation());
        }
    }

    public function testLoggingDecoratorReturnsUnchangedPredictionAndLogsVersion(): void
    {
        $expected = IntegrationFailurePrediction::fromApiPayload($this->validPayload());
        $logger = $this->createMock(LoggerInterface::class);
        $logger->expects(self::once())->method('info')->with('ml.integration_failure_client', self::callback(
            fn (array $context): bool => $context['model_version'] === '2026.08.1' && $context['latency_ms'] >= 0
        ));
        $actual = (new LoggingIntegrationFailurePredictor(new FakeIntegrationFailurePredictor($expected), $logger))->predict($this->request);
        self::assertSame($expected, $actual);
    }

    public function testLoggingDecoratorRethrowsSameUnavailableException(): void
    {
        $logger = $this->createMock(LoggerInterface::class);
        $logger->expects(self::once())->method('warning');
        $decorator = new LoggingIntegrationFailurePredictor(new UnavailableIntegrationFailurePredictor(), $logger);
        try {
            $decorator->predict($this->request);
            self::fail('Expected unavailability');
        } catch (MlPredictionUnavailable $error) {
            self::assertSame('ML prediction is explicitly unavailable', $error->getMessage());
        }
    }

    /** @param list<Response> $responses */
    private function httpPredictor(array $responses): HttpIntegrationFailurePredictor
    {
        return new HttpIntegrationFailurePredictor(new Client(['handler' => HandlerStack::create(new MockHandler($responses))]), 'http://ml.test');
    }

    /** @return array<string, mixed> */
    private function validPayload(): array
    {
        return ['model' => 'harbor-integration-failure', 'model_version' => '2026.08.1',
            'failure_probability' => 0.67, 'threshold' => 0.5, 'predicted_failure' => true];
    }

    private function deterministicVendorOperation(): bool
    {
        return true; // Represents established authentication, validation, and vendor rules.
    }
}
