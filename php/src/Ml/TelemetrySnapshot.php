<?php

declare(strict_types=1);

namespace Harbor\Ml;

final readonly class TelemetrySnapshot
{
    public function __construct(
        public float $apiLatencyMs, public float $errorRate, public int $dbConnections,
        public int $queueDepth, public float $vendorLatencyMs, public float $requestsPerMinute,
        public int $retryCount,
    ) {
    }

    /** @return array<string, float|int> */
    public function toApiPayload(): array
    {
        return ['api_latency_ms' => $this->apiLatencyMs, 'error_rate' => $this->errorRate,
            'db_connections' => $this->dbConnections, 'queue_depth' => $this->queueDepth,
            'vendor_latency_ms' => $this->vendorLatencyMs, 'requests_per_minute' => $this->requestsPerMinute,
            'retry_count' => $this->retryCount];
    }
}
