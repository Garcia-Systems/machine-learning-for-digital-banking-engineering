<?php

declare(strict_types=1);

namespace Harbor\Ml;

final readonly class IntegrationFailureRequest
{
    public function __construct(
        public string $vendor,
        public string $endpoint,
        public float $recentVendorLatencyMs,
        public float $recentVendorErrorRate,
        public int $queueDepth,
        public int $retryCount,
        public int $requestSizeBytes,
        public int $hourOfDay,
    ) {
    }

    /** @return array<string, string|float|int> */
    public function toApiPayload(): array
    {
        return [
            'vendor' => $this->vendor,
            'endpoint' => $this->endpoint,
            'recent_vendor_latency_ms' => $this->recentVendorLatencyMs,
            'recent_vendor_error_rate' => $this->recentVendorErrorRate,
            'queue_depth' => $this->queueDepth,
            'retry_count' => $this->retryCount,
            'request_size_bytes' => $this->requestSizeBytes,
            'hour_of_day' => $this->hourOfDay,
        ];
    }
}
