<?php
declare(strict_types=1);
namespace Harbor\Ml;
final readonly class MlClientConfig { public function __construct(public string $baseUrl, public float $connectTimeoutSeconds = 0.2, public float $requestTimeoutSeconds = 0.5) {} }
