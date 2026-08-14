<?php
declare(strict_types=1);
namespace Harbor\Application;
final readonly class VerificationResult { public function __construct(public bool $verified,public string $code) {} }
