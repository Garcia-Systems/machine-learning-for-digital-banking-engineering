<?php
declare(strict_types=1);
namespace Harbor\Application;
interface VerificationVendor { public function verify(string $verificationReference): VerificationResult; }
