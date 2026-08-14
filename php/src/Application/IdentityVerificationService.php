<?php
declare(strict_types=1);
namespace Harbor\Application;
use Harbor\Ml\{IntegrationFailureRequest,TelemetrySnapshot};
final readonly class IdentityVerificationService
{
    public function __construct(private VerificationVendor $vendor,private MlObservationService $ml) {}
    public function verify(string $reference,TelemetrySnapshot $telemetry,IntegrationFailureRequest $request): VerificationResult
    {
        $this->ml->observe($telemetry,$request); // advisory observation cannot decide the outcome
        return $this->vendor->verify($reference);
    }
}
