<?php

declare(strict_types=1);

namespace Harbor\Ml;

enum IncidentClass: string
{
    case Normal = 'normal';
    case VendorDegradation = 'vendor_degradation';
    case DatabasePressure = 'database_pressure';
    case TrafficSpike = 'traffic_spike';
    case ApplicationRegression = 'application_regression';
}
