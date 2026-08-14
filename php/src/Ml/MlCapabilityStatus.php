<?php
declare(strict_types=1);
namespace Harbor\Ml;
enum MlCapabilityStatus:string { case AllAvailable='all_available'; case PartiallyAvailable='partially_available'; case Unavailable='unavailable'; }
