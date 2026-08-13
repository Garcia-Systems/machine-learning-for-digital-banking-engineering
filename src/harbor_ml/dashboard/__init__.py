"""Server-rendered, ML-assisted engineering dashboard for Chapter 20."""

from .models import DashboardSnapshot, MLAvailability, TelemetrySnapshot
from .service import (
    DashboardService,
    INVESTIGATION_GUIDANCE,
    build_teaching_service,
    calculate_severity,
    classify_availability,
    is_ambiguous,
)

__all__ = [
    "DashboardService",
    "DashboardSnapshot",
    "INVESTIGATION_GUIDANCE",
    "MLAvailability",
    "TelemetrySnapshot",
    "build_teaching_service",
    "calculate_severity",
    "classify_availability",
    "is_ambiguous",
]
