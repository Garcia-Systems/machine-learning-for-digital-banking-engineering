"""Server-rendered, ML-assisted engineering dashboard for Chapter 20."""

from .models import (CapstoneDashboardSnapshot, DashboardSnapshot, MLAvailability,
                     SignalAvailability, TelemetrySnapshot)
from .service import (
    CapstoneDashboardService, DashboardService,
    INVESTIGATION_GUIDANCE,
    build_teaching_service,
    calculate_severity,
    classify_availability,
    build_capstone_dashboard,
    is_ambiguous,
)

__all__ = [
    "DashboardService",
    "CapstoneDashboardService",
    "CapstoneDashboardSnapshot",
    "DashboardSnapshot",
    "INVESTIGATION_GUIDANCE",
    "MLAvailability",
    "SignalAvailability",
    "TelemetrySnapshot",
    "build_teaching_service",
    "build_capstone_dashboard",
    "calculate_severity",
    "classify_availability",
    "is_ambiguous",
]
