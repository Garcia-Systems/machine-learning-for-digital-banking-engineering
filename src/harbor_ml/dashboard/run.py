"""Development entry point used by the Chapter 32 startup command."""

from pathlib import Path

from .app import create_dashboard_app
from .service import build_capstone_dashboard

ROOT = Path(__file__).resolve().parents[3]
service, rows = build_capstone_dashboard(ROOT)
timeline = [service.build_snapshot(row) for row in rows]
app = create_dashboard_app(service, timeline[-1], timeline)
