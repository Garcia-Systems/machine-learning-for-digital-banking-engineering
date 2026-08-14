"""FastAPI boundary for the internal, server-rendered engineering dashboard."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import DashboardSnapshot
from .service import DashboardService

HERE = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))


def create_dashboard_app(service: DashboardService, current: DashboardSnapshot,
                         timeline: list[DashboardSnapshot] | None = None) -> FastAPI:
    app = FastAPI(title="Harbor Engineering Dashboard", version="2")
    app.mount("/dashboard/static", StaticFiles(directory=str(HERE / "static")), name="dashboard-static")
    app.state.dashboard_service, app.state.dashboard_snapshot = service, current
    app.state.dashboard_timeline = timeline or service.history or [current]

    def render(request, snapshot, template="dashboard.html"):
        return TEMPLATES.TemplateResponse(request=request, name=template,
            context={"snapshot": snapshot, "history": app.state.dashboard_timeline})

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request):
        return render(request, app.state.dashboard_snapshot)

    @app.get("/dashboard/incident", response_class=HTMLResponse)
    def incident(request: Request, time: str | None = None):
        selected = app.state.dashboard_timeline[-1]
        if time:
            matches = [item for item in app.state.dashboard_timeline if item.telemetry.timestamp.strftime("%H:%M") == time]
            if not matches:
                raise HTTPException(404, "incident time not found")
            selected = matches[0]
        return render(request, selected, "incident_playback.html")

    return app
