"""FastAPI presentation boundary for Harbor's internal engineering dashboard."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .models import DashboardSnapshot
from .service import DashboardService

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_dashboard_app(service: DashboardService, current: DashboardSnapshot) -> FastAPI:
    """Create an injected, in-process-testable server-rendered dashboard."""
    app = FastAPI(title="Harbor Engineering Dashboard", version="1")
    app.state.dashboard_service = service
    app.state.dashboard_snapshot = current

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={"snapshot": app.state.dashboard_snapshot, "history": service.history},
        )

    return app
