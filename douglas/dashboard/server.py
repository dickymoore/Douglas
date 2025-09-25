"""Dashboard exposing Douglas progress state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - fallback when FastAPI is not installed
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except Exception:  # pragma: no cover - fallback branch
    FastAPI = None  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

from douglas.dashboard.data import DashboardData, load_dashboard_data
from douglas.logging import get_logger


logger = get_logger(__name__)


def create_app(state_root: Path | str = Path(".")) -> FastAPI:
    """Create a FastAPI app serving Douglas dashboard data."""

    root_path = Path(state_root)
    if FastAPI is None:  # pragma: no cover - fallback without FastAPI installed
        return _FallbackDashboard(root_path)  # type: ignore[return-value]
    app = FastAPI(title="Douglas Dashboard", version="1.0.0")

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/summary")
    def api_summary() -> dict[str, Any]:
        data = _load(root_path)
        return {
            "counts": data.summary.counts,
            "inbox": data.inbox,
        }

    @app.get("/api/burndown")
    def api_burndown() -> list[dict[str, Any]]:
        data = _load(root_path)
        return [
            {
                "date": point.date.isoformat(),
                "remaining": point.remaining,
                "completed": point.completed,
            }
            for point in data.burndown
        ]

    @app.get("/api/cumulative-flow")
    def api_cumulative_flow() -> dict[str, list[tuple[str, int]]]:
        data = _load(root_path)
        return data.cumulative_flow

    @app.get("/")
    def index() -> HTMLResponse:
        dashboard_html = _render_template()
        return HTMLResponse(content=dashboard_html)

    return app


def _load(root_path: Path) -> DashboardData:
    if not root_path.exists():
        raise HTTPException(status_code=404, detail="State root not found")
    data = load_dashboard_data(root_path)
    logger.debug("Loaded dashboard data", extra={"metadata": {"root": str(root_path)}})
    return data


def _render_template() -> str:
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


class _FallbackDashboard:
    """Simplified dashboard used when FastAPI is not available."""

    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    def healthcheck(self) -> dict[str, str]:  # pragma: no cover - trivial
        return {"status": "ok"}

    def api_summary(self) -> dict[str, Any]:
        data = _load(self._root_path)
        return {"counts": data.summary.counts, "inbox": data.inbox}

    def api_burndown(self) -> list[dict[str, Any]]:
        data = _load(self._root_path)
        return [
            {
                "date": point.date.isoformat(),
                "remaining": point.remaining,
                "completed": point.completed,
            }
            for point in data.burndown
        ]

    def api_cumulative_flow(self) -> dict[str, list[tuple[str, int]]]:
        data = _load(self._root_path)
        return data.cumulative_flow

    def index(self) -> str:
        return _render_template()


def render_static_dashboard(state_root: Path | str, output_dir: Path | str) -> Path:
    """Render the dashboard summary to a standalone HTML file."""

    data = load_dashboard_data(state_root)
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    template = _render_template()
    hydrated = template.replace("__DASHBOARD_PAYLOAD__", json.dumps(_to_payload(data)))
    target = output_directory / "index.html"
    target.write_text(hydrated, encoding="utf-8")
    logger.info("Static dashboard written to %s", target)
    return target


def _to_payload(data: DashboardData) -> dict[str, Any]:
    return {
        "counts": data.summary.counts,
        "inbox": data.inbox,
        "burndown": [
            {
                "date": point.date.isoformat(),
                "remaining": point.remaining,
                "completed": point.completed,
            }
            for point in data.burndown
        ],
        "cumulative_flow": data.cumulative_flow,
    }


__all__ = ["create_app", "render_static_dashboard"]
