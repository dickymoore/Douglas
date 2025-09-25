import json
from datetime import datetime

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - FastAPI not installed
    TestClient = None

from douglas.dashboard.server import create_app, render_static_dashboard


def _seed_state(root):
    features_dir = root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "feat1.yaml").write_text("status: in_progress\n", encoding="utf-8")
    (features_dir / "feat2.yaml").write_text("status: finished\n", encoding="utf-8")

    inbox_dir = root / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    (inbox_dir / "question1.yaml").write_text("question: foo\n", encoding="utf-8")
    (inbox_dir / "question2.yaml").write_text(
        "question: bar\nanswer: baz\n", encoding="utf-8"
    )

    state_dir = root / ".douglas" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    history = [
        {"date": datetime.utcnow().isoformat(), "remaining": 5, "completed": 2},
    ]
    (state_dir / "sprint_history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    flow = {"in_progress": [{"date": datetime.utcnow().isoformat(), "count": 2}]}
    (state_dir / "cumulative_flow.json").write_text(json.dumps(flow), encoding="utf-8")


def test_dashboard_api_endpoints(tmp_path):
    _seed_state(tmp_path)
    app = create_app(tmp_path)
    if TestClient is not None:
        client = TestClient(app)
        summary = client.get("/api/summary").json()
        burndown = client.get("/api/burndown").json()
        flow = client.get("/api/cumulative-flow").json()
    else:
        summary = app.api_summary()
        burndown = app.api_burndown()
        flow = app.api_cumulative_flow()

    assert summary["counts"]["features"]["in_progress"] == 1
    assert summary["counts"]["features"]["finished"] == 1
    assert summary["inbox"] == {"unanswered": 1, "answered": 1}
    assert len(burndown) == 1
    assert "in_progress" in flow


def test_render_static_dashboard(tmp_path):
    _seed_state(tmp_path)
    output = tmp_path / "dashboard"
    target = render_static_dashboard(tmp_path, output)
    assert target.exists()
    contents = target.read_text(encoding="utf-8")
    assert "__DASHBOARD_PAYLOAD__" not in contents
    assert "features" in contents
