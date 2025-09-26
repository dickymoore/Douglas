import pytest

from douglas.core import Douglas
from douglas.dashboard.data import load_dashboard_data
from douglas.net.offline_guard import activate_offline_guard, deactivate_offline_guard
from tests.integration.test_offline_sprint_zero import _prepare_project


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_dashboard_loader_after_mock_run(tmp_path, monkeypatch):
    project_dir = _prepare_project(tmp_path)
    monkeypatch.setenv("DOUGLAS_OFFLINE", "1")
    activate_offline_guard()
    try:
        Douglas(config_path=project_dir / "douglas.yaml").run_loop()
    finally:
        deactivate_offline_guard()

    data = load_dashboard_data(project_dir / ".douglas")
    assert data.summary.counts
    assert data.inbox["unanswered"] >= 0
    assert isinstance(data.cumulative_flow, dict)
