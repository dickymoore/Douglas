import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from douglas.core import Douglas
from douglas.net.offline_guard import activate_offline_guard, deactivate_offline_guard

EXAMPLE_PROJECT = Path(__file__).resolve().parents[2] / "examples" / "hello-douglas"


def _prepare_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "hello-douglas"
    shutil.copytree(EXAMPLE_PROJECT, project_dir)
    config_path = project_dir / "douglas.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("ai", {})
    config["ai"].update({"mode": "mock", "seed": 123, "record_cassettes": False})
    loop = config.setdefault("loop", {})
    steps = loop.get("steps", [])
    filtered = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") in {"plan", "generate", "review"}
    ]
    if not filtered:
        filtered = [
            {"name": "plan", "role": "ProductOwner"},
            {"name": "generate", "role": "Developer"},
            {"name": "review", "role": "Reviewer"},
        ]
    loop["steps"] = filtered
    loop["max_iterations"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    subprocess.run(["git", "init"], cwd=project_dir, check=True, stdout=subprocess.PIPE)
    return project_dir


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_offline_sprint_zero(tmp_path, monkeypatch):
    project_dir = _prepare_project(tmp_path)
    monkeypatch.setenv("DOUGLAS_OFFLINE", "1")
    activate_offline_guard()
    try:
        orchestrator = Douglas(config_path=project_dir / "douglas.yaml")
        orchestrator.run_loop()
    finally:
        deactivate_offline_guard()

    readme = (project_dir / "README.md").read_text(encoding="utf-8")
    assert "Feature notes (mock)" in readme

    backlog_path = project_dir / ".douglas" / "state" / "backlog.json"
    assert backlog_path.exists()
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    assert backlog["items"], "Backlog should contain mock entries"

    test_files = list((project_dir / "tests").glob("test_smoke_*.py"))
    assert test_files, "Mock provider should create smoke tests"

    history_path = project_dir / "ai-inbox" / "history.jsonl"
    assert history_path.exists()
