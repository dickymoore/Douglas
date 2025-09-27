import json
from pathlib import Path

import pytest
import yaml

from douglas.core import Douglas
from douglas.net.offline_guard import activate_offline_guard, deactivate_offline_guard
from douglas.steps import planning as planning_step
from tests.integration.test_offline_sprint_zero import _prepare_project


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sprint_planning_is_deterministic(tmp_path, monkeypatch):
    project_dir = _prepare_project(tmp_path)
    config_path = project_dir / "douglas.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    planning_cfg = config.setdefault("planning", {})
    planning_cfg["items_per_sprint"] = 2
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setenv("DOUGLAS_OFFLINE", "1")
    activate_offline_guard()
    try:
        orchestrator = Douglas(config_path=config_path)
        orchestrator.run_loop()
    finally:
        deactivate_offline_guard()

    state_dir = project_dir / ".douglas" / "state"
    backlog_path = state_dir / "backlog.json"
    plan_path = state_dir / "sprint_plan_1.json"
    markdown_path = project_dir / "ai-inbox" / "planning" / "sprint_1.md"

    assert backlog_path.exists(), "backlog.json should be created"
    assert plan_path.exists(), "sprint plan JSON should be created"
    assert markdown_path.exists(), "sprint plan markdown should be created"

    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    markdown_text = markdown_path.read_text(encoding="utf-8")

    backlog_items = backlog.get("items") or []
    backlog_ids = {
        str(item.get("id"))
        for item in backlog_items
        if isinstance(item, dict) and item.get("id") is not None
    }
    plan_commitments = plan_data.get("commitments") or []
    selected_ids = {
        commitment.get("id")
        for commitment in plan_commitments
        if isinstance(commitment, dict) and commitment.get("id") is not None
    }
    assert selected_ids, "Sprint plan should include commitments"
    assert selected_ids <= backlog_ids, "Commitments must exist in the backlog"

    assert plan_data.get("sprint") == 1
    assert "Mock Sprint" in markdown_text

    ai_config = config.get("ai", {}) or {}
    seed_value = ai_config.get("seed", 0)
    planning_seed = _safe_int(seed_value)

    items_per_sprint = _safe_int(plan_data.get("items_requested", 0))

    rerun_context = planning_step.PlanningContext(
        project_root=project_dir,
        backlog_state_path=backlog_path,
        sprint_index=1,
        items_per_sprint=items_per_sprint,
        seed=planning_seed,
        provider=None,
        summary_intro=planning_cfg.get("goal"),
    )
    planning_step.run_planning(rerun_context)

    rerun_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert rerun_plan == plan_data, "Sprint plan JSON should be deterministic"
