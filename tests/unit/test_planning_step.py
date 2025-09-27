import json
from pathlib import Path

from douglas.steps import planning as planning_step


def test_extract_fallback_items_handles_nested_backlog():
    fallback_payload = {
        "epics": [
            {"identifier": "EP-1", "title": "Epic One"},
        ],
        "features": [
            {"identifier": "FT-1", "title": "Feature One", "state": "ready"},
        ],
        "stories": [
            {"identifier": "ST-1", "title": "Story One", "owner": "dev"},
        ],
    }

    items = planning_step._extract_fallback_items(fallback_payload)

    identifiers = {item.get("id") for item in items if isinstance(item, dict)}
    assert {"EP-1", "FT-1", "ST-1"} <= identifiers


def test_run_planning_overwrites_invalid_existing_plan(tmp_path):
    project_root = Path(tmp_path)
    state_dir = project_root / ".douglas" / "state"
    state_dir.mkdir(parents=True)
    backlog_path = state_dir / "backlog.json"
    backlog_payload = {
        "items": [
            {"id": "FT-1", "title": "Feature One", "status": "ready"},
            {"id": "ST-2", "title": "Story Two", "status": "todo"},
        ]
    }
    backlog_path.write_text(json.dumps(backlog_payload), encoding="utf-8")

    invalid_plan_path = state_dir / "sprint_plan_1.json"
    invalid_plan_path.write_text("{", encoding="utf-8")

    context = planning_step.PlanningContext(
        project_root=project_root,
        backlog_state_path=backlog_path,
        sprint_index=1,
    )

    result = planning_step.run_planning(context)

    assert result.success
    refreshed = json.loads(invalid_plan_path.read_text(encoding="utf-8"))
    assert refreshed["sprint"] == 1


def test_run_planning_uses_fallback_when_state_missing(tmp_path):
    project_root = Path(tmp_path)
    state_dir = project_root / ".douglas" / "state"
    state_dir.mkdir(parents=True)
    backlog_path = state_dir / "backlog.json"

    fallback_payload = {
        "features": [
            {"identifier": "FT-1", "title": "Feature One", "state": "ready"},
            {"identifier": "FT-2", "title": "Feature Two", "state": "todo"},
        ]
    }

    context = planning_step.PlanningContext(
        project_root=project_root,
        backlog_state_path=backlog_path,
        sprint_index=1,
        backlog_fallback=fallback_payload,
        items_per_sprint=1,
    )

    result = planning_step.run_planning(context)

    assert result.success
    assert result.used_fallback
    assert result.plan is not None
    assert result.plan.commitments
