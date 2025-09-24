from pathlib import Path

import yaml

from douglas.pipelines import standup as standuppipe


def _write_backlog(tmp_path: Path):
    backlog = {
        "epics": [
            {"id": "EP-1", "name": "Onboarding"},
        ],
        "stories": [
            {
                "id": "US-1",
                "name": "Create account",
                "tasks": [
                    {"id": "TK-1", "description": "Implement signup"},
                ],
            }
        ],
    }
    backlog_path = tmp_path / "ai-inbox" / "backlog"
    backlog_path.mkdir(parents=True, exist_ok=True)
    (backlog_path / "pre-features.yaml").write_text(
        yaml.safe_dump(backlog, sort_keys=False),
        encoding="utf-8",
    )
    return backlog_path / "pre-features.yaml"


def _write_question(tmp_path: Path):
    questions_dir = tmp_path / "user-portal" / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    question_path = questions_dir / "sprint-1-developer-Q-1.md"
    question_path.write_text(
        """---
role: Developer
id: Q-1
status: OPEN
topic: Environment access
context: Waiting on VPN approval
---
Body text
""",
        encoding="utf-8",
    )
    return question_path


def test_standup_pipeline_outputs_markdown(tmp_path):
    backlog_path = _write_backlog(tmp_path)
    questions_dir = _write_question(tmp_path).parent
    output_dir = tmp_path / "ai-inbox" / "sprints" / "sprint-1" / "standups"

    context = standuppipe.StandupContext(
        project_root=tmp_path,
        sprint_index=1,
        sprint_day=1,
        backlog_path=backlog_path,
        questions_dir=questions_dir,
        output_dir=output_dir,
        planning_config={},
    )

    result = standuppipe.run_standup(context)

    assert result.wrote_report is True
    assert result.output_path.exists()
    content = result.output_path.read_text(encoding="utf-8")
    assert "Sprint 1 – Day 1 Standup" in content
    assert "Create account" in content
    assert "Environment access" in content
