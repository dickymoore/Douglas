"""StepResult schema and orchestrator integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from douglas.domain.step_result import StepArtifact, StepEvent, StepResult
from douglas.orchestrator import StepResultOrchestrator


def test_step_result_round_trip_json() -> None:
    artifact = StepArtifact(
        path=".douglas/state/backlog.json",
        content="{\n  \"items\": []\n}\n",
        description="State snapshot",
        metadata={"kind": "state"},
    )
    event = StepEvent(message="Completed planning", level="info")
    result = StepResult(
        step_name="plan",
        role="ProductOwner",
        agent="planner",
        seed=42,
        prompt_hash="abc123",
        timestamps={"started_at": "2024-01-01T00:00:00+00:00"},
        artifacts=[artifact],
        state_deltas={"items_added": 1},
        events=[event],
        questions=["What is next?"],
        answers=["Focus on delivery."],
        ci_reports=[{"name": "lint", "status": "passed"}],
        test_reports=[{"name": "unit", "status": "passed"}],
        retro_notes=["Keep deterministic outputs."],
        commits=[{"sha": "deadbeef"}],
        errors=[{"message": "none"}],
    )

    payload = result.to_json()
    decoded = json.loads(payload)
    assert set(decoded.keys()) == {
        "step_name",
        "role",
        "agent",
        "seed",
        "prompt_hash",
        "timestamps",
        "artifacts",
        "state_deltas",
        "events",
        "questions",
        "answers",
        "ci_reports",
        "test_reports",
        "retro_notes",
        "commits",
        "errors",
    }

    round_tripped = StepResult.from_json(payload)
    assert round_tripped == result


def test_orchestrator_applies_step_result(tmp_path: Path) -> None:
    orchestrator = StepResultOrchestrator(project_root=tmp_path)
    state_artifact = StepArtifact(
        path=".douglas/state/status.json",
        content=json.dumps({"status": "ok"}, sort_keys=True) + "\n",
    )
    inbox_artifact = StepArtifact(
        path="ai-inbox/summary.md",
        content="# Summary\n\nAll systems nominal.\n",
    )
    result = StepResult(
        step_name="status",
        role="Developer",
        agent="dev",
        timestamps=StepResult.default_timestamps(),
        artifacts=[state_artifact, inbox_artifact],
    )

    orchestrator.apply(result)

    state_path = tmp_path / ".douglas" / "state" / "status.json"
    inbox_path = tmp_path / "ai-inbox" / "summary.md"
    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    snapshot_path = tmp_path / ".douglas" / "state" / "last_step_result.json"

    assert state_path.read_text(encoding="utf-8") == state_artifact.content
    assert inbox_path.read_text(encoding="utf-8") == inbox_artifact.content
    history_lines = history_path.read_text(encoding="utf-8").splitlines()
    assert history_lines
    latest = json.loads(history_lines[-1])
    assert latest["step_name"] == "status"
    assert snapshot_path.read_text(encoding="utf-8")
