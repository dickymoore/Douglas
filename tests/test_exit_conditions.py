import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import yaml

from douglas.core import Douglas
from douglas.pipelines import demo as demo_pipeline
from douglas.pipelines import test as testpipe


def _init_repo(
    tmp_path: Path,
    *,
    steps: List[dict],
    exit_conditions: Optional[List[str]] = None,
    max_iterations: Optional[int] = None,
) -> Path:
    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "config", "user.email", "exit@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Exit Conditions"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    loop_config: dict = {"steps": steps}
    if exit_conditions is not None:
        loop_config["exit_conditions"] = exit_conditions
    if max_iterations is not None:
        loop_config["max_iterations"] = max_iterations

    config = {
        "project": {"name": "ExitConditions", "language": "python"},
        "ai": {"provider": "openai"},
        "loop": loop_config,
    }

    config_path = tmp_path / "douglas.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    (tmp_path / "README.md").write_text("Initial content\n", encoding="utf-8")

    subprocess.run(
        ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "chore: initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    return config_path


def test_exit_condition_allows_remaining_steps(monkeypatch, tmp_path):
    config_path = _init_repo(
        tmp_path,
        steps=[{"name": "test"}, {"name": "push"}],
        exit_conditions=["tests_pass"],
    )

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())
    monkeypatch.setattr(Douglas, "_run_local_checks", lambda self: (True, "ok"))
    monkeypatch.setattr(Douglas, "_monitor_ci", lambda self, *args, **kwargs: None)

    test_calls: list[str] = []

    def fake_run_tests():
        test_calls.append("test")

    monkeypatch.setattr(testpipe, "run_tests", fake_run_tests)

    push_calls: list[str] = []

    def fake_push(self):
        push_calls.append("push")
        return True, "pushed"

    monkeypatch.setattr(Douglas, "_run_git_push", fake_push)

    douglas = Douglas(config_path)
    douglas.sprint_manager.mark_feature_completed("demo-feature")
    douglas.sprint_manager.commits_since_last_push = 1

    douglas.run_loop()

    assert test_calls == ["test"], "Test step should execute exactly once."
    assert push_calls == [
        "push"
    ], "Push step should still execute despite exit condition."


def test_exit_condition_stops_additional_iterations(monkeypatch, tmp_path):
    config_path = _init_repo(
        tmp_path,
        steps=[{"name": "test"}],
        exit_conditions=["tests_pass"],
        max_iterations=3,
    )

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    test_calls: list[str] = []

    def fake_run_tests():
        test_calls.append("test")

    monkeypatch.setattr(testpipe, "run_tests", fake_run_tests)

    douglas = Douglas(config_path)
    douglas.run_loop()

    assert test_calls == ["test"], "Loop should exit after first successful iteration."


def test_loop_repeats_until_iteration_limit(monkeypatch, tmp_path):
    config_path = _init_repo(
        tmp_path,
        steps=[{"name": "test"}],
        exit_conditions=[],
        max_iterations=2,
    )

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    test_calls: list[str] = []

    def fake_run_tests():
        test_calls.append("test")

    monkeypatch.setattr(testpipe, "run_tests", fake_run_tests)

    douglas = Douglas(config_path)
    douglas.run_loop()

    assert test_calls == [
        "test",
        "test",
    ], "Loop should run for the configured iteration limit when exit conditions are absent."


def test_exit_condition_for_demo_completion(monkeypatch, tmp_path):
    config_path = _init_repo(
        tmp_path,
        steps=[{"name": "demo", "cadence": "daily"}],
        exit_conditions=["sprint_demo_complete"],
        max_iterations=3,
    )

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    demo_calls: list[int] = []

    def fake_demo(context):
        demo_calls.append(context["sprint_manager"].current_iteration)
        project_root = context["project_root"]
        output_path = project_root / "demos" / "demo.md"
        return SimpleNamespace(
            output_path=output_path,
            format="md",
            sprint_folder=f"sprint-{context['sprint_manager'].sprint_index}",
            as_event_payload=lambda: {"output": str(output_path)},
        )

    monkeypatch.setattr(demo_pipeline, "write_demo_pack", fake_demo)

    douglas = Douglas(config_path)
    douglas.run_loop()

    assert demo_calls == [1]
