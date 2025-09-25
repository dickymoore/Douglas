from __future__ import annotations

from douglas.core import Douglas


def _make_orchestrator(loop_overrides: dict | None = None) -> Douglas:
    config = {
        "project": {"name": "Test", "language": "python"},
        "ai": {"default_provider": "codex", "providers": {"codex": {"provider": "codex"}}},
        "loop": {
            "steps": [],
            "exit_condition_mode": "all",
            "exit_conditions": [],
        },
    }
    if loop_overrides:
        config["loop"].update(loop_overrides)
    return Douglas(config_data=config)

MERGE_CONFLICT< feature/codex
MERGE_CONFLICT=
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
MERGE_CONFLICT> main

def test_all_features_delivered_requires_release_and_completion():
    orchestrator = _make_orchestrator()
    orchestrator.sprint_manager.completed_features.add("feature-1")
    orchestrator._loop_outcomes["push"] = True

    assert orchestrator._all_features_delivered() is True

MERGE_CONFLICT< feature/codex
    orchestrator = _make_orchestrator()
    orchestrator.sprint_manager.completed_features.add("feature-1")
    orchestrator.sprint_manager.pending_events["feature"] = 1
    orchestrator._loop_outcomes["push"] = True
MERGE_CONFLICT=
    assert test_calls == [
        "test",
        "test",
    ], "Loop should run for the configured iteration limit when exit conditions are absent."
MERGE_CONFLICT> main

    assert orchestrator._all_features_delivered() is False


def test_feature_delivery_goal_met_respects_goal_and_tests():
    orchestrator = _make_orchestrator({"feature_goal": 2})
    orchestrator.sprint_manager.completed_features.update({"feat-1", "feat-2"})
    orchestrator._loop_outcomes["push"] = True
    orchestrator._loop_outcomes["test"] = True

    assert orchestrator._feature_delivery_goal_met() is True

    orchestrator = _make_orchestrator({"feature_goal": 3})
    orchestrator.sprint_manager.completed_features.update({"feat-1", "feat-2"})
    orchestrator._loop_outcomes["push"] = True
    orchestrator._loop_outcomes["test"] = True

    assert orchestrator._feature_delivery_goal_met() is False

    orchestrator = _make_orchestrator({"feature_goal": 1})
    orchestrator.sprint_manager.completed_features.add("feat-1")
    orchestrator._loop_outcomes["push"] = True
    orchestrator._loop_outcomes["test"] = False

    assert orchestrator._feature_delivery_goal_met() is False
