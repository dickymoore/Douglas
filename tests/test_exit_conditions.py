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


def test_all_features_delivered_requires_release_and_completion():
    orchestrator = _make_orchestrator()
    orchestrator.sprint_manager.completed_features.add("feature-1")
    orchestrator._loop_outcomes["push"] = True

    assert orchestrator._all_features_delivered() is True

    orchestrator = _make_orchestrator()
    orchestrator.sprint_manager.completed_features.add("feature-1")
    orchestrator.sprint_manager.pending_events["feature"] = 1
    orchestrator._loop_outcomes["push"] = True

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
