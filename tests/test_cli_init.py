from pathlib import Path

import yaml
from typer.testing import CliRunner


from douglas import cli as cli_module
from douglas.cli import app
from douglas.core import Douglas


def test_load_default_init_config_uses_repo_template(monkeypatch):
    calls = []

    def _record_secho(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(cli_module.typer, "secho", _record_secho)

    config = cli_module._load_default_init_config()

    developer_cadence = (
        config.get("cadence", {}).get("Developer", {}).get("development")
    )

    assert developer_cadence == "daily"
    assert calls == []


def test_cli_init_without_config(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    result = runner.invoke(
        app,
        [
            "init",
            "sample-project",
            "--non-interactive",
            "--template",
            "python",
            "--push-policy",
            "per_feature_complete",
        ],
    )

    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "sample-project"
    scaffold_config = yaml.safe_load(
        (project_dir / "douglas.yaml").read_text(encoding="utf-8")
    )

    assert scaffold_config["project"]["name"] == "sample-project"
    assert scaffold_config["ai"]["default_provider"] == "codex"
    providers = scaffold_config["ai"].get("providers", {})
    codex_cfg = providers.get("codex", {})
    assert codex_cfg.get("provider") == "codex"
    assert codex_cfg.get("model") == "gpt-5-codex"
    loop_cfg = scaffold_config["loop"]
    assert loop_cfg["exit_condition_mode"] == "all"
    assert loop_cfg["exit_conditions"] == [
        "feature_delivery_complete",
        "sprint_demo_complete",
    ]
    assert loop_cfg["exhaustive"] is False
    assert scaffold_config["history"]["max_log_excerpt_length"] == 4000
    assert scaffold_config["sprint"]["length_days"] == 10
    assert scaffold_config["push_policy"] == "per_feature_complete"
    planning = scaffold_config.get("planning", {})
    assert planning.get("enabled") is True
    assert planning.get("sprint_zero_only") is False
    assert planning.get("first_day_only") is True
    charters = planning.get("charters", {})
    assert charters.get("enabled", True) is True

    plan_step = next(
        (step for step in scaffold_config["loop"]["steps"] if step["name"] == "plan"),
        {},
    )
    assert plan_step.get("cadence") == "daily"

    agents = scaffold_config.get("agents", {}).get("roles", [])
    assert "Account Manager" in agents

    accountability = scaffold_config.get("accountability", {})
    assert accountability.get("enabled") is True
    assert accountability.get("stall_iterations") == 3
    assert accountability.get("soft_stop") is True


def test_cli_init_uses_default_factory_once(monkeypatch, tmp_path):
    sentinel_config = {"from_factory": True}

    class DummyDouglas:
        init_calls = []
        init_project_calls = []

        def __init__(self, *args, **kwargs):
            type(self).init_calls.append(kwargs)
            self._config_kwargs = kwargs

        def run_loop(self):  # pragma: no cover - not used in this test
            raise AssertionError("run_loop should not be called during init")

        def check(self):  # pragma: no cover - not used in this test
            raise AssertionError("check should not be called during init")

        def init_project(self, *args, **kwargs):
            type(self).init_project_calls.append((args, kwargs))

    monkeypatch.setattr(
        cli_module, "_load_default_init_config", lambda: sentinel_config
    )
    monkeypatch.setattr(cli_module, "Douglas", DummyDouglas)
    DummyDouglas.init_calls = []
    DummyDouglas.init_project_calls = []

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "demo-project", "--non-interactive"])

    assert result.exit_code == 0, result.output
    assert len(DummyDouglas.init_calls) == 1

    init_kwargs = DummyDouglas.init_calls[0]
    used_config = init_kwargs.get("config_data")
    if used_config is None:
        used_config = init_kwargs.get("config")
    assert used_config == sentinel_config

    assert len(DummyDouglas.init_project_calls) == 1
    init_args, init_call_kwargs = DummyDouglas.init_project_calls[0]
    assert init_args == (Path("demo-project"),)
    assert init_call_kwargs["name"] is None
    assert init_call_kwargs["template"] == "python"
    assert init_call_kwargs["push_policy"] == "per_feature"


def test_cli_init_supports_provider_override(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    result = runner.invoke(
        app,
        [
            "init",
            "with-alt-provider",
            "--non-interactive",
            "--provider",
            "claude_code",
            "--model",
            "claude-test",
        ],
    )

    assert result.exit_code == 0, result.output

    config = yaml.safe_load(
        (tmp_path / "with-alt-provider" / "douglas.yaml").read_text(encoding="utf-8")
    )
    assert config["ai"]["default_provider"] == "claude_code"
    assert config["ai"]["providers"]["claude_code"]["model"] == "claude-test"
