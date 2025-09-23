import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas import cli as cli_module  # noqa: E402
from douglas.cli import app  # noqa: E402
from douglas.core import Douglas  # noqa: E402


def test_cli_init_without_config(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    result = runner.invoke(app, ["init", "sample-project", "--non-interactive"])

    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "sample-project"
    scaffold_config = yaml.safe_load(
        (project_dir / "douglas.yaml").read_text(encoding="utf-8")
    )

    assert scaffold_config["project"]["name"] == "sample-project"
    assert scaffold_config["ai"]["provider"] == "openai"
    assert scaffold_config["loop"]["exit_conditions"] == ["ci_pass"]
    assert scaffold_config["history"]["max_log_excerpt_length"] == 4000
    assert scaffold_config["sprint"]["length_days"] == 10


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

        def init_project(self, project_name: str, non_interactive: bool = False):
            type(self).init_project_calls.append((project_name, non_interactive))

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

    assert DummyDouglas.init_project_calls == [("demo-project", True)]
