import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
