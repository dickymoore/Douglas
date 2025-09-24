import os
import subprocess
from pathlib import Path

from typer.testing import CliRunner


from douglas.cli import app
from douglas.core import Douglas


def test_cli_init_python_template(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "init",
            "demo-app",
            "--non-interactive",
            "--template",
            "python",
            "--push-policy",
            "per_feature",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    project_dir = tmp_path / "demo-app"
    assert (project_dir / "douglas.yaml").exists()
    assert (project_dir / "requirements-dev.txt").exists()
    assert (project_dir / "tests" / "test_app.py").exists()

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(project_dir / "src") + (
        os.pathsep + existing if existing else ""
    )
    subprocess.run(
        ["pytest", "-q"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_init_with_git_and_license(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "init",
            "demo-git",
            "--non-interactive",
            "--git",
            "--ci",
            "github",
            "--license",
            "mit",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    project_dir = tmp_path / "demo-git"
    assert (project_dir / ".git").exists()
    assert (project_dir / "LICENSE").exists()
    assert (project_dir / ".github" / "workflows" / "ci.yml").exists()
