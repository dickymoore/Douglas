import subprocess
from pathlib import Path

import pytest


from douglas.pipelines import lint


def _completed(command: list[str]):
    return subprocess.CompletedProcess(command, 0)


def test_run_lint_executes_all_commands(monkeypatch):
    calls = []

    def fake_run(command, check=True):
        calls.append((tuple(command), check))
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    lint.run_lint(additional_commands=[["echo", "ok"]])

    expected_commands = [
        ("ruff", "check", "."),
        ("black", "--check", "."),
        ("isort", "--check-only", "."),
        ("echo", "ok"),
    ]
    assert [call[0] for call in calls] == expected_commands
    assert all(check is True for _, check in calls)


def test_run_lint_fails_on_linter_error(monkeypatch):
    def fake_run(command, check=True):
        if command[0] == "black":
            raise subprocess.CalledProcessError(returncode=3, cmd=command)
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        lint.run_lint()

    assert exc_info.value.code == 3


def test_run_lint_handles_missing_command(monkeypatch):
    def fake_run(command, check=True):
        if command[0] == "ruff":
            raise FileNotFoundError("ruff not found")
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        lint.run_lint()

    assert exc_info.value.code == 1
