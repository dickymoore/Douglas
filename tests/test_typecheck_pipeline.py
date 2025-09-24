import subprocess
from pathlib import Path

import pytest


from douglas.pipelines import typecheck


def _completed(command: list[str]):
    return subprocess.CompletedProcess(command, 0)


def test_run_typecheck_executes_all_commands(monkeypatch):
    calls = []

    def fake_run(command, check=True):
        calls.append((tuple(command), check))
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    typecheck.run_typecheck(additional_commands=[["echo", "ok"]])

    expected_commands = [
        ("mypy", "."),
        ("echo", "ok"),
    ]
    assert [call[0] for call in calls] == expected_commands
    assert all(check is True for _, check in calls)


def test_run_typecheck_fails_on_type_error(monkeypatch):
    def fake_run(command, check=True):
        if command[0] == "mypy":
            raise subprocess.CalledProcessError(returncode=2, cmd=command)
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        typecheck.run_typecheck()

    assert exc_info.value.code == 2


def test_run_typecheck_handles_missing_command(monkeypatch):
    def fake_run(command, check=True):
        if command[0] == "mypy":
            raise FileNotFoundError("mypy not found")
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        typecheck.run_typecheck()

    assert exc_info.value.code == 1
