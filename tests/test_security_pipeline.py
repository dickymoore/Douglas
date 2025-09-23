import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.pipelines import security


def _completed(command: list[str]):
    return subprocess.CompletedProcess(command, 0)


def test_run_security_executes_available_tools(monkeypatch):
    calls = []

    def fake_which(tool: str):
        if tool in {"bandit", "semgrep"}:
            return f"/usr/bin/{tool}"
        return None

    def fake_run(command, check=True):
        calls.append((tuple(command), check))
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(security.shutil, "which", fake_which)

    security.run_security(additional_commands=[["echo", "ok"]])

    expected = [
        ("bandit", "-q", "-r", "."),
        ("semgrep", "--config", "auto"),
        ("echo", "ok"),
    ]
    assert [call[0] for call in calls] == expected
    assert all(check is True for _, check in calls)


def test_run_security_handles_command_failure(monkeypatch):
    def fake_which(tool: str):
        return f"/usr/bin/{tool}" if tool == "bandit" else None

    def fake_run(command, check=True):
        raise subprocess.CalledProcessError(returncode=5, cmd=command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(security.shutil, "which", fake_which)

    with pytest.raises(SystemExit) as exc_info:
        security.run_security()

    assert exc_info.value.code == 5


def test_run_security_requires_at_least_one_tool(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(args[0]))
    monkeypatch.setattr(security.shutil, "which", lambda tool: None)

    with pytest.raises(SystemExit) as exc_info:
        security.run_security()

    assert exc_info.value.code == 1


def test_run_security_skips_optional_tools(monkeypatch):
    calls = []

    def fake_which(tool: str):
        return f"/usr/bin/{tool}" if tool == "bandit" else None

    def fake_run(command, check=True):
        calls.append(tuple(command))
        return _completed(command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(security.shutil, "which", fake_which)

    security.run_security()

    assert calls == [("bandit", "-q", "-r", ".")]
