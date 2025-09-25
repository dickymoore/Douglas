import subprocess

import pytest


from douglas.pipelines import security


def _completed(command, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_run_security_executes_bandit_by_default(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_which(tool: str):
        return f"/usr/bin/{tool}" if tool == "bandit" else None

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        return _completed(command, stdout="bandit ok")

    monkeypatch.setattr(security.shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    report = security.run_security()

    assert [call[0] for call in calls] == [["bandit", "-q", "-r", "."]]
    assert report.tool_names() == ["bandit"]
    assert report.skipped_tools == ["semgrep"]
    assert report.results[0].stdout == "bandit ok"
    options = calls[0][1]
    assert options.get("check") is True
    assert options.get("capture_output") is True
    assert options.get("text") is True


def test_run_security_executes_additional_commands(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_which(tool: str):
        return None

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return _completed(command, stdout="ok")

    monkeypatch.setattr(security.shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    report = security.run_security(additional_commands=[["echo", "ok"]])

    assert calls == [["echo", "ok"]]
    assert report.tool_names() == ["echo"]
    assert report.skipped_tools == ["bandit", "semgrep"]


def test_run_security_handles_command_failure(monkeypatch: pytest.MonkeyPatch):
    def fake_which(tool: str):
        return "/usr/bin/bandit" if tool == "bandit" else None

    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=5,
            cmd=command,
            output="issues found",
            stderr="security failure",
        )

    monkeypatch.setattr(security.shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(security.SecurityCheckError) as exc_info:
        security.run_security()

    err = exc_info.value
    assert isinstance(err, SystemExit)
    assert err.tool == "bandit"
    assert err.exit_code == 5
    assert "bandit" in " ".join(err.command)
    assert err.stdout == "issues found"
    assert err.stderr == "security failure"


def test_run_security_requires_at_least_one_tool(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(security.shutil, "which", lambda tool: None)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(args[0]))

    with pytest.raises(security.SecurityCheckError) as exc_info:
        security.run_security()

    err = exc_info.value
    assert err.exit_code == 1
    assert err.tool == "security"


def test_run_security_supports_semgrep_configuration(monkeypatch: pytest.MonkeyPatch):
    recorded: list[list[str]] = []

    def fake_run(command, **kwargs):
        recorded.append(list(command))
        return _completed(command, stdout="semgrep ok")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = security.run_security(
        tools=[
            {
                "name": "semgrep",
                "args": ["--baseline", "baseline.json"],
                "paths": ["src", "tests"],
            }
        ],
        default_paths=["."],
    )

    assert recorded == [
        [
            "semgrep",
            "--config",
            "auto",
            "--error",
            "--baseline",
            "baseline.json",
            "src",
            "tests",
        ]
    ]
    assert report.tool_names() == ["semgrep"]
    assert report.results[0].stdout == "semgrep ok"


def test_run_security_rejects_unknown_tool():
    with pytest.raises(security.SecurityConfigurationError):
        security.run_security(tools=["unknown-tool"])
