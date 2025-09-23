import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.core import Douglas
from douglas.pipelines import security as securitypipe


def _completed(command, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_run_security_executes_bandit_by_default(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        return _completed(command, stdout="bandit ok")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = securitypipe.run_security()

    assert calls
    command, options = calls[0]
    assert command == ["bandit", "-q", "-r", "."]
    assert options.get("check") is True
    assert options.get("capture_output") is True
    assert options.get("text") is True
    assert report.results[0].name == "bandit"
    assert report.results[0].stdout == "bandit ok"


def test_run_security_supports_semgrep_configuration(monkeypatch: pytest.MonkeyPatch):
    recorded: list[list[str]] = []

    def fake_run(command, **kwargs):
        recorded.append(list(command))
        return _completed(command, stdout="semgrep ok")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = securitypipe.run_security(
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
    assert report.results[0].name == "semgrep"
    assert report.results[0].stdout == "semgrep ok"


def test_run_security_raises_on_tool_failure(monkeypatch: pytest.MonkeyPatch):
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=5,
            cmd=command,
            output="issues found",
            stderr="security failure",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(securitypipe.SecurityCheckError) as exc_info:
        securitypipe.run_security(tools=["bandit"])

    err = exc_info.value
    assert err.tool == "bandit"
    assert err.exit_code == 5
    assert "bandit" in " ".join(err.command)
    assert "security failure" in err.stderr


def test_run_security_rejects_unknown_tool():
    with pytest.raises(securitypipe.SecurityConfigurationError):
        securitypipe.run_security(tools=["unknown-tool"])


def _write_security_config(tmp_path: Path) -> Path:
    config_text = """
project:
  name: 'SecurityStep'
ai:
  provider: 'openai'
paths:
  inbox_dir: 'ai-inbox'
  sprint_prefix: 'sprint-'
loop:
  steps:
    - name: security
      role: Security
"""
    config_path = tmp_path / "douglas.yaml"
    config_path.write_text(config_text.strip() + "\n", encoding="utf-8")
    return config_path


class _StubProvider:
    def generate_code(self, prompt: str) -> str:  # pragma: no cover - stub API
        return ""


def test_security_step_records_history_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_security_config(tmp_path)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: _StubProvider())

    douglas = Douglas(config_path)

    report = securitypipe.SecurityReport(
        results=[
            securitypipe.SecurityToolResult(
                name="bandit",
                command=["bandit", "-q", "-r", "."],
                stdout="no issues",
                stderr="",
                exit_code=0,
            )
        ]
    )

    monkeypatch.setattr(securitypipe, "run_security", lambda **_: report)

    decision = douglas.cadence_manager.evaluate_step("security", {"name": "security"})
    result = douglas._execute_step("security", {"name": "security"}, decision)

    assert result.executed is True
    assert result.success is True

    summary_path = (
        tmp_path
        / "ai-inbox"
        / "sprints"
        / "sprint-1"
        / "roles"
        / "security"
        / "summary.md"
    )
    history_path = tmp_path / "ai-inbox" / "history.jsonl"

    assert summary_path.exists()
    assert history_path.exists()

    summary_text = summary_path.read_text(encoding="utf-8")
    history_entries = history_path.read_text(encoding="utf-8").strip().splitlines()

    assert "Completed security checks with bandit." in summary_text
    assert any("security_checks_passed" in line for line in history_entries)


def test_security_step_failure_records_bug_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_security_config(tmp_path)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: _StubProvider())

    douglas = Douglas(config_path)

    error = securitypipe.SecurityCheckError(
        "Bandit detected issues.",
        tool="bandit",
        command=["bandit", "-q", "-r", "."],
        exit_code=7,
        stdout="stdout details",
        stderr="stderr details",
    )

    def failing_security(**_):
        raise error

    monkeypatch.setattr(securitypipe, "run_security", failing_security)

    decision = douglas.cadence_manager.evaluate_step("security", {"name": "security"})

    with pytest.raises(SystemExit) as exc_info:
        douglas._execute_step("security", {"name": "security"}, decision)

    assert exc_info.value.code == 7

    summary_path = (
        tmp_path
        / "ai-inbox"
        / "sprints"
        / "sprint-1"
        / "roles"
        / "security"
        / "summary.md"
    )
    bugs_path = tmp_path / "ai-inbox" / "bugs.md"
    history_path = tmp_path / "ai-inbox" / "history.jsonl"

    assert summary_path.exists()
    assert bugs_path.exists()

    summary_text = summary_path.read_text(encoding="utf-8")
    history_entries = history_path.read_text(encoding="utf-8").strip().splitlines()

    assert "failed with exit code 7" in summary_text
    assert any("step_failure" in line for line in history_entries)
