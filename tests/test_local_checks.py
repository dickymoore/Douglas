import json
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import yaml


from douglas.core import Douglas


class StaticProvider:
    def generate_code(self, prompt: str) -> str:
        return "chore: update"


def _init_repo(
    tmp_path: Path, *, steps: List[dict], exit_conditions: Optional[List[str]] = None
) -> Path:
    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    config_data: dict = {
        "project": {"name": "LocalChecks", "language": "python"},
        "ai": {"provider": "openai"},
        "loop": {"steps": steps},
        "push_policy": "per_feature",
    }
    if exit_conditions is not None:
        config_data["loop"]["exit_conditions"] = exit_conditions

    config_path = tmp_path / "douglas.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8"
    )

    readme = tmp_path / "README.md"
    readme.write_text("Initial content\n", encoding="utf-8")

    subprocess.run(
        ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "chore: initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    return config_path


def _patch_subprocess_for_command(
    monkeypatch,
    target_command: Sequence[str],
    *,
    exception_factory: Optional[Callable[[], BaseException]] = None,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
):
    """Patch ``subprocess.run`` to simulate behaviour for a single command."""

    original_run = subprocess.run
    expected = list(target_command)

    def fake_run(command, *args, **kwargs):
        if list(command) == expected:
            if exception_factory is not None:
                raise exception_factory()
            return subprocess.CompletedProcess(
                list(command),
                returncode,
                stdout=stdout,
                stderr=stderr,
            )
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return fake_run


def test_push_step_creates_bug_on_local_check_failure(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{"name": "push"}], exit_conditions=[])

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: StaticProvider())

    douglas = Douglas(config_path)
    douglas.sprint_manager.mark_feature_completed("demo-feature")
    douglas.sprint_manager.commits_since_last_push = 1

    def unexpected_push(self):
        raise AssertionError("push should not run when local checks fail")

    monkeypatch.setattr(
        Douglas, "_discover_local_check_commands", lambda self: [["fake-tool"]]
    )
    monkeypatch.setattr(Douglas, "_run_git_push", unexpected_push)
    monkeypatch.setattr(Douglas, "_monitor_ci", lambda self, branch, timeout=60: None)

    _patch_subprocess_for_command(
        monkeypatch,
        ["fake-tool"],
        exception_factory=lambda: FileNotFoundError("simulated local check failure"),
    )

    douglas.run_loop()

    bug_file = tmp_path / "ai-inbox" / "bugs.md"
    assert bug_file.exists()
    bug_contents = bug_file.read_text(encoding="utf-8")
    assert "simulated local check failure" in bug_contents
    assert "### Log Excerpt" in bug_contents
    assert "Local check command 'fake-tool' not found" in bug_contents

    handoff_path = (
        tmp_path
        / "ai-inbox"
        / "sprints"
        / "sprint-1"
        / "roles"
        / "devops"
        / "handoffs.md"
    )
    assert handoff_path.exists()
    handoff_contents = handoff_path.read_text(encoding="utf-8")
    assert "Resolve local guard check failures" in handoff_contents
    assert "simulated local check failure" in handoff_contents

    summary_path = (
        tmp_path
        / "ai-inbox"
        / "sprints"
        / "sprint-1"
        / "roles"
        / "devops"
        / "summary.md"
    )
    assert summary_path.exists()
    summary_contents = summary_path.read_text(encoding="utf-8")
    assert "Push blocked because required local checks failed." in summary_contents
    assert "HANDOFF-" in summary_contents

    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    assert history_path.exists()
    history_entries = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [entry["event"] for entry in history_entries]
    assert "local_checks_fail" in events
    assert "bug_reported" in events
    assert "step_failure" in events
    step_failures = [
        entry for entry in history_entries if entry["event"] == "step_failure"
    ]
    assert step_failures and step_failures[0].get("bug_id", "").startswith("FEAT-BUG-")


def test_run_local_checks_success_records_history(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{"name": "push"}], exit_conditions=[])

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: StaticProvider())
    douglas = Douglas(config_path)

    monkeypatch.setattr(
        Douglas, "_discover_local_check_commands", lambda self: [["echo", "ok"]]
    )

    _patch_subprocess_for_command(
        monkeypatch,
        ["echo", "ok"],
        stdout="all good\n",
    )

    success, logs = douglas._run_local_checks()
    assert success is True
    assert "all good" in logs

    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    assert history_path.exists()
    entries = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(entry["event"] == "local_checks_pass" for entry in entries)


def test_semgrep_network_failure_is_skipped(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{"name": "push"}], exit_conditions=[])

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: StaticProvider())
    douglas = Douglas(config_path)

    semgrep_command = ["semgrep", "--config", "auto"]
    monkeypatch.setattr(
        Douglas, "_discover_local_check_commands", lambda self: [semgrep_command]
    )

    _patch_subprocess_for_command(
        monkeypatch,
        semgrep_command,
        returncode=2,
        stderr=(
            "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='semgrep.dev', "
            "port=443): Max retries exceeded with url: /api/... (Caused by ProxyError('Cannot "
            "connect to proxy.'))"
        ),
    )

    success, logs = douglas._run_local_checks()

    assert success is True
    assert "Semgrep command failed" in logs

    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    assert history_path.exists()
    events = [
        json.loads(line)["event"]
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "local_checks_skip" in events


def test_semgrep_forbidden_failure_is_skipped(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{"name": "push"}], exit_conditions=[])

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: StaticProvider())
    douglas = Douglas(config_path)

    semgrep_command = ["semgrep", "--config", "auto"]
    monkeypatch.setattr(
        Douglas, "_discover_local_check_commands", lambda self: [semgrep_command]
    )

    _patch_subprocess_for_command(
        monkeypatch,
        semgrep_command,
        returncode=2,
        stderr=(
            "HTTPError: 403 Client Error: Forbidden for url: "
            "https://semgrep.dev/api/public/rules"
        ),
    )

    success, logs = douglas._run_local_checks()

    assert success is True
    assert "Semgrep command failed" in logs

    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    assert history_path.exists()
    events = [
        json.loads(line)["event"]
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "local_checks_skip" in events


def test_semgrep_login_required_failure_is_skipped(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{"name": "push"}], exit_conditions=[])

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: StaticProvider())
    douglas = Douglas(config_path)

    semgrep_command = ["semgrep", "--config", "auto"]
    monkeypatch.setattr(
        Douglas, "_discover_local_check_commands", lambda self: [semgrep_command]
    )

    _patch_subprocess_for_command(
        monkeypatch,
        semgrep_command,
        returncode=2,
        stderr=(
            "error: --config auto requires authentication. Run 'semgrep login' or "
            "set the SEMGREP_APP_TOKEN environment variable."
        ),
    )

    success, logs = douglas._run_local_checks()

    assert success is True
    assert "semgrep login" in logs.lower()

    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    assert history_path.exists()
    events = [
        json.loads(line)["event"]
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "local_checks_skip" in events


def test_log_excerpt_limit_respects_configuration(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{"name": "push"}], exit_conditions=[])

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data.setdefault("history", {})["max_log_excerpt_length"] = 120
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8"
    )

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: StaticProvider())
    douglas = Douglas(config_path)

    sample = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 characters
    excerpt = douglas._tail_log_excerpt(sample, limit=400)

    assert len(excerpt) == 120
    assert excerpt == sample[-120:]
