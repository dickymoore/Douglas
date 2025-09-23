import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.core import Douglas


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
        "project": {"name": "CITracking", "language": "python"},
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

    (tmp_path / "README.md").write_text("Initial content\n", encoding="utf-8")

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


def _force_gh_available(monkeypatch):
    original = shutil.which

    def fake_which(command: str):
        if command == "gh":
            return "/usr/bin/gh"
        return original(command)

    monkeypatch.setattr(shutil, "which", fake_which)


def _read_summary(tmp_path: Path) -> str:
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
    return summary_path.read_text(encoding="utf-8")


def _read_handoffs(tmp_path: Path) -> str:
    handoffs_path = (
        tmp_path
        / "ai-inbox"
        / "sprints"
        / "sprint-1"
        / "roles"
        / "devops"
        / "handoffs.md"
    )
    assert handoffs_path.exists()
    return handoffs_path.read_text(encoding="utf-8")


def test_monitor_ci_records_failure(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{"name": "commit"}], exit_conditions=[])
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())
    douglas = Douglas(config_path)

    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    gh_runs = json.dumps(
        [
            {
                "databaseId": 42,
                "headSha": commit_sha,
                "status": "completed",
                "conclusion": "failure",
                "url": "https://ci.example.test/run",
            }
        ]
    )

    original_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command[:3] == ["gh", "run", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout=gh_runs, stderr="")
        if command[:3] == ["gh", "run", "view"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="CI failure log\n", stderr=""
            )
        return original_run(command, *args, **kwargs)

    _force_gh_available(monkeypatch)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = douglas._monitor_ci(source_step="pr", max_attempts=1, poll_interval=0)
    assert result is False
    assert douglas._ci_status == "failure"

    bug_file = tmp_path / "ai-inbox" / "bugs.md"
    assert bug_file.exists()
    bug_contents = bug_file.read_text(encoding="utf-8")
    assert "CI run 42 failed with conclusion failure." in bug_contents
    assert "CI failure log" in bug_contents

    summary_text = _read_summary(tmp_path)
    assert "CI checks failed for the latest release" in summary_text
    assert "**status**: failed" in summary_text
    assert "**run id**: 42" in summary_text
    assert "**bug id**: FEAT-BUG-" in summary_text
    assert "**Handoffs Raised**" in summary_text

    handoff_text = _read_handoffs(tmp_path)
    assert "Investigate failing CI run" in handoff_text
    assert "run_id: 42" in handoff_text
    assert "conclusion: failure" in handoff_text

    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    history_entries = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [entry["event"] for entry in history_entries]
    assert "ci_fail" in events
    assert "bug_reported" in events


def test_monitor_ci_records_success(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{"name": "commit"}], exit_conditions=[])
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())
    douglas = Douglas(config_path)

    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    gh_runs = json.dumps(
        [
            {
                "databaseId": 101,
                "headSha": commit_sha,
                "status": "completed",
                "conclusion": "success",
                "url": "https://ci.example.test/run/101",
            }
        ]
    )

    original_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command[:3] == ["gh", "run", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout=gh_runs, stderr="")
        if command[:3] == ["gh", "run", "view"]:
            raise AssertionError("gh run view should not be called when CI succeeds")
        return original_run(command, *args, **kwargs)

    _force_gh_available(monkeypatch)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = douglas._monitor_ci(source_step="pr", max_attempts=1, poll_interval=0)
    assert result is True
    assert douglas._ci_status == "success"

    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    history_entries = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(entry["event"] == "ci_pass" for entry in history_entries)
    assert not (tmp_path / "ai-inbox" / "bugs.md").exists()

    summary_text = _read_summary(tmp_path)
    assert "CI checks succeeded for the latest release." in summary_text
    assert "**status**: success" in summary_text
    assert "**run id**: 101" in summary_text
