import json
import subprocess
from pathlib import Path

import yaml


from douglas.core import Douglas


def _init_repo(tmp_path: Path) -> Path:
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

    config = {
        "project": {"name": "History", "language": "python"},
        "ai": {"provider": "openai"},
        "loop": {"steps": [{"name": "commit"}]},
    }
    config_path = tmp_path / "douglas.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

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


def test_commit_adds_history_entry(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path)

    class Provider:
        def generate_code(self, prompt: str) -> str:
            return "feat: update docs"

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: Provider())

    douglas = Douglas(config_path)
    readme = tmp_path / "README.md"
    readme.write_text("Initial content\nMore docs\n", encoding="utf-8")

    douglas.run_loop()

    history_path = tmp_path / "ai-inbox" / "history.jsonl"
    assert history_path.exists()
    entries = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    commit_events = [entry for entry in entries if entry["event"] == "commit"]
    assert commit_events, "Expected at least one commit event in history."
    assert commit_events[-1]["message"] == "feat: update docs"
