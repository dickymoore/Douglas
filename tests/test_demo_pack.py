import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.pipelines import demo


def _init_repo(tmp_path: Path) -> tuple[str, str]:
    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "config", "user.email", "demo@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Demo Bot"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    (tmp_path / "README.md").write_text("Initial content\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()

    subprocess.run(
        ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "chore: initial scaffold"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    first_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()

    (tmp_path / "src" / "main.py").write_text('print("demo")\n', encoding="utf-8")
    subprocess.run(
        ["git", "add", "src/main.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "feat: add demo entrypoint"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    head_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    return first_commit, head_commit


def test_write_demo_pack_creates_markdown(tmp_path):
    repo_root = tmp_path
    first_commit, head_commit = _init_repo(repo_root)

    history_path = repo_root / "ai-inbox" / "history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps({"event": "demo_pack_generated", "head_commit": first_commit})
        + "\n",
        encoding="utf-8",
    )

    sprint_root = repo_root / "ai-inbox" / "sprints" / "sprint-1"
    summary_path = sprint_root / "roles" / "developer" / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "Implemented ingestion pipeline and dashboards.\nValidated metrics with QA.",
        encoding="utf-8",
    )

    limitations_path = sprint_root / "limitations.md"
    limitations_path.write_text(
        "- Requires manual refresh for nightly jobs.", encoding="utf-8"
    )
    next_steps_path = sprint_root / "next_steps.md"
    next_steps_path.write_text(
        "- Automate dashboard refresh schedule.", encoding="utf-8"
    )

    logs_dir = repo_root / "ai-inbox" / "tests"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "pytest.log").write_text("1 passed in 0.01s\n", encoding="utf-8")

    config = {
        "loop": {
            "steps": [
                {"name": "lint"},
                {"name": "typecheck"},
                {"name": "test"},
                {"name": "review"},
                {"name": "demo"},
            ]
        },
        "paths": {
            "app_src": "src",
            "tests": "tests",
            "demos_dir": "demos",
            "sprint_prefix": "sprint-",
        },
        "demo": {
            "format": "md",
            "include": [
                "implemented_features",
                "how_to_run",
                "test_results",
                "limitations",
                "next_steps",
            ],
        },
    }

    context = {
        "project_root": repo_root,
        "config": config,
        "sprint_manager": SimpleNamespace(sprint_index=1),
        "history_path": history_path,
        "loop_outcomes": {"test": True},
    }

    metadata = demo.write_demo_pack(context)
    output_path = metadata.output_path

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "# Sprint 1 Demo" in content
    assert "Implemented ingestion pipeline and dashboards." in content
    assert "pytest" in content
    assert "Tests passed" in content
    assert "Requires manual refresh" in content
    assert "Automate dashboard refresh schedule." in content

    assert metadata.sprint_folder == "sprint-1"
    assert metadata.head_commit == head_commit
    assert any("feat: add demo entrypoint" in entry for entry in metadata.commits)
