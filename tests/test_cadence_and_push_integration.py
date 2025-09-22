import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

from douglas.core import Douglas
from douglas.pipelines import demo as demo_pipeline


class SequencedProvider:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate_code(self, prompt: str) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "chore: automated commit"


def _init_repo(
    tmp_path: Path,
    *,
    push_policy: str,
    demo_cadence: Optional[str],
) -> Path:
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "cadence@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Cadence Bot"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    steps = [
        {"name": "commit"},
        {"name": "demo"},
        {"name": "push"},
        {"name": "pr"},
    ]

    config_data: dict = {
        "project": {"name": "CadenceIntegration", "language": "python"},
        "ai": {"provider": "openai"},
        "loop": {"steps": steps},
        "push_policy": push_policy,
        "sprint": {"length_days": 2},
    }
    if demo_cadence is not None:
        config_data["cadence"] = {"ProductOwner": {"sprint_review": demo_cadence}}

    config_path = tmp_path / "douglas.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("Initial content\n", encoding="utf-8")

    subprocess.run(
        ["git", "add", "."],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "chore: initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    return config_path


@pytest.mark.parametrize(
    (
        "demo_override",
        "push_policy",
        "commit_messages",
        "expected_push_counts",
        "expected_demo_counts",
    ),
    [
        (None, "per_sprint", ["feat: add report", "feat: finalize report"], [0, 1], [0, 1]),
        ("daily", "per_sprint", ["feat: add report", "feat: finalize report"], [0, 1], [1, 2]),
        (None, "per_feature", ["feat: add report", "feat: finalize report"], [1, 2], [0, 1]),
        ("daily", "per_feature", ["feat: add report", "feat: finalize report"], [1, 2], [1, 2]),
        (None, "per_bug", ["fix: resolve crash", "fix: harden edge case"], [1, 2], [0, 1]),
        ("daily", "per_bug", ["fix: resolve crash", "fix: harden edge case"], [1, 2], [1, 2]),
    ],
    ids=
    [
        "per_sprint_default",
        "per_sprint_daily_demo",
        "per_feature_default_demo",
        "per_feature_daily_demo",
        "per_bug_default_demo",
        "per_bug_daily_demo",
    ],
)
def test_cadence_and_push_matrix(
    monkeypatch,
    tmp_path,
    demo_override,
    push_policy,
    commit_messages,
    expected_push_counts,
    expected_demo_counts,
):
    config_path = _init_repo(tmp_path, push_policy=push_policy, demo_cadence=demo_override)

    provider = SequencedProvider(commit_messages)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: provider)

    push_calls: list[str] = []
    pr_calls: list[str] = []
    demo_calls: list[int] = []

    def fake_push(self):
        push_calls.append("push")
        return True, "pushed"

    def fake_pr(self):
        pr_calls.append("pr")
        return True, {"url": "https://example.test/pr"}

    def fake_demo(context):
        demo_calls.append(context["sprint_manager"].current_iteration)
        return SimpleNamespace(
            output_path=context["project_root"] / "demos" / "demo.md",
            sprint_folder=f"sprint-{context['sprint_manager'].sprint_index}",
            head_commit=None,
            previous_commit=None,
            commits=[],
            generated_at="",
            format="md",
            as_event_payload=lambda: {},
        )

    monkeypatch.setattr(Douglas, "_run_git_push", fake_push)
    monkeypatch.setattr(Douglas, "_open_pull_request", fake_pr)
    monkeypatch.setattr(Douglas, "_monitor_ci", lambda self: None)
    monkeypatch.setattr(Douglas, "_run_local_checks", lambda self: (True, "ok"))
    monkeypatch.setattr(demo_pipeline, "write_demo_pack", fake_demo)

    douglas = Douglas(config_path)
    readme = tmp_path / "README.md"

    push_progress: list[int] = []
    pr_progress: list[int] = []
    demo_progress: list[int] = []

    for index in range(len(commit_messages)):
        readme.write_text(
            readme.read_text(encoding="utf-8") + f"Update {index}\n",
            encoding="utf-8",
        )
        douglas.run_loop()
        push_progress.append(len(push_calls))
        pr_progress.append(len(pr_calls))
        demo_progress.append(len(demo_calls))

    assert push_progress == expected_push_counts
    assert pr_progress == expected_push_counts
    assert demo_progress == expected_demo_counts
