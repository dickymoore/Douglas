import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.core import Douglas
from douglas.pipelines import test as testpipe


def _write_config(tmp_path: Path) -> Path:
    config_text = """
project:
  name: 'AgentJournal'
ai:
  provider: 'openai'
paths:
  inbox_dir: 'ai-inbox'
  sprint_prefix: 'sprint-'
loop:
  steps: []
"""
    config_path = tmp_path / "douglas.yaml"
    config_path.write_text(config_text.strip() + "\n", encoding="utf-8")
    return config_path


class StubProvider:
    def __init__(self, response: str = "") -> None:
        self.response = response

    def generate_code(self, prompt: str) -> str:
        return self.response


def test_generate_step_appends_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    provider = StubProvider("diff --git a/foo b/foo\n")
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: provider)

    douglas = Douglas(config_path)
    monkeypatch.setattr(Douglas, "_build_generation_prompt", lambda self: "work")
    monkeypatch.setattr(Douglas, "_apply_llm_output", lambda self, _: {"src/app.py"})
    monkeypatch.setattr(Douglas, "_stage_changes", lambda self, paths: None)

    douglas._generate_impl()

    summary_path = tmp_path / "ai-inbox" / "sprints" / "sprint-1" / "roles" / "developer" / "summary.md"
    assert summary_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Applied generated updates to 1 file" in summary_text
    assert "src/app.py" in summary_text


def test_failed_tests_record_handoff_and_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: StubProvider())

    douglas = Douglas(config_path)

    def failing_tests() -> None:
        raise SystemExit(1)

    monkeypatch.setattr(testpipe, "run_tests", failing_tests)

    decision = douglas.cadence_manager.evaluate_step("test", {"name": "test"})

    with pytest.raises(SystemExit):
        douglas._execute_step("test", {"name": "test"}, decision)

    summary_path = tmp_path / "ai-inbox" / "sprints" / "sprint-1" / "roles" / "tester" / "summary.md"
    handoff_path = tmp_path / "ai-inbox" / "sprints" / "sprint-1" / "roles" / "tester" / "handoffs.md"

    assert summary_path.exists()
    assert handoff_path.exists()

    summary_text = summary_path.read_text(encoding="utf-8")
    handoff_lines = handoff_path.read_text(encoding="utf-8").strip().splitlines()

    assert "Test suite failed" in summary_text
    assert "Handoffs Raised" in summary_text

    assert handoff_lines[0].startswith("## HANDOFF-")
    handoff_id = handoff_lines[0][3:].strip()
    assert handoff_id in summary_text
    assert any("to_role: developer" in line for line in handoff_lines)
    assert any("blocking: true" in line for line in handoff_lines)
