from pathlib import Path

import pytest
import yaml

from douglas.controls import run_state
from douglas.core import Douglas


class DummyProvider:
    def generate_code(self, prompt: str) -> str:  # pragma: no cover - simple stub
        return ""


def _write_config(tmp_path: Path, *, steps: list[dict]) -> Path:
    config = {
        "project": {"name": "RunState", "language": "python"},
        "ai": {"provider": "openai"},
        "loop": {"steps": steps},
        "paths": {"run_state_file": "run-state.txt"},
    }
    config_path = tmp_path / "douglas.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_read_run_state_defaults_to_continue(tmp_path):
    missing_path = tmp_path / "run-state.txt"
    state = run_state.read_run_state(missing_path)
    assert state is run_state.RunState.CONTINUE


def test_run_loop_hard_stop_at_start(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, steps=[])
    run_state_path = tmp_path / "run-state.txt"
    run_state_path.write_text("HARD_STOP", encoding="utf-8")

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: DummyProvider())
    monkeypatch.setattr(Douglas, "_commit_if_needed", lambda self: (False, None))

    douglas = Douglas(config_path)
    with pytest.raises(SystemExit) as exc:
        douglas.run_loop()

    assert exc.value.code == 1


def test_run_loop_soft_stop_exits_after_completion(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, steps=[])
    run_state_path = tmp_path / "run-state.txt"
    run_state_path.write_text("SOFT_STOP", encoding="utf-8")

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: DummyProvider())
    monkeypatch.setattr(Douglas, "_commit_if_needed", lambda self: (False, None))

    douglas = Douglas(config_path)
    with pytest.raises(SystemExit) as exc:
        douglas.run_loop()

    assert exc.value.code == 0


def test_agent_wrapper_detects_hard_stop_mid_run(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, steps=[{"name": "generate"}])
    run_state_path = tmp_path / "run-state.txt"
    run_state_path.write_text("CONTINUE", encoding="utf-8")

    class HardStopProvider:
        def __init__(self, path: Path):
            self.path = path
            self.calls = 0

        def generate_code(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                self.path.write_text("HARD_STOP", encoding="utf-8")
            return ""

    provider = HardStopProvider(run_state_path)

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: provider)
    monkeypatch.setattr(Douglas, "_commit_if_needed", lambda self: (False, None))
    monkeypatch.setattr(Douglas, "_build_generation_prompt", lambda self: "prompt")
    monkeypatch.setattr(Douglas, "_apply_llm_output", lambda self, _: [])

    douglas = Douglas(config_path)

    with pytest.raises(SystemExit) as exc:
        douglas.run_loop()

    assert exc.value.code == 1
    assert provider.calls == 1
