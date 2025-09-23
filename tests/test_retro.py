import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.pipelines import retro


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompt: Optional[str] = None

    def generate_code(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_run_retro_creates_outputs(tmp_path):
    sprint_root = tmp_path / "ai-inbox" / "sprints" / "sprint-3"
    role_dir = sprint_root / "roles" / "developer"
    role_dir.mkdir(parents=True, exist_ok=True)
    (role_dir / "summary.md").write_text(
        "Implemented analytics dashboard and resolved alert noise.", encoding="utf-8"
    )
    (role_dir / "handoffs.md").write_text(
        "Coordinate with QA on regression coverage next sprint.", encoding="utf-8"
    )

    config = {
        "paths": {"sprint_prefix": "sprint-"},
        "retro": {
            "outputs": ["role_instructions", "pre_feature_backlog"],
            "backlog_file": "ai-inbox/backlog/pre-features.yaml",
        },
    }

    response_payload = {
        "wins": ["Delivered analytics visibility for leadership"],
        "pain_points": ["Alert fatigue slowed response times"],
        "risks": ["Data pipeline failures could recur"],
        "role_instructions": {
            "Developer": [
                "Refine alert thresholds based on new telemetry.",
                "Pair with QA on regression coverage plan.",
            ],
            "QA": [],
        },
        "pre_feature_items": [
            {
                "title": "Automate alert triage dashboards",
                "rationale": "Reduce manual review of noisy alerts",
                "suggested_owner": "Developer",
                "acceptance_hints": [
                    "Dashboard summarizes alert categories",
                    "Includes filters for severity and service",
                ],
            }
        ],
    }

    stub_llm = StubLLM(json.dumps(response_payload))

    context = {
        "project_root": tmp_path,
        "config": config,
        "sprint_manager": SimpleNamespace(sprint_index=3),
        "llm": stub_llm,
    }

    result = retro.run_retro(context)

    developer_instructions = sprint_root / "roles" / "developer" / "instructions.md"
    assert developer_instructions.exists()
    developer_text = developer_instructions.read_text(encoding="utf-8")
    assert "Retro Actions for Developer" in developer_text
    assert "Refine alert thresholds" in developer_text
    assert "## Key Wins" in developer_text

    qa_instructions = sprint_root / "roles" / "qa" / "instructions.md"
    assert qa_instructions.exists()
    qa_text = qa_instructions.read_text(encoding="utf-8")
    assert "_No action items recorded._" in qa_text

    backlog_path = tmp_path / "ai-inbox" / "backlog" / "pre-features.yaml"
    assert backlog_path.exists()
    backlog_entries = yaml.safe_load(backlog_path.read_text(encoding="utf-8"))
    assert backlog_entries[0]["id"] == "PREF-3-1"
    assert backlog_entries[0]["originated_from"] == ["sprint-3", "retro"]
    assert (
        "Dashboard summarizes alert categories"
        in backlog_entries[0]["acceptance_hints"][0]
    )

    assert result.sprint_folder == "sprint-3"
    assert result.instructions["developer"].resolve() == developer_instructions
    assert "qa" in result.instructions
    assert result.backlog_entries[0]["title"] == "Automate alert triage dashboards"

    assert "Implemented analytics dashboard" in stub_llm.prompt
    assert "Coordinate with QA" in stub_llm.prompt
