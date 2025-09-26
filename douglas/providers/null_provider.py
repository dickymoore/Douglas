"""Null provider that records skipped steps without editing source files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from douglas.providers.llm_provider import LLMProvider


@dataclass
class _NullContext:
    project_root: Path
    agent_label: str
    step_name: str

    def state_path(self) -> Path:
        directory = self.project_root / ".douglas" / "state" / "null_provider"
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{self.agent_label.lower()}_{self.step_name.lower()}.json"
        return directory / filename


class _NullProviderContextual(LLMProvider):
    def __init__(self, context: _NullContext) -> None:
        self._context = context

    def generate_code(self, prompt: str) -> str:
        payload: Dict[str, object] = {
            "status": "skipped",
            "reason": "skipped by null provider",
            "agent": self._context.agent_label,
            "step": self._context.step_name,
            "timestamp": "0001-01-01T00:00:00Z",
        }
        path = self._context.state_path()
        relative = path.relative_to(self._context.project_root)
        return "```" + relative.as_posix() + "\n" + json.dumps(payload, indent=2) + "\n```"


class NullProvider(LLMProvider):
    provider_id = "null"

    def __init__(self, project_root: Path) -> None:
        self._project_root = Path(project_root)
        self._contexts: Dict[Tuple[str, str], _NullProviderContextual] = {}

    def with_context(self, agent_label: str, step_name: str) -> LLMProvider:
        key = (agent_label or "agent", step_name or "step")
        if key not in self._contexts:
            context = _NullContext(
                project_root=self._project_root,
                agent_label=agent_label,
                step_name=step_name,
            )
            self._contexts[key] = _NullProviderContextual(context)
        return self._contexts[key]

    def generate_code(self, prompt: str) -> str:
        raise RuntimeError("NullProvider requires contextualisation via with_context().")

