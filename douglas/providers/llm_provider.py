from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from douglas.domain.step_result import StepResult


class LLMProvider(ABC):
    @abstractmethod
    def generate_code(self, prompt: str) -> str:
        pass

    def generate_step_result(
        self,
        prompt: str,
        *,
        step_name: Optional[str] = None,
        agent: Optional[str] = None,
        role: Optional[str] = None,
        seed: Optional[int] = None,
        prompt_hash: Optional[str] = None,
        timestamps: Optional[Mapping[str, Any]] = None,
    ) -> StepResult:
        """Return a ``StepResult`` for the given prompt."""

        start = datetime.now(timezone.utc).isoformat()
        raw_output = self.generate_code(prompt)
        completed = datetime.now(timezone.utc).isoformat()
        default_timestamps = {
            "started_at": start,
            "completed_at": completed,
        }
        merged_timestamps = dict(default_timestamps)
        if timestamps:
            merged_timestamps.update(dict(timestamps))
        result = StepResult.ensure(
            raw_output,
            step_name=step_name,
            agent=agent,
            role=role,
            seed=seed,
            prompt_hash=prompt_hash,
            prompt=prompt,
            timestamps=merged_timestamps,
        )
        return result

    @staticmethod
    def create_provider(name: str, **options):
        normalized = (name or "").strip().lower()

        model = options.get("model") or options.get("model_name")
        api_key = options.get("api_key") or options.get("token")
        base_url = options.get("base_url") or options.get("api_base")
        project_root = options.get("project_root")
        if project_root is not None and not isinstance(project_root, Path):
            project_root = Path(project_root)
        seed_option = options.get("seed")
        seed = int(seed_option) if seed_option is not None else 0

        if normalized in {"openai", "gpt", "gpt-4", "gpt4"}:
            from douglas.providers.openai_provider import OpenAIProvider

            return OpenAIProvider(model_name=model, api_key=api_key, base_url=base_url)

        if normalized in {"codex", "openai-codex", "codex-openai", "code-davinci"}:
            from douglas.providers.codex_provider import CodexProvider

            return CodexProvider(model_name=model, api_key=api_key, base_url=base_url)

        if normalized in {"claude", "claude_code", "claude-code", "anthropic"}:
            from douglas.providers.claude_code_provider import ClaudeCodeProvider

            return ClaudeCodeProvider(model_name=model, api_key=api_key)

        if normalized in {"gemini", "google", "google-ai", "googleai"}:
            from douglas.providers.gemini_provider import GeminiProvider

            return GeminiProvider(model_name=model, api_key=api_key)

        if normalized in {"copilot", "github-copilot", "github"}:
            from douglas.providers.copilot_provider import CopilotProvider

            token = options.get("token") or api_key
            return CopilotProvider(model_name=model, token=token)

        if normalized in {"mock", "deterministic_mock"}:
            from douglas.providers.mock_provider import DeterministicMockProvider

            if project_root is None:
                raise ValueError("Deterministic mock provider requires project_root.")
            return DeterministicMockProvider(project_root=project_root, seed=seed)

        if normalized == "replay":
            from douglas.providers.replay_provider import ReplayProvider

            store = options.get("cassette_store")
            project_fingerprint = options.get("project_fingerprint", "")
            if store is None:
                raise ValueError("Replay provider requires a cassette store instance.")
            if project_root is None:
                project_root = Path.cwd()
            provider_label = options.get("provider_name") or options.get("provider_label")
            return ReplayProvider(
                store=store,
                project_root=project_root,
                project_fingerprint=str(project_fingerprint),
                base_seed=seed,
                model_name=model,
                provider_name=str(provider_label) if provider_label else "replay",
            )

        if normalized == "null":
            from douglas.providers.null_provider import NullProvider

            if project_root is None:
                project_root = Path.cwd()
            return NullProvider(project_root=project_root)

        raise ValueError(f"Unsupported LLM provider: {name}")
