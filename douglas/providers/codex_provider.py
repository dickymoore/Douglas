"""Codex provider built on top of the OpenAI provider implementation."""

from __future__ import annotations

import os
from typing import Optional

from douglas.providers.openai_provider import OpenAIProvider


class CodexProvider(OpenAIProvider):
    """LLM provider that targets the (legacy) OpenAI Codex models."""

    DEFAULT_MODEL = "code-davinci-002"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        resolved_model = (
            model_name
            or os.getenv("OPENAI_CODEX_MODEL")
            or os.getenv("OPENAI_MODEL")
            or self.DEFAULT_MODEL
        )
        super().__init__(
            model_name=resolved_model,
            api_key=api_key,
            base_url=base_url,
        )

    def _fallback(self, prompt: str) -> str:  # pragma: no cover - log output only
        print("[CodexProvider] Falling back to placeholder output. Prompt preview:")
        print(prompt[:200])
        return "# Codex API unavailable; no code generated."
