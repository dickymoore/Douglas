"""Claude Code provider that gracefully degrades to a local stub when unavailable."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional

from douglas.providers.llm_provider import LLMProvider


class ClaudeCodeProvider(LLMProvider):
    """Concrete provider for Anthropic's Claude Code models."""

    DEFAULT_MODEL = "claude-3-5-sonnet-20240620"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.model = (
            model_name
            or os.getenv("CLAUDE_CODE_MODEL")
            or os.getenv("ANTHROPIC_MODEL")
            or self.DEFAULT_MODEL
        )
        self._api_key = (
            api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        )
        self._max_tokens = max_tokens or 4096
        self._client: Optional[Any] = None

        if not self._api_key:
            print(
                "Warning: ANTHROPIC_API_KEY not set. Falling back to Claude Code stub output."
            )
            return

        try:
            anthropic = importlib.import_module("anthropic")
        except ImportError:
            print(
                "Warning: The 'anthropic' package is not installed. Install it with "
                "'pip install anthropic' to enable Claude Code integration."
            )
            return

        client_factory = getattr(anthropic, "Anthropic", None)
        if client_factory is None:
            print(
                "Warning: Anthropic SDK does not expose an Anthropic client. "
                "Falling back to Claude Code stub output."
            )
            return

        try:
            self._client = client_factory(api_key=self._api_key)
        except Exception as exc:  # pragma: no cover - defensive against SDK issues
            print(f"Warning: Failed to initialise Anthropic client: {exc}")
            self._client = None

    def generate_code(self, prompt: str) -> str:
        if not self._client:
            return self._fallback(prompt)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pragma: no cover - network/SDK failures
            print(
                f"Warning: Claude Code request failed ({exc}). Falling back to stub output."
            )
            return self._fallback(prompt)

        text = self._extract_text(response)
        if text:
            return text

        print(
            "Warning: Received empty response from Claude Code. Falling back to stub output."
        )
        return self._fallback(prompt)

    def _extract_text(self, response: Any) -> str:
        content = getattr(response, "content", None)
        text = self._coerce_content(content)
        if text:
            return text

        if isinstance(response, Mapping):
            return self._coerce_content(response.get("content"))

        return ""

    def _coerce_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            stripped = content.strip()
            return stripped if stripped else ""
        if isinstance(content, Mapping):
            if content.get("type") == "text":
                value = content.get("text")
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        return stripped
            collected: List[str] = []
            for value in content.values():
                piece = self._coerce_content(value)
                if piece:
                    collected.append(piece)
            return "\n".join(collected).strip()
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            items: List[str] = []
            for item in content:
                piece = self._coerce_content(item)
                if piece:
                    items.append(piece)
            return "\n".join(items).strip()
        return ""

    def _fallback(self, prompt: str) -> str:  # pragma: no cover - log output only
        print(
            "[ClaudeCodeProvider] Falling back to placeholder output. Prompt preview:"
        )
        print(prompt[:200])
        return "# Claude Code API unavailable; no code generated."
