"""GitHub Copilot provider stub that emits deterministic placeholder responses."""

from __future__ import annotations

import os
from typing import Optional

from douglas.logging_utils import get_logger
from douglas.providers.llm_provider import LLMProvider


logger = get_logger(__name__)


class CopilotProvider(LLMProvider):
    """Stubbed provider for GitHub Copilot.

    GitHub Copilot does not currently expose a stable public API. This provider
    exists so Douglas can be configured with Copilot-style roles while falling
    back to deterministic placeholder output. When official APIs become
    available the implementation can be swapped without impacting callers.
    """

    DEFAULT_MODEL = "gpt-4o-copilot"

    def __init__(
        self,
        model_name: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.model = model_name or os.getenv("COPILOT_MODEL") or self.DEFAULT_MODEL
        self._token = token or os.getenv("COPILOT_TOKEN") or os.getenv("GITHUB_TOKEN")
        if not self._token:
            logger.warning(
                "GitHub Copilot token not configured. Falling back to Copilot stub output."
            )

    def generate_code(self, prompt: str) -> str:
        return self._fallback(prompt)

    def _fallback(self, prompt: str) -> str:  # pragma: no cover - log output only
        logger.warning(
            "GitHub Copilot integration is stubbed; returning placeholder output."
        )
        logger.debug("Copilot fallback prompt preview:\n%s", prompt[:200])
        return "# GitHub Copilot API unavailable; no code generated."
