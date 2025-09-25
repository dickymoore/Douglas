"""Google Gemini provider with graceful fallback behaviour."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional

from douglas.logging_utils import get_logger
from douglas.providers.llm_provider import LLMProvider


logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Provider implementation backed by the google-generativeai SDK."""

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        client: Optional[Any] = None,
    ) -> None:
        self.model = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or os.getenv("GOOGLE_MODEL")
            or self.DEFAULT_MODEL
        )
        self._api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        self._client: Optional[Any] = client

        if self._client is not None:
            return

        if not self._api_key:
            logger.warning(
                "GEMINI_API_KEY not set. Falling back to Gemini stub output."
            )
            return

        try:
            genai = importlib.import_module("google.generativeai")
        except ImportError:
            logger.warning(
                "The 'google-generativeai' package is not installed. Install it with 'pip install google-generativeai' to enable Gemini integration."
            )
            return

        try:
            configure = getattr(genai, "configure", None)
            if callable(configure):
                configure(api_key=self._api_key)
            client_factory = getattr(genai, "GenerativeModel", None)
            if client_factory is None:
                logger.warning(
                    "google-generativeai SDK does not expose GenerativeModel. Falling back to Gemini stub output."
                )
                return
            self._client = client_factory(self.model)
        except Exception as exc:  # pragma: no cover - defensive against SDK issues
            logger.warning("Failed to initialise Gemini client: %s", exc)
            self._client = None

    def generate_code(self, prompt: str) -> str:
        if not self._client:
            return self._fallback(prompt)

        try:
            response = self._client.generate_content(prompt)
        except Exception as exc:  # pragma: no cover - network/SDK failures
            logger.warning("Gemini request failed (%s). Falling back to stub output.", exc)
            return self._fallback(prompt)

        text = self._extract_text(response)
        if text:
            return text

        logger.warning("Received empty response from Gemini. Falling back to stub output.")
        return self._fallback(prompt)

    def _extract_text(self, response: Any) -> str:
        direct_text = getattr(response, "text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()

        content = getattr(response, "content", None)
        text = self._coerce_content(content)
        if text:
            return text

        candidates = getattr(response, "candidates", None)
        if candidates:
            return self._coerce_content(candidates)

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
            aggregated: List[str] = []
            for value in content.values():
                piece = self._coerce_content(value)
                if piece:
                    aggregated.append(piece)
            return "\n".join(aggregated).strip()
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            pieces: List[str] = []
            for item in content:
                # Gemini responses often nest content under parts with "text" attributes.
                if hasattr(item, "text"):
                    text = getattr(item, "text")
                    if isinstance(text, str) and text.strip():
                        pieces.append(text.strip())
                        continue
                piece = self._coerce_content(item)
                if piece:
                    pieces.append(piece)
            return "\n".join(pieces).strip()
        return ""

    def _fallback(self, prompt: str) -> str:  # pragma: no cover - log output only
        logger.warning(
            "Gemini provider fallback triggered; returning placeholder output."
        )
        logger.debug("Gemini fallback prompt preview:\n%s", prompt[:200])
        return "# Gemini API unavailable; no code generated."
