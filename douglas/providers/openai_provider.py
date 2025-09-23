import os
from typing import Any, Optional

from douglas.providers.llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """Concrete LLM provider backed by the OpenAI Python SDK.

    The provider lazily initialises the OpenAI client using the modern SDK when
    available (``from openai import OpenAI``) and falls back to the legacy
    ``openai`` module interface otherwise. Missing credentials or a missing SDK
    trigger a graceful local stub so that Douglas remains usable in offline
    scenarios and during tests.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model_name: Optional[str] = None):
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        self.model = model_name or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL
        self._client: Optional[Any] = None
        self._use_responses_api = False

        if not self._api_key:
            print("Warning: OPENAI_API_KEY not set. Falling back to local stub output.")
            return

        if self._initialize_modern_client():
            return

        self._initialize_legacy_client()

    def _initialize_modern_client(self) -> bool:
        """Attempt to configure the post-1.0 OpenAI SDK."""

        try:
            from openai import OpenAI
        except ImportError:
            return False

        try:
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = OpenAI(**client_kwargs)
            self._use_responses_api = True
            return True
        except Exception as exc:  # pragma: no cover - defensive against SDK issues
            print(
                "Warning: Failed to initialise OpenAI client via the modern SDK: "
                f"{exc}"
            )
            self._client = None
            return False

    def _initialize_legacy_client(self) -> None:
        """Fallback initialisation for the legacy ``openai`` module."""

        try:
            import openai
        except ImportError:
            print(
                "Warning: The 'openai' package is not installed. Install it with "
                "'pip install openai' to enable OpenAI integration."
            )
            self._client = None
            return

        try:
            openai.api_key = self._api_key
            if self._base_url:
                openai.api_base = self._base_url
            self._client = openai
            self._use_responses_api = False
        except Exception as exc:  # pragma: no cover - defensive against SDK issues
            print(
                "Warning: Failed to initialise OpenAI client via the legacy SDK: "
                f"{exc}"
            )
            self._client = None

    def generate_code(self, prompt: str) -> str:
        if not self._client:
            return self._fallback(prompt)

        try:
            if self._use_responses_api:
                text = self._call_with_modern_client(prompt)
            else:
                text = self._call_with_legacy_client(prompt)
        except Exception as exc:
            print(
                f"Warning: OpenAI request failed ({exc}). Falling back to stub output."
            )
            return self._fallback(prompt)

        if not text:
            print(
                "Warning: Received empty response from OpenAI. Falling back to stub output."
            )
            return self._fallback(prompt)

        return text

    def _call_with_modern_client(self, prompt: str) -> str:
        if not self._client:
            return ""

        responses_api = getattr(self._client, "responses", None)
        if responses_api is not None:
            try:
                response = responses_api.create(model=self.model, input=prompt)
            except Exception:  # pragma: no cover - defensive fallback
                pass
            else:
                text = self._extract_responses_text(response)
                if text:
                    return text

        chat_interface = getattr(self._client, "chat", None)
        if chat_interface is not None and hasattr(chat_interface, "completions"):
            try:
                response = chat_interface.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception:  # pragma: no cover - defensive fallback
                pass
            else:
                text = self._extract_chat_completion_text(response)
                if text:
                    return text

        return ""

    def _call_with_legacy_client(self, prompt: str) -> str:
        if not self._client:
            return ""

        try:
            response = self._client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:  # pragma: no cover - defensive fallback
            return ""

        return self._extract_chat_completion_text(response)

    def _normalize_response_content(self, content: Any) -> str:
        """Normalize response message content into a usable text string."""

        if isinstance(content, str):
            stripped = content.strip()
            return stripped if stripped else ""

        if isinstance(content, list):
            segments: list[str] = []
            for block in content:
                text_value: Any
                if isinstance(block, str):
                    text_value = block
                elif isinstance(block, dict):
                    text_value = block.get("text")
                    if text_value is None:
                        text_value = block.get("value")
                    if text_value is None:
                        text_value = block.get("content")
                    if text_value is None:
                        text_value = block.get("output_text")
                    if text_value is None:
                        text_value = block
                else:
                    text_value = getattr(block, "text", None)
                    if text_value is None:
                        text_value = getattr(block, "value", None)
                    if text_value is None:
                        text_value = getattr(block, "content", None)
                    if text_value is None:
                        text_value = getattr(block, "output_text", None)
                    if text_value is None:
                        continue

                segment = self._coerce_text_value(text_value)
                if segment:
                    segments.append(segment)

            if segments:
                return "\n".join(segments).strip()

            return ""

        if isinstance(content, dict):
            return self._coerce_text_value(content)

        return ""

    def _coerce_text_value(self, value: Any) -> str:
        """Attempt to coerce nested response text blocks into a string."""

        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else ""

        if isinstance(value, list):
            return self._normalize_response_content(value)

        if isinstance(value, dict):
            for key in ("value", "text", "content", "output_text"):
                if key in value:
                    result = self._coerce_text_value(value[key])
                    if result:
                        return result
            return ""

        if value is None:
            return ""

        for attr in ("value", "text", "content", "output_text"):
            if hasattr(value, attr):
                result = self._coerce_text_value(getattr(value, attr))
                if result:
                    return result

        text = str(value).strip()
        return text if text else ""

    def _extract_responses_text(self, response: Any) -> str:
        text = self._normalize_response_content(getattr(response, "output_text", None))
        if text:
            return text

        text = self._normalize_response_content(getattr(response, "output", None))
        if text:
            return text

        # Try to extract from attribute-based response (OpenAI SDK object)
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            message = (
                choices[0].get("message")
                if isinstance(choices[0], dict)
                else getattr(choices[0], "message", None)
            )
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            text = self._normalize_response_content(content)
            if text:
                return text

        # Try to extract from dict-based response
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content")
                text = self._normalize_response_content(content)
                if text:
                    return text

            text = self._normalize_response_content(response.get("output"))
            if text:
                return text

        # Defensive fallback: try to extract from model_dump if available
        payload = None
        if hasattr(response, "model_dump"):
            try:
                payload = response.model_dump()
            except Exception:  # pragma: no cover - defensive fallback
                payload = None
        if isinstance(payload, dict):
            choices = payload.get("choices") or []
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content")
                text = self._normalize_response_content(content)
                if text:
                    return text

            text = self._normalize_response_content(payload.get("output"))
            if text:
                return text

        return ""

    def _extract_chat_completion_text(self, response: Any) -> str:
        if hasattr(response, "choices"):
            try:
                choices = response.choices
            except Exception:  # pragma: no cover - defensive fallback
                choices = None
            if choices:
                message = (
                    choices[0].get("message")
                    if isinstance(choices[0], dict)
                    else getattr(choices[0], "message", None)
                )
                if isinstance(message, dict):
                    content = message.get("content")
                else:
                    content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content.strip()

        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()

        return ""

    def _fallback(self, prompt: str) -> str:
        print("[OpenAIProvider] Falling back to placeholder output. Prompt preview:")
        print(prompt[:200])
        return "# OpenAI API unavailable; no code generated."
