import os
from collections.abc import Iterable, Mapping, Sequence
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
    # Limit recursion when parsing nested provider payloads. A depth of 32 easily
    # exceeds what OpenAI currently returns while still protecting against
    # accidental or maliciously deep structures.
    _MAX_NORMALIZATION_DEPTH = 32

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        resolved_model = model_name or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL
        self.model = str(resolved_model)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        )
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
                f"Warning: Failed to initialise OpenAI client via the modern SDK: {exc}"
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
                f"Warning: Failed to initialise OpenAI client via the legacy SDK: {exc}"
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

    def _normalize_response_content(
        self,
        content: Any,
        *,
        depth: int = 0,
        seen: Optional[set[int]] = None,
        reference_chain: Optional[list[Any]] = None,
    ) -> str:
        """Normalise nested response content into a string with cycle protection."""

        if content is None:
            return ""

        if depth > self._MAX_NORMALIZATION_DEPTH:
            return ""

        if seen is None:
            seen = set()

        if reference_chain is None:
            reference_chain = []

        if isinstance(content, str):
            stripped = content.strip()
            return stripped if stripped else ""

        if isinstance(content, (bytes, bytearray)):
            stripped = bytes(content).decode("utf-8", "ignore").strip()
            return stripped if stripped else ""

        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            obj_id = id(content)
            if obj_id in seen or any(obj is content for obj in reference_chain):
                return ""
            seen.add(obj_id)
            reference_chain.append(content)
            try:
                pieces: list[str] = []
                for item in content:
                    piece = self._normalize_response_content(
                        item,
                        depth=depth + 1,
                        seen=seen,
                        reference_chain=reference_chain,
                    )
                    if piece:
                        pieces.append(piece)
                if pieces:
                    return "\n".join(pieces).strip()
                return ""
            finally:
                seen.discard(obj_id)
                reference_chain.pop()

        return self._coerce_text_value(
            content,
            depth=depth + 1,
            seen=seen,
            reference_chain=reference_chain,
        )

    def _coerce_text_value(
        self,
        value: Any,
        *,
        depth: int = 0,
        seen: Optional[set[int]] = None,
        reference_chain: Optional[list[Any]] = None,
    ) -> str:
        """Attempt to coerce nested response text blocks into a string."""

        if value is None:
            return ""

        if depth > self._MAX_NORMALIZATION_DEPTH:
            return ""

        if seen is None:
            seen = set()

        if reference_chain is None:
            reference_chain = []

        obj_id = id(value)
        if obj_id in seen or any(obj is value for obj in reference_chain):
            return ""

        seen.add(obj_id)
        reference_chain.append(value)
        try:
            if isinstance(value, str):
                stripped = value.strip()
                return stripped if stripped else ""

            if isinstance(value, (bytes, bytearray)):
                stripped = bytes(value).decode("utf-8", "ignore").strip()
                return stripped if stripped else ""

            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                pieces: list[str] = []
                for item in value:
                    result = self._normalize_response_content(
                        item,
                        depth=depth + 1,
                        seen=seen,
                        reference_chain=reference_chain,
                    )
                    if result:
                        pieces.append(result)
                if pieces:
                    return "\n".join(pieces).strip()
                return ""

            if isinstance(value, Mapping):
                prioritized_keys = (
                    "output_text",
                    "text",
                    "value",
                    "content",
                    "string_value",
                )
                for key in prioritized_keys:
                    if key in value:
                        result = self._normalize_response_content(
                            value[key],
                            depth=depth + 1,
                            seen=seen,
                            reference_chain=reference_chain,
                        )
                        if result:
                            return result

                aggregated: list[str] = []
                for nested_value in value.values():
                    result = self._normalize_response_content(
                        nested_value,
                        depth=depth + 1,
                        seen=seen,
                        reference_chain=reference_chain,
                    )
                    if result:
                        aggregated.append(result)
                if aggregated:
                    return "\n".join(aggregated).strip()
                return ""

            for attr in (
                "output_text",
                "text",
                "value",
                "content",
                "string_value",
            ):
                if hasattr(value, attr):
                    result = self._normalize_response_content(
                        getattr(value, attr),
                        depth=depth + 1,
                        seen=seen,
                        reference_chain=reference_chain,
                    )
                    if result:
                        return result

            if isinstance(value, Iterable) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                pieces: list[str] = []
                for item in value:
                    result = self._normalize_response_content(
                        item,
                        depth=depth + 1,
                        seen=seen,
                        reference_chain=reference_chain,
                    )
                    if result:
                        pieces.append(result)
                if pieces:
                    return "\n".join(pieces).strip()

            text = str(value).strip()
            return text if text else ""
        finally:
            seen.discard(obj_id)
            reference_chain.pop()

    def _extract_responses_text(self, response: Any) -> str:
        text = self._normalize_response_content(getattr(response, "output_text", None))
        if text:
            return text

        attribute_output = self._normalize_response_content(
            getattr(response, "output", None)
        )
        if attribute_output:
            return attribute_output

        if isinstance(response, dict):
            dict_text = self._normalize_response_content(response.get("output_text"))
            if dict_text:
                return dict_text

            dict_output = self._normalize_response_content(response.get("output"))
            if dict_output:
                return dict_output

        choices = getattr(response, "choices", None)
        if choices:
            message = (
                choices[0].get("message")
                if isinstance(choices[0], dict)
                else getattr(choices[0], "message", None)
            )
            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            normalized = self._normalize_response_content(content)
            if normalized:
                return normalized

        if isinstance(response, dict):
            dict_choices = response.get("choices") or []
            if dict_choices:
                message = dict_choices[0].get("message", {})
                normalized = self._normalize_response_content(message.get("content"))
                if normalized:
                    return normalized

        payload = None
        if hasattr(response, "model_dump"):
            try:
                payload = response.model_dump()
            except Exception:  # pragma: no cover - defensive fallback
                payload = None
        if isinstance(payload, dict):
            payload_text = self._normalize_response_content(payload.get("output_text"))
            if payload_text:
                return payload_text

            payload_choices = payload.get("choices") or []
            if payload_choices:
                message = payload_choices[0].get("message", {})
                normalized = self._normalize_response_content(message.get("content"))
                if normalized:
                    return normalized

            payload_output = self._normalize_response_content(payload.get("output"))
            if payload_output:
                return payload_output

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
