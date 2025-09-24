"""Codex provider that prefers the Codex CLI for authentication."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional

from douglas.providers.openai_provider import OpenAIProvider


class CodexProvider(OpenAIProvider):
    """LLM provider that targets the (legacy) OpenAI Codex models."""

    DEFAULT_MODEL = "code-davinci-002"
    _CLI_EXECUTABLE_ENV = "CODEX_CLI_PATH"
    _CLI_DEFAULT_EXECUTABLE = "codex"
    _TOKEN_KEYS = ("token", "access_token", "api_key")
    _TOKEN_CHARACTERS = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cli_path: Optional[str] = None,
    ) -> None:
        cli_token = None
        if api_key is None:
            resolved_cli_path = self._resolve_cli_path(cli_path)
            cli_token = self._resolve_cli_token(resolved_cli_path)
        else:
            resolved_cli_path = None

        resolved_model = (
            model_name
            or os.getenv("OPENAI_CODEX_MODEL")
            or os.getenv("OPENAI_MODEL")
            or self.DEFAULT_MODEL
        )

        super().__init__(
            model_name=resolved_model,
            api_key=api_key or cli_token,
            base_url=base_url,
        )

        self._cli_path = resolved_cli_path
        self._cli_token = cli_token

    def _resolve_cli_path(self, cli_path: Optional[str]) -> Optional[str]:
        explicit = cli_path or os.getenv(self._CLI_EXECUTABLE_ENV)
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return shutil.which(self._CLI_DEFAULT_EXECUTABLE)

    def _resolve_cli_token(self, cli_path: Optional[str]) -> Optional[str]:
        if not cli_path:
            return None

        try:
            result = subprocess.run(
                [cli_path, "auth", "token"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None
        except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
            message = exc.stderr.strip() or exc.stdout.strip()
            if message:
                print(
                    "Warning: Failed to retrieve Codex token via CLI; "
                    f"falling back to environment variables. Details: {message}"
                )
            return None

        token = self._parse_cli_token(result.stdout, result.stderr)
        if token:
            return token

        print(
            "Warning: Codex CLI returned no token; falling back to environment "
            "variables."
        )
        return None

    @classmethod
    def _parse_cli_token(cls, stdout: str, stderr: str) -> Optional[str]:
        for stream in (stdout, stderr):
            token = cls._parse_token_from_stream(stream)
            if token:
                return token
        return None

    @classmethod
    def _parse_token_from_stream(cls, stream: str) -> Optional[str]:
        if not stream:
            return None

        for raw_line in reversed(stream.splitlines()):
            token = cls._parse_token_from_line(raw_line)
            if token:
                return token
        return None

    @classmethod
    def _parse_token_from_line(cls, line: str) -> Optional[str]:
        stripped = line.strip()
        if not stripped:
            return None

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                for key in cls._TOKEN_KEYS:
                    value = payload.get(key)
                    if isinstance(value, str):
                        candidate = value.strip()
                        if cls._looks_like_token(candidate):
                            return candidate

        if ":" in stripped:
            prefix, _, remainder = stripped.partition(":")
            normalized_prefix = prefix.strip().lower().replace(" ", "_")
            if normalized_prefix in cls._TOKEN_KEYS:
                candidate = remainder.strip()
                if cls._looks_like_token(candidate):
                    return candidate

        if cls._looks_like_token(stripped):
            return stripped

        return None

    @classmethod
    def _looks_like_token(cls, value: str) -> bool:
        if not value:
            return False
        return all(character in cls._TOKEN_CHARACTERS for character in value)

    def _fallback(self, prompt: str) -> str:  # pragma: no cover - log output only
        print("[CodexProvider] Falling back to placeholder output. Prompt preview:")
        print(prompt[:200])
        return "# Codex API unavailable; no code generated."
