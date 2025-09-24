"""Codex provider that shells out to the Codex CLI when available."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from douglas.providers.openai_provider import OpenAIProvider


class CodexProvider(OpenAIProvider):
    """LLM provider that targets the (legacy) OpenAI Codex models."""

    DEFAULT_MODEL = "code-davinci-002"
    _CLI_EXECUTABLE_ENV = "CODEX_CLI_PATH"
    _CLI_DEFAULT_EXECUTABLE = "codex"
    _CLI_AUTH_FILE_ENV = "CODEX_AUTH_FILE"
    _CLI_HOME_ENV = "CODEX_HOME"
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
        resolved_cli_path = self._resolve_cli_path(cli_path)
        cli_token = None
        if api_key is None and resolved_cli_path:
            cli_token = self._resolve_cli_token(resolved_cli_path)

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
        self._cli_timeout_seconds = int(os.getenv("CODEX_CLI_TIMEOUT", "300"))

    def _resolve_cli_path(self, cli_path: Optional[str]) -> Optional[str]:
        explicit = cli_path or os.getenv(self._CLI_EXECUTABLE_ENV)
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return shutil.which(self._CLI_DEFAULT_EXECUTABLE)

    def generate_code(self, prompt: str) -> str:
        cli_result = self._invoke_codex_cli(prompt)
        if cli_result:
            return cli_result

        return super().generate_code(prompt)

    def _invoke_codex_cli(self, prompt: str) -> Optional[str]:
        if not self._cli_path:
            return None

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as handle:
            last_message_path = Path(handle.name)

        try:
            command = [
                self._cli_path,
                "exec",
                "-",
                "--output-last-message",
                str(last_message_path),
                "--color",
                "never",
                "--full-auto",
                "--skip-git-repo-check",
            ]

            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self._cli_timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            last_message_path.unlink(missing_ok=True)
            return None
        except subprocess.TimeoutExpired:
            print("Warning: Codex CLI timed out; falling back to OpenAI provider.")
            last_message_path.unlink(missing_ok=True)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"Warning: Codex CLI invocation failed ({exc}). Falling back to OpenAI provider."
            )
            last_message_path.unlink(missing_ok=True)
            return None

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if stderr:
                print(
                    "Warning: Codex CLI exited with code "
                    f"{result.returncode}: {stderr}. Falling back to OpenAI provider."
                )
            else:
                print(
                    "Warning: Codex CLI exited with a non-zero status; falling back to "
                    "OpenAI provider."
                )
            last_message_path.unlink(missing_ok=True)
            return None

        try:
            message = last_message_path.read_text(encoding="utf-8").strip()
        except OSError:
            message = ""
        finally:
            last_message_path.unlink(missing_ok=True)

        if message:
            return message

        stdout = (result.stdout or "").strip()
        if stdout:
            return stdout

        stderr = (result.stderr or "").strip()
        if stderr:
            return stderr

        return None

    def _resolve_cli_token(self, cli_path: Optional[str]) -> Optional[str]:
        if not cli_path:
            return None

        commands = ([cli_path, "auth", "token"], [cli_path, "token"])
        failures: list[str] = []

        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                return None
            except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
                message = self._summarise_cli_error(exc)
                if message:
                    failures.append(f"`{' '.join(command)}` ({message})")
                continue

            token = self._parse_cli_token(result.stdout, result.stderr)
            if token:
                return token

        token = self._load_cli_cached_token()
        if token:
            return token

        if failures:
            last_failure = failures[-1]
            print(
                "Warning: Failed to retrieve Codex token via CLI; "
                f"falling back to environment variables. Details: {last_failure}"
            )
        else:
            print(
                "Warning: Codex CLI returned no token; falling back to environment "
                "variables."
            )

        return None

    @staticmethod
    def _summarise_cli_error(exc: subprocess.CalledProcessError) -> str:
        output = (exc.stderr or exc.stdout or "").strip()
        if not output:
            return f"exit code {exc.returncode}"
        first_line = output.splitlines()[0].strip()
        return first_line or f"exit code {exc.returncode}"

    def _load_cli_cached_token(self) -> Optional[str]:
        for path in self._candidate_auth_paths():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except FileNotFoundError:
                continue
            except OSError:  # pragma: no cover - defensive
                continue
            except json.JSONDecodeError:  # pragma: no cover - defensive
                continue

            token = self._extract_token_from_mapping(payload)
            if token:
                return token

        return None

    def _candidate_auth_paths(self) -> Iterable[Path]:
        override = os.getenv(self._CLI_AUTH_FILE_ENV)
        if override:
            yield Path(override).expanduser()

        home_override = os.getenv(self._CLI_HOME_ENV)
        if home_override:
            yield Path(home_override).expanduser() / "auth.json"

        yield Path.home() / ".codex" / "auth.json"

    def _extract_token_from_mapping(self, payload: object) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        token = self._extract_token_from_dict(payload)
        if token:
            return token

        tokens_section = payload.get("tokens")
        if isinstance(tokens_section, dict):
            return self._extract_token_from_dict(tokens_section)

        return None

    def _extract_token_from_dict(self, values: dict) -> Optional[str]:
        for key in self._TOKEN_KEYS:
            value = values.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if self._looks_like_token(candidate):
                    return candidate
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
