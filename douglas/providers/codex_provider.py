"""Codex provider that shells out to the Codex CLI when available."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

from douglas.agents.locks import FileLockManager
from douglas.agents.workspace import AgentWorkspace
from douglas.logging_utils import get_logger
from douglas.providers.openai_provider import OpenAIProvider


logger = get_logger(__name__)


class CodexProvider(OpenAIProvider):
    """LLM provider that targets the (legacy) OpenAI Codex models."""

    DEFAULT_MODEL = "gpt-5-codex"
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
        self._cli_timeout_seconds = int(os.getenv("CODEX_CLI_TIMEOUT", "120"))
        self._cli_heartbeat_seconds = int(os.getenv("CODEX_CLI_HEARTBEAT", "10"))
        self._workspace_root = Path(
            os.getenv("DOUGLAS_WORKSPACE_ROOT", ".douglas/workspaces")
        )
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._lock_manager = FileLockManager(self._workspace_root.parent / "locks")
        self._workspaces: dict[str, AgentWorkspace] = {}

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

    def _workspace_for(self, agent_id: str) -> AgentWorkspace:
        if agent_id not in self._workspaces:
            self._workspaces[agent_id] = AgentWorkspace(
                agent_id,
                self._workspace_root / agent_id,
                self._lock_manager,
            )
        return self._workspaces[agent_id]

    def _invoke_codex_cli(self, prompt: str) -> Optional[str]:
        if not self._cli_path:
            return None

        agent_id = os.getenv("DOUGLAS_AGENT_ID", uuid.uuid4().hex)
        workspace = self._workspace_for(agent_id)
        last_message_path = (
            workspace.artifacts_dir / f"last_message_{uuid.uuid4().hex}.txt"
        )

        auth_file = None
        for candidate in self._candidate_auth_paths():
            if candidate.exists():
                auth_file = candidate
                break

        env = os.environ.copy()
        if auth_file is not None:
            env.setdefault("CODEX_AUTH_FILE", str(auth_file))
        if "CODEX_HOME" not in env:
            env["CODEX_HOME"] = str(workspace.root)

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
        logger.info(
            "Launching Codex CLI: %s (CODEX_HOME=%s, timeout=%ss)",
            shlex.join(command),
            env.get("CODEX_HOME", os.getenv("CODEX_HOME", "<unset>")),
            self._cli_timeout_seconds,
        )

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except FileNotFoundError:
            last_message_path.unlink(missing_ok=True)
            return None

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        communicated = False
        start_time = time.monotonic()

        try:
            while True:
                try:
                    stdout, stderr = process.communicate(
                        prompt if not communicated else None,
                        timeout=self._cli_heartbeat_seconds,
                    )
                    communicated = True
                    stdout_chunks.append(stdout or "")
                    stderr_chunks.append(stderr or "")
                    break
                except subprocess.TimeoutExpired:
                    communicated = True
                    elapsed = time.monotonic() - start_time
                    logger.info(
                        "Codex CLI still running (elapsed %.1fs)",
                        elapsed,
                    )
                    if elapsed >= self._cli_timeout_seconds:
                        process.kill()
                        try:
                            process.wait(timeout=1)
                        except Exception:
                            pass
                        logger.warning(
                            "Codex CLI timed out after %.1fs; falling back to OpenAI provider.",
                            elapsed,
                        )
                        last_message_path.unlink(missing_ok=True)
                        return None
                    continue
        except Exception as exc:  # pragma: no cover - defensive
            process.kill()
            try:
                process.wait(timeout=1)
            except Exception:
                pass
            logger.warning(
                "Codex CLI invocation failed (%s); falling back to OpenAI provider.",
                exc,
            )
            last_message_path.unlink(missing_ok=True)
            return None

        elapsed = time.monotonic() - start_time
        stdout_text = "".join(stdout_chunks).strip()
        stderr_text = "".join(stderr_chunks).strip()

        logger.info(
            "Codex CLI finished in %.2fs with return code %s",
            elapsed,
            process.returncode,
        )
        if stdout_text:
            logger.debug("Codex CLI stdout:\n%s", stdout_text)
        if stderr_text:
            logger.debug("Codex CLI stderr:\n%s", stderr_text)

        if process.returncode != 0:
            logger.warning(
                "Codex CLI exited with non-zero status (%s); falling back to OpenAI provider.",
                process.returncode,
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
            workspace.record_change("last_message.txt", message)
            return message

        if stdout_text:
            workspace.record_change("stdout.txt", stdout_text)
            return stdout_text

        if stderr_text:
            workspace.record_change("stderr.txt", stderr_text)
            return stderr_text

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
            logger.warning(
                "Failed to retrieve Codex token via CLI; falling back to environment variables. Last failure: %s",
                last_failure,
            )
        else:
            logger.warning(
                "Codex CLI returned no token; falling back to environment variables."
            )

        return None

    @staticmethod
    def _summarise_cli_error(exc: subprocess.CalledProcessError) -> str:
        output = (exc.stderr or exc.stdout or "").strip()
        if not output:
            return f"exit code {exc.returncode}"
        lines = output.splitlines()
        if lines:
            first_line = lines[0].strip()
            return first_line or f"exit code {exc.returncode}"
        return f"exit code {exc.returncode}"

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
        logger.warning(
            "Codex provider fallback triggered; returning placeholder output."
        )
        logger.debug("Codex fallback prompt preview", extra={"prompt": prompt[:200]})
        return "# Codex API unavailable; no code generated."
