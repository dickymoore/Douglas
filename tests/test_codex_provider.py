import json
import stat
import textwrap
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.providers import codex_provider
from douglas.providers.codex_provider import CodexProvider
from douglas.providers.openai_provider import OpenAIProvider


class _DummyResult:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _clear_openai_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)


def test_codex_provider_defaults_to_codex_model(monkeypatch):
    monkeypatch.delenv("OPENAI_CODEX_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = CodexProvider()

    assert provider.model == CodexProvider.DEFAULT_MODEL


def test_codex_provider_prefers_codex_specific_env(monkeypatch):
    monkeypatch.setenv("OPENAI_CODEX_MODEL", "code-cushman-001")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    provider = CodexProvider()

    assert provider.model == "code-cushman-001"


def test_codex_provider_fallback_includes_placeholder(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_CODEX_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(codex_provider.shutil, "which", lambda executable: None)

    provider = CodexProvider()

    result = provider.generate_code("print('hello world')")
    captured = capsys.readouterr()

    assert result == "# Codex API unavailable; no code generated."
    assert "[CodexProvider] Falling back to placeholder output." in captured.out


def test_codex_provider_prefers_cli_token(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(
        codex_provider.shutil, "which", lambda executable: "/usr/bin/codex"
    )

    fake_subprocess = types.SimpleNamespace(
        run=lambda *args, **kwargs: _DummyResult(stdout="token-from-cli\n"),
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    monkeypatch.setattr(codex_provider, "subprocess", fake_subprocess)

    provider = CodexProvider()

    assert provider._cli_token == "token-from-cli"
    assert provider._api_key == "token-from-cli"


def test_codex_provider_reads_cli_auth_file(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Force detection of the CLI but make command execution fail
    monkeypatch.setattr(
        codex_provider.shutil, "which", lambda executable: "/usr/bin/codex"
    )

    def _failing_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=args[0],
            stderr="error: unexpected argument 'token' found\nUsage: codex ...",
        )

    monkeypatch.setattr(
        codex_provider,
        "subprocess",
        types.SimpleNamespace(
            run=_failing_run,
            CalledProcessError=subprocess.CalledProcessError,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )

    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    auth_file = codex_home / "auth.json"
    auth_file.write_text(
        json.dumps({"tokens": {"access_token": "cli-file-token"}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(tmp_path))

    provider = CodexProvider()

    captured = capsys.readouterr()

    assert provider._cli_token == "cli-file-token"
    assert "unexpected argument" not in captured.out
    assert "Usage" not in captured.out


@pytest.mark.parametrize(
    "stdout, stderr, expected",
    [
        ("token-value\n", "", "token-value"),
        ("Access token: another-token\n", "", "another-token"),
        ("{\"token\": \"json-token\"}\n", "", "json-token"),
        ("", "{\"access_token\": \"stderr-token\"}", "stderr-token"),
    ],
)
def test_parse_cli_token(stdout: str, stderr: str, expected: str) -> None:
    assert CodexProvider._parse_cli_token(stdout, stderr) == expected


def test_parse_cli_token_rejects_noise() -> None:
    assert CodexProvider._parse_cli_token("Please login", "") is None


def _create_stub_codex_cli(tmp_path: Path, *, exit_code: int = 0, body: str = "Generated code") -> Path:
    script = tmp_path / "codex"
    script.write_text(
        textwrap.dedent(
            f"""#!/usr/bin/env python3
import sys
from pathlib import Path

args = sys.argv[1:]
output_file = None
for index, value in enumerate(args):
    if value == "--output-last-message" and index + 1 < len(args):
        output_file = Path(args[index + 1])

prompt = sys.stdin.read().strip()
message = "{body}: " + prompt

if output_file is not None:
    output_file.write_text(message, encoding="utf-8")
else:
    print(message)

sys.exit({exit_code})
"""
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_codex_provider_uses_cli_output(monkeypatch, tmp_path):
    cli_stub = _create_stub_codex_cli(tmp_path)
    monkeypatch.setattr(codex_provider.shutil, "which", lambda executable: str(cli_stub))
    monkeypatch.setenv("CODEX_CLI_TIMEOUT", "5")

    provider = CodexProvider()

    result = provider.generate_code("print('hello world')")

    assert "Generated code" in result
    assert "print('hello world')" in result


def test_codex_provider_falls_back_when_cli_fails(monkeypatch, tmp_path):
    cli_stub = _create_stub_codex_cli(tmp_path, exit_code=1)
    monkeypatch.setattr(codex_provider.shutil, "which", lambda executable: str(cli_stub))

    fallback_guard = types.SimpleNamespace(called=False)

    def _fallback(self, prompt: str) -> str:
        fallback_guard.called = True
        return "fallback-result"

    monkeypatch.setattr(OpenAIProvider, "generate_code", _fallback)

    provider = CodexProvider()

    result = provider.generate_code("print('fallback')")

    assert result == "fallback-result"
    assert fallback_guard.called
