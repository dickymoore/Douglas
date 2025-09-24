import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.providers import codex_provider
from douglas.providers.codex_provider import CodexProvider


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
        run=lambda *args, **kwargs: _DummyResult(stdout="token-from-cli\n")
    )
    monkeypatch.setattr(codex_provider, "subprocess", fake_subprocess)

    provider = CodexProvider()

    assert provider._cli_token == "token-from-cli"
    assert provider._api_key == "token-from-cli"


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
