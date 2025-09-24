import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.providers.codex_provider import CodexProvider


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
