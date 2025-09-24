import types

import pytest

from douglas.providers import codex_provider
from douglas.providers.codex_provider import CodexProvider


class _DummyResult:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


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
