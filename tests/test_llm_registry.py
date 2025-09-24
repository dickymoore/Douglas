from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.providers.claude_code_provider import ClaudeCodeProvider
from douglas.providers.codex_provider import CodexProvider
from douglas.providers.gemini_provider import GeminiProvider
from douglas.providers.provider_registry import LLMProviderRegistry


def test_registry_defaults_to_codex():
    registry = LLMProviderRegistry({})
    provider = registry.default_provider
    assert isinstance(provider, CodexProvider)
    assert registry.resolve("Developer", "generate") is provider


def test_registry_respects_assignments():
    config = {
        "default_provider": "codex",
        "providers": {
            "codex": {"provider": "codex"},
            "claude": {"provider": "claude_code"},
        },
        "assignments": {
            "developer": {"review": "claude"},
        },
    }
    registry = LLMProviderRegistry(config)
    assert isinstance(registry.resolve("Developer", "review"), ClaudeCodeProvider)
    assert isinstance(registry.resolve("Developer", "generate"), CodexProvider)


def test_registry_instantiates_providers_from_assignments():
    config = {
        "default_provider": "codex",
        "assignments": {"tester": {"test": "gemini"}},
    }
    registry = LLMProviderRegistry(config)
    assert isinstance(registry.resolve("Tester", "test"), GeminiProvider)
