"""Utilities for configuring and resolving language model providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

from douglas.providers.codex_provider import CodexProvider
from douglas.providers.llm_provider import LLMProvider


class StaticLLMProviderRegistry:
    """Minimal registry used when tests monkeypatch ``create_llm_provider``."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @property
    def default_provider(self) -> LLMProvider:
        return self._provider

    def resolve(self, agent_label: str, step_name: str) -> LLMProvider:
        return self._provider


class LLMProviderRegistry:
    """Registry that supports multiple providers and per-agent assignments."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._aliases: Dict[str, str] = {}
        self._assignments: Dict[str, Dict[str, str]] = {}
        self._default_key: Optional[str] = None

        self._initialize(config or {})

        if not self._providers:
            fallback = CodexProvider()
            self._register_provider("codex", fallback, aliases={"codex", "default"})
            self._default_key = "codex"

        if self._default_key is None:
            self._default_key = next(iter(self._providers), None)

        if self._default_key is not None:
            self._aliases.setdefault("default", self._default_key)

    @property
    def default_provider(self) -> LLMProvider:
        if self._default_key and self._default_key in self._providers:
            return self._providers[self._default_key]
        return next(iter(self._providers.values()))

    def resolve(self, agent_label: str, step_name: str) -> LLMProvider:
        agent_key = self._normalize(agent_label)
        step_key = self._normalize(step_name)

        if agent_key:
            agent_map = self._assignments.get(agent_key)
            if agent_map:
                candidate_key = agent_map.get(step_key) or agent_map.get("*")
                if candidate_key and candidate_key in self._providers:
                    return self._providers[candidate_key]

        return self.default_provider

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize(self, config: Mapping[str, Any]) -> None:
        providers_cfg = config.get("providers")
        if isinstance(providers_cfg, Mapping):
            for label, spec in providers_cfg.items():
                self._load_named_provider(label, spec)

        top_level_provider = config.get("provider")
        if isinstance(top_level_provider, str) and top_level_provider.strip():
            sanitized = {
                key: value
                for key, value in config.items()
                if key
                not in {
                    "provider",
                    "providers",
                    "default_provider",
                    "assignments",
                    "prompt",
                }
            }
            canonical = self._lookup_or_create(top_level_provider, sanitized)
            if canonical:
                self._aliases["default"] = canonical
                if self._default_key is None:
                    self._default_key = canonical

        default_provider_name = config.get("default_provider")
        if isinstance(default_provider_name, str) and default_provider_name.strip():
            canonical = self._lookup_or_create(default_provider_name, {})
            if canonical:
                self._default_key = canonical

        assignments_cfg = config.get("assignments")
        if isinstance(assignments_cfg, Mapping):
            for agent, mapping in assignments_cfg.items():
                self._load_assignment(agent, mapping)

    def _load_named_provider(self, label: str, spec: Any) -> None:
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"Provider entry '{label}' must be a mapping of options to values."
            )

        provider_name = spec.get("provider") or spec.get("type") or label
        options = {
            key: value
            for key, value in spec.items()
            if key not in {"provider", "type", "aliases", "default"}
        }
        aliases = set()
        raw_aliases = spec.get("aliases")
        if isinstance(raw_aliases, str):
            aliases.add(raw_aliases)
        elif isinstance(raw_aliases, Mapping):
            aliases.update(str(item) for item in raw_aliases.values())
        elif isinstance(raw_aliases, (list, set, tuple)):
            aliases.update(str(item) for item in raw_aliases)

        canonical = self._lookup_or_create(provider_name, options, preferred_key=label)
        if canonical:
            aliases.update({label, provider_name})
            if spec.get("default"):
                self._default_key = canonical
            self._aliases.setdefault("default", self._default_key or canonical)
            for alias in aliases:
                normalized = self._normalize(alias)
                if normalized:
                    self._aliases[normalized] = canonical

    def _load_assignment(self, agent: str, mapping: Any) -> None:
        normalized_agent = self._normalize(agent)
        if not normalized_agent:
            return

        agent_map: Dict[str, str] = {}
        if isinstance(mapping, Mapping):
            for step, provider_ref in mapping.items():
                normalized_step = self._normalize(step) or "*"
                canonical = self._resolve_assignment_provider(provider_ref)
                if canonical:
                    agent_map[normalized_step] = canonical
        elif isinstance(mapping, str):
            canonical = self._resolve_assignment_provider(mapping)
            if canonical:
                agent_map["*"] = canonical

        if agent_map:
            self._assignments[normalized_agent] = agent_map

    def _resolve_assignment_provider(self, reference: Any) -> Optional[str]:
        if isinstance(reference, Mapping):
            provider_name = reference.get("provider") or reference.get("type")
            if not provider_name:
                return None
            options = {
                key: value
                for key, value in reference.items()
                if key not in {"provider", "type"}
            }
            return self._lookup_or_create(provider_name, options)
        if isinstance(reference, str):
            return self._lookup_or_create(reference, {})
        return None

    def _lookup_or_create(
        self,
        provider_name: str,
        options: Mapping[str, Any],
        *,
        preferred_key: Optional[str] = None,
    ) -> Optional[str]:
        normalized = self._normalize(provider_name)
        if not normalized:
            return None

        canonical = self._aliases.get(normalized)
        if canonical:
            return canonical

        provider = self._instantiate_provider(provider_name, options)
        key = preferred_key or provider_name
        return self._register_provider(key, provider, aliases={provider_name})

    def _instantiate_provider(
        self, provider_name: str, options: Mapping[str, Any]
    ) -> LLMProvider:
        sanitized = {
            key: value
            for key, value in options.items()
            if key not in {"aliases", "default"} and value is not None
        }
        if "model" in sanitized and "model_name" not in sanitized:
            sanitized["model_name"] = sanitized["model"]
        try:
            return LLMProvider.create_provider(provider_name, **sanitized)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported LLM provider '{provider_name}': {exc}"
            ) from exc

    def _register_provider(
        self,
        key: str,
        provider: LLMProvider,
        *,
        aliases: Optional[set[str]] = None,
    ) -> str:
        canonical = self._normalize(key) or f"provider_{len(self._providers)}"
        self._providers[canonical] = provider

        alias_set = {canonical}
        if aliases:
            alias_set.update(self._normalize(alias) for alias in aliases if alias)

        for alias in alias_set:
            self._aliases[alias] = canonical

        return canonical

    @staticmethod
    def _normalize(value: Any) -> str:
        return str(value).strip().lower() if value is not None else ""
