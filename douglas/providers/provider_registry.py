"""Utilities for configuring and resolving language model providers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Set

from douglas.providers.codex_provider import CodexProvider
from douglas.providers.llm_provider import LLMProvider
from douglas.providers.replay_provider import (
    CassetteRecordingProvider,
    CassetteStore,
    compute_project_fingerprint,
)


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

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        project_root: Optional[Path] = None,
    ) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._aliases: Dict[str, str] = {}
        self._assignments: Dict[str, Dict[str, str]] = {}
        self._default_key: Optional[str] = None
        self._provider_labels: Dict[str, str] = {}
        self._provider_models: Dict[str, Optional[str]] = {}

        self._project_root = Path(project_root or Path.cwd())
        self._ai_config = dict(config or {})
        self._base_seed = self._resolve_seed(self._ai_config)
        self._cassette_dir = self._resolve_cassette_dir(self._ai_config)
        self._cassette_store = CassetteStore(self._cassette_dir)
        self._project_fingerprint = compute_project_fingerprint(
            self._project_root, self._ai_config
        )
        self._record_cassettes = self._resolve_bool(
            self._ai_config.get("record_cassettes")
        )
        self._mode = self._resolve_mode(self._ai_config)

        self._initialize(self._ai_config)

        if not self._providers:
            fallback = CodexProvider()
            self._register_provider("codex", fallback, aliases={"codex", "default"})
            self._default_key = "codex"

        if self._default_key is None:
            self._default_key = next(iter(self._providers), None)

        if self._default_key is not None:
            self._aliases.setdefault("default", self._default_key)

        if self._record_cassettes and self._mode == "real":
            self._wrap_providers_for_recording()

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
                    provider = self._providers[candidate_key]
                    return self._contextualize(provider, agent_label, step_name)

        provider = self.default_provider
        return self._contextualize(provider, agent_label, step_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize(self, config: Mapping[str, Any]) -> None:
        if self._mode in {"mock", "replay", "null"}:
            options = self._offline_provider_options(self._mode)
            provider = self._instantiate_provider(self._mode, options)
            self._default_key = self._register_provider(
                self._mode,
                provider,
                aliases={self._mode, "default"},
                provider_name=self._mode,
                model_name=options.get("model"),
            )
            # In offline modes (mock, replay, null), we intentionally skip the standard
            # provider configuration logic below. Only a single offline provider is needed,
            # and the rest of the configuration (named providers, assignments, etc.) is not
            # relevant in these modes. This early return ensures that only the offline
            # provider is registered and used.
            return

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
        model_name = options.get("model") or options.get("model_name")
        return self._register_provider(
            key,
            provider,
            aliases={provider_name},
            provider_name=provider_name,
            model_name=model_name,
        )

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
        aliases: Optional[Set[str]] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        canonical = self._normalize(key) or f"provider_{len(self._providers)}"
        self._providers[canonical] = provider
        existing_label = getattr(provider, "douglas_provider_id", None)
        if existing_label:
            label = str(existing_label)
        else:
            label = provider_name or canonical
        self._provider_labels[canonical] = label
        if existing_label != label:
            setattr(provider, "douglas_provider_id", label)
        if model_name is None:
            model_attr = getattr(provider, "model", None)
            if isinstance(model_attr, str):
                model_name = model_attr
        self._provider_models[canonical] = model_name
        if model_name is not None:
            setattr(provider, "douglas_model_name", model_name)

        alias_set = {canonical}
        if aliases:
            alias_set.update(self._normalize(alias) for alias in aliases if alias)

        for alias in alias_set:
            self._aliases[alias] = canonical

        return canonical

    @staticmethod
    def _normalize(value: Any) -> str:
        return str(value).strip().lower() if value is not None else ""

    def _resolve_seed(self, config: Mapping[str, Any]) -> int:
        raw_seed = config.get("seed")
        try:
            return int(raw_seed)
        except (TypeError, ValueError):
            return 0

    def _resolve_cassette_dir(self, config: Mapping[str, Any]) -> Path:
        raw_dir = config.get("replay_dir") or ".douglas/cassettes"
        candidate = Path(raw_dir)
        if not candidate.is_absolute():
            candidate = self._project_root / candidate
        return candidate

    def _resolve_mode(self, config: Mapping[str, Any]) -> str:
        raw_mode = config.get("mode", "real")
        mode = str(raw_mode).strip().lower() if raw_mode is not None else "real"
        sentinel = self._project_root / ".douglas" / "mock.on"
        if sentinel.exists():
            return "mock"
        if mode in {"mock", "replay", "null", "real"}:
            return mode
        return "real"

    def _resolve_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    def _offline_provider_options(self, mode: str) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "project_root": self._project_root,
            "seed": self._base_seed,
        }
        if mode == "replay":
            options.update(
                {
                    "cassette_store": self._cassette_store,
                    "project_fingerprint": self._project_fingerprint,
                    "model": self._ai_config.get("model")
                    or self._provider_models.get(self._default_key or "", None),
                }
            )
        return options

    def _wrap_providers_for_recording(self) -> None:
        for key, provider in list(self._providers.items()):
            if isinstance(provider, CassetteRecordingProvider):
                continue
            label = self._provider_labels.get(key, key)
            model_name = self._provider_models.get(key)
            wrapped = CassetteRecordingProvider(
                provider,
                store=self._cassette_store,
                provider_name=label,
                model_name=model_name,
                project_fingerprint=self._project_fingerprint,
                base_seed=self._base_seed,
            )
            self._providers[key] = wrapped

    def _contextualize(
        self, provider: LLMProvider, agent_label: str, step_name: str
    ) -> LLMProvider:
        if hasattr(provider, "with_context"):
            try:
                contextual = getattr(provider, "with_context")(agent_label, step_name)
            except Exception:
                return provider
            if isinstance(contextual, LLMProvider):
                return contextual
        return provider
