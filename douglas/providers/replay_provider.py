"""Replay provider and cassette recording helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from douglas import __version__
from douglas.providers.llm_provider import LLMProvider


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(_normalize_prompt(prompt).encode("utf-8")).hexdigest()


def derive_context_seed(base_seed: int, agent: str, step: str, prompt_hash: str) -> int:
    material = f"{base_seed}:{agent}:{step}:{prompt_hash}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16)


def _stable_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def compute_project_fingerprint(project_root: Path, ai_config: Mapping[str, object]) -> str:
    hasher = hashlib.sha256()
    config_path = project_root / "douglas.yaml"
    if config_path.exists():
        try:
            hasher.update(config_path.read_bytes())
        except OSError:
            hasher.update(b"<unreadable>")
    else:
        hasher.update(b"<missing-config>")

    normalized_ai = _stable_json(_serialize_mapping(ai_config))
    hasher.update(normalized_ai.encode("utf-8"))

    layout_tokens = []
    for candidate in (
        project_root / "ai-inbox",
        project_root / ".douglas",
        project_root / "src",
        project_root / "tests",
    ):
        token = f"{candidate.name}:{candidate.exists()}:{candidate.is_dir()}"
        layout_tokens.append(token)
    hasher.update("|".join(sorted(layout_tokens)).encode("utf-8"))
    return hasher.hexdigest()


def _serialize_mapping(data: Mapping[str, object]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            result[str(key)] = _serialize_mapping(value)
        elif isinstance(value, (list, tuple)):
            result[str(key)] = [
                _serialize_mapping(item)
                if isinstance(item, Mapping)
                else _serialize_value(item)
                for item in value
            ]
        else:
            result[str(key)] = _serialize_value(value)
    return result


def _serialize_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class CassetteKey:
    provider: str
    model: Optional[str]
    role: Optional[str]
    step: str
    agent_id: str
    project_fingerprint: str
    prompt_hash: str
    seed: Optional[int]

    def as_tuple(self) -> Tuple[str, str, str, str, str, str, str, str]:
        return (
            self.provider,
            self.model or "",
            self.role or "",
            self.step,
            self.agent_id,
            self.project_fingerprint,
            self.prompt_hash,
            str(self.seed) if self.seed is not None else "",
        )

    def as_dict(self) -> Dict[str, object]:
        data = {
            "provider": self.provider,
            "model": self.model,
            "role": self.role,
            "step": self.step,
            "agent_id": self.agent_id,
            "project_fingerprint": self.project_fingerprint,
            "prompt_hash": self.prompt_hash,
            "seed": self.seed,
        }
        return data


def _cassette_sort_key(path: Path) -> Tuple[str, int, str]:
    stem = path.stem
    parts = stem.split("-")
    suffix = 0
    if len(parts) > 2 and parts[-1].isdigit():
        suffix = int(parts[-1])
        prefix = "-".join(parts[:-1])
    else:
        prefix = stem
    return (prefix, suffix, path.name)


class CassetteStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._loaded = False
        self._index: Dict[Tuple[str, ...], Dict[str, object]] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._index.clear()
        if not self.directory.exists():
            self._loaded = True
            return
        for path in sorted(
            self.directory.glob("*.jsonl"), key=_cassette_sort_key
        ):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key_payload = data.get("key")
                        if not isinstance(key_payload, Mapping):
                            continue
                        key = CassetteKey(
                            provider=str(key_payload.get("provider", "")),
                            model=(
                                str(key_payload.get("model"))
                                if key_payload.get("model") is not None
                                else None
                            ),
                            role=(
                                str(key_payload.get("role"))
                                if key_payload.get("role") is not None
                                else None
                            ),
                            step=str(key_payload.get("step", "")),
                            agent_id=str(key_payload.get("agent_id", "")),
                            project_fingerprint=str(
                                key_payload.get("project_fingerprint", "")
                            ),
                            prompt_hash=str(key_payload.get("prompt_hash", "")),
                            seed=(
                                int(key_payload.get("seed"))
                                if key_payload.get("seed") not in (None, "")
                                else None
                            ),
                        )
                        self._index[key.as_tuple()] = data
            except OSError:
                continue
        self._loaded = True

    def list_keys(self) -> Iterable[Dict[str, object]]:
        self._load()
        for entry in self._index.values():
            key = entry.get("key")
            if isinstance(key, Mapping):
                yield dict(key)

    def lookup(self, key: CassetteKey) -> Optional[Dict[str, object]]:
        self._load()
        return self._index.get(key.as_tuple())

    def record(self, key: CassetteKey, entry: Dict[str, object]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(entry, sort_keys=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        base_name = f"cassette-{timestamp}.jsonl"
        target = self.directory / base_name

        existing = self.lookup(key)
        if existing:
            existing_text = (
                existing.get("output", {}).get("text")
                if isinstance(existing.get("output"), Mapping)
                else None
            )
            new_text = (
                entry.get("output", {}).get("text")
                if isinstance(entry.get("output"), Mapping)
                else None
            )
            if existing_text != new_text:
                key_json = json.dumps(key.as_dict(), sort_keys=True)
                existing_hash = (
                    hashlib.sha256(existing_text.encode("utf-8")).hexdigest()
                    if isinstance(existing_text, str) else "N/A"
                )
                new_hash = (
                    hashlib.sha256(new_text.encode("utf-8")).hexdigest()
                    if isinstance(new_text, str) else "N/A"
                )
                warnings.warn(
                    f"Cassette key collision detected; writing to new replay file.\n"
                    f"Key: {key_json}\n"
                    f"Existing output hash: {existing_hash}\n"
                    f"New output hash: {new_hash}",
                    RuntimeWarning,
                )
                counter = 1
                while True:
                    candidate = self.directory / f"cassette-{timestamp}-{counter}.jsonl"
                    if not candidate.exists():
                        target = candidate
                        break
                    counter += 1

        with target.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        self._loaded = False
        self._index[key.as_tuple()] = entry
        return target


class ReplayProvider(LLMProvider):
    def __init__(
        self,
        *,
        store: CassetteStore,
        project_root: Path,
        project_fingerprint: str,
        base_seed: int = 0,
        model_name: Optional[str] = None,
        provider_name: str = "replay",
    ) -> None:
        self._store = store
        self._project_root = Path(project_root)
        self._fingerprint = project_fingerprint
        self._base_seed = base_seed
        self._model_name = model_name
        self._provider_name = provider_name
        self.provider_id = provider_name

    def with_context(self, agent_label: str, step_name: str) -> LLMProvider:
        return _ReplayContextProvider(
            store=self._store,
            provider_name=self._provider_name,
            model_name=self._model_name,
            agent_label=agent_label,
            step_name=step_name,
            fingerprint=self._fingerprint,
            base_seed=self._base_seed,
        )

    def generate_code(self, prompt: str) -> str:
        raise RuntimeError("ReplayProvider requires contextualisation via with_context().")


class _ReplayContextProvider(LLMProvider):
    def __init__(
        self,
        *,
        store: CassetteStore,
        provider_name: str,
        model_name: Optional[str],
        agent_label: str,
        step_name: str,
        fingerprint: str,
        base_seed: int,
    ) -> None:
        self._store = store
        self._provider_name = provider_name
        self._model_name = model_name
        self._agent_label = agent_label
        self._step_name = step_name
        self._fingerprint = fingerprint
        self._base_seed = base_seed

    def generate_code(self, prompt: str) -> str:
        prompt_hash = _hash_prompt(prompt)
        seed = derive_context_seed(
            self._base_seed, self._agent_label, self._step_name, prompt_hash
        )
        key = CassetteKey(
            provider=self._provider_name,
            model=self._model_name,
            role=self._agent_label,
            step=self._step_name,
            agent_id=self._agent_label,
            project_fingerprint=self._fingerprint,
            prompt_hash=prompt_hash,
            seed=seed,
        )
        entry = self._store.lookup(key)
        if not entry:
            available = list(self._store.list_keys())
            sample = json.dumps(available[:5], indent=2)
            raise KeyError(
                "Replay cassette not found for key\n"
                f"{json.dumps(key.as_dict(), indent=2)}\n"
                f"Replay directory: {self._store.directory}\n"
                "Available keys: " + sample
            )
        output = entry.get("output", {})
        text = output.get("text") if isinstance(output, Mapping) else None
        if not isinstance(text, str):
            raise ValueError("Replay cassette missing textual output.")
        return text


class CassetteRecordingProvider(LLMProvider):
    def __init__(
        self,
        base: LLMProvider,
        *,
        store: CassetteStore,
        provider_name: str,
        model_name: Optional[str],
        project_fingerprint: str,
        base_seed: int,
    ) -> None:
        self._base = base
        self._store = store
        self._provider_name = provider_name
        self._model_name = model_name
        self._fingerprint = project_fingerprint
        self._base_seed = base_seed

    def with_context(self, agent_label: str, step_name: str) -> LLMProvider:
        if hasattr(self._base, "with_context"):
            contextual = getattr(self._base, "with_context")(
                agent_label, step_name
            )
        else:
            contextual = self._base
        return _RecordingContextProvider(
            contextual,
            store=self._store,
            provider_name=self._provider_name,
            model_name=self._model_name,
            fingerprint=self._fingerprint,
            base_seed=self._base_seed,
            agent_label=agent_label,
            step_name=step_name,
        )

    def generate_code(self, prompt: str) -> str:
        raise RuntimeError("CassetteRecordingProvider requires context via with_context().")


class _RecordingContextProvider(LLMProvider):
    def __init__(
        self,
        base: LLMProvider,
        *,
        store: CassetteStore,
        provider_name: str,
        model_name: Optional[str],
        fingerprint: str,
        base_seed: int,
        agent_label: str,
        step_name: str,
    ) -> None:
        self._base = base
        self._store = store
        self._provider_name = provider_name
        self._model_name = model_name
        self._fingerprint = fingerprint
        self._base_seed = base_seed
        self._agent_label = agent_label
        self._step_name = step_name

    def generate_code(self, prompt: str) -> str:
        prompt_hash = _hash_prompt(prompt)
        seed = derive_context_seed(
            self._base_seed, self._agent_label, self._step_name, prompt_hash
        )
        response = self._base.generate_code(prompt)
        if not isinstance(response, str):
            return response

        key = CassetteKey(
            provider=self._provider_name,
            model=self._model_name,
            role=self._agent_label,
            step=self._step_name,
            agent_id=self._agent_label,
            project_fingerprint=self._fingerprint,
            prompt_hash=prompt_hash,
            seed=seed,
        )
        record = {
            "version": "1",
            "key": key.as_dict(),
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "douglas_version": __version__,
                "os": platform.platform(),
                "python": sys.version,
            },
            "output": {
                "text": response,
                "notes": [
                    {
                        "level": "info",
                        "msg": "Recorded by CassetteRecordingProvider",
                    }
                ],
            },
        }
        existing = self._store.lookup(key)
        if existing:
            existing_text = (
                existing.get("output", {}).get("text")
                if isinstance(existing.get("output"), Mapping)
                else None
            )
            if existing_text == response:
                return response
        self._store.record(key, record)
        return response

