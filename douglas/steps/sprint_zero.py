"""Sprint Zero backlog scaffolding step."""

from __future__ import annotations

import hashlib
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

from douglas.domain.backlog import (
    Epic,
    Feature,
    Story,
    render_backlog_markdown,
    serialize_backlog,
)
from douglas.providers.llm_provider import LLMProvider

__all__ = ["SprintZeroContext", "StepResult", "run_sprint_zero"]


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(_normalize_prompt(prompt).encode("utf-8")).hexdigest()


def _ci_stub(project_name: str) -> str:
    name = project_name.strip() or "Sprint Zero"
    return textwrap.dedent(
        f"""name: {name} CI
on:
  push:
    branches: [ main ]
  pull_request:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Sprint Zero placeholder
        run: echo \"Sprint Zero generated initial workflow\"
"""
    )


def _coerce_artifacts(value: Mapping[str, object]) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    for key, raw_content in value.items():
        path = str(key)
        if not path:
            continue
        if isinstance(raw_content, str):
            content = raw_content
        else:
            content = json.dumps(raw_content, indent=2, sort_keys=True)
        if not content.endswith("\n"):
            content += "\n"
        artifacts[path] = content
    return artifacts


@dataclass
class SprintZeroContext:
    project_root: Path
    project_name: str
    project_description: str
    agent_label: str
    seed: int
    llm: Optional[LLMProvider] = None
    step_name: str = "sprint_zero"
    prompt: Optional[str] = None
    backlog_state_path: Path = field(
        default_factory=lambda: Path(".douglas/state/backlog.json")
    )
    backlog_markdown_path: Path = field(
        default_factory=lambda: Path("ai-inbox/backlog.md")
    )
    ci_config_path: Path = field(
        default_factory=lambda: Path(".github/workflows/app.yml")
    )


@dataclass(eq=True)
class StepResult:
    epics: list[Epic]
    features: list[Feature]
    stories: list[Story]
    artifacts: Dict[str, str]
    prompt: str
    prompt_hash: str
    seed: int
    raw_response: str | None = None

    def persist(self, project_root: Path) -> None:
        root = project_root.resolve()
        for relative_path, content in self.artifacts.items():
            candidate = Path(relative_path)
            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                resolved = (root / candidate).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    "Sprint Zero artifact path escapes project root"
                ) from exc
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")


def _build_prompt(context: SprintZeroContext) -> str:
    description = context.project_description.strip()
    if not description:
        description = "No project description provided."
    return textwrap.dedent(
        f"""You are {context.agent_label} planning Sprint Zero backlog for {context.project_name}.

Project description:
{description}

Return JSON with keys: epics, features, stories, artifacts. Each epic should reference features and each feature should reference stories via identifiers.
Artifacts must include '.douglas/state/backlog.json' and 'ai-inbox/backlog.md'.
"""
    ).strip()


def _ensure_backlog_entries(payload: Mapping[str, object]) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]], list[Mapping[str, object]]]:
    epics_raw = []
    features_raw = []
    stories_raw = []
    raw_epics = payload.get("epics")
    if isinstance(raw_epics, list):
        epics_raw = [item for item in raw_epics if isinstance(item, Mapping)]
    raw_features = payload.get("features")
    if isinstance(raw_features, list):
        features_raw = [item for item in raw_features if isinstance(item, Mapping)]
    raw_stories = payload.get("stories")
    if isinstance(raw_stories, list):
        stories_raw = [item for item in raw_stories if isinstance(item, Mapping)]
    return epics_raw, features_raw, stories_raw


def _resolve_provider(context: SprintZeroContext) -> Optional[LLMProvider]:
    provider = context.llm
    if provider is None:
        return None
    if hasattr(provider, "with_context"):
        contextual = provider.with_context(context.agent_label, context.step_name)
        if isinstance(contextual, LLMProvider):
            return contextual
    return provider


def run_sprint_zero(context: SprintZeroContext) -> StepResult:
    prompt = context.prompt.strip() if context.prompt else _build_prompt(context)
    prompt_hash = _hash_prompt(prompt)
    provider = _resolve_provider(context)
    if provider is None:
        raise ValueError("Sprint Zero requires an LLM provider.")

    raw_response = provider.generate_code(prompt)
    if not isinstance(raw_response, str):
        raise ValueError("Sprint Zero provider must return textual JSON output.")

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("Sprint Zero provider returned invalid JSON.") from exc

    epics_raw, features_raw, stories_raw = _ensure_backlog_entries(payload)
    epics = [Epic.from_mapping(item, index=index + 1) for index, item in enumerate(epics_raw)]
    features: list[Feature] = []
    for index, item in enumerate(features_raw):
        epic_id = str(
            item.get("epic_id")
            or item.get("epic")
            or item.get("epicId")
            or ""
        )
        features.append(
            Feature.from_mapping(
                item,
                index=index + 1,
                default_epic_id=epic_id,
            )
        )

    stories: list[Story] = []
    for index, item in enumerate(stories_raw):
        feature_id = str(
            item.get("feature_id")
            or item.get("feature")
            or item.get("featureId")
            or ""
        )
        stories.append(
            Story.from_mapping(
                item,
                index=index + 1,
                default_feature_id=feature_id,
            )
        )

    artifacts_payload = payload.get("artifacts")
    artifacts: Dict[str, str] = {}
    if isinstance(artifacts_payload, Mapping):
        artifacts = _coerce_artifacts(artifacts_payload)

    backlog_dict = serialize_backlog(epics, features, stories)
    backlog_json = json.dumps(backlog_dict, indent=2, sort_keys=True) + "\n"
    artifacts.setdefault(context.backlog_state_path.as_posix(), backlog_json)

    backlog_markdown = render_backlog_markdown(
        context.project_name,
        epics,
        features,
        stories,
    )
    artifacts.setdefault(context.backlog_markdown_path.as_posix(), backlog_markdown)

    artifacts.setdefault(
        context.ci_config_path.as_posix(),
        _ci_stub(context.project_name),
    )

    metadata = payload.get("metadata")
    seed = context.seed
    recorded_hash = prompt_hash
    if isinstance(metadata, Mapping):
        meta_seed = metadata.get("seed")
        if isinstance(meta_seed, int):
            seed = meta_seed
        meta_hash = metadata.get("prompt_hash")
        if isinstance(meta_hash, str) and meta_hash:
            recorded_hash = meta_hash

    result = StepResult(
        epics=epics,
        features=features,
        stories=stories,
        artifacts=artifacts,
        prompt=prompt,
        prompt_hash=recorded_hash,
        seed=seed,
        raw_response=raw_response,
    )
    result.persist(context.project_root)
    return result
