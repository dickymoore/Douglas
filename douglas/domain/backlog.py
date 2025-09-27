"""Backlog domain models used for Sprint Zero planning."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "Epic",
    "Feature",
    "Story",
    "render_backlog_markdown",
    "serialize_backlog",
]


def _slugify(value: str, *, prefix: str, index: int) -> str:
    cleaned = "".join(ch for ch in value.lower() if ch.isalnum())
    if not cleaned:
        cleaned = f"{prefix}{index}"
    return f"{prefix}-{cleaned[:12]}"


def _coerce_identifier(
    candidate: object, *, fallback_name: str, prefix: str, index: int
) -> str:
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    name = fallback_name.strip() or f"{prefix.title()} {index}"
    return _slugify(name, prefix=prefix, index=index)


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_criteria(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        results: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    results.append(text)
            elif item is not None:
                rendered = str(item).strip()
                if rendered:
                    results.append(rendered)
        return results
    return []


@dataclass(frozen=True)
class Epic:
    """Represents an epic generated during Sprint Zero."""

    identifier: str
    title: str
    description: str = ""

    @staticmethod
    def from_mapping(data: Mapping[str, object], *, index: int) -> "Epic":
        title = _coerce_text(data.get("title") or data.get("name") or "")
        identifier = _coerce_identifier(
            data.get("id") or data.get("identifier"),
            fallback_name=title or "Epic",
            prefix="epic",
            index=index,
        )
        description = _coerce_text(
            data.get("description") or data.get("summary") or ""
        )
        return Epic(identifier=identifier, title=title or identifier, description=description)

    def to_dict(self) -> MutableMapping[str, object]:
        return {
            "id": self.identifier,
            "title": self.title,
            "description": self.description,
        }


@dataclass(frozen=True)
class Feature:
    """Represents a feature aligned to an epic."""

    identifier: str
    title: str
    epic_id: str
    description: str = ""
    acceptance_criteria: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def from_mapping(
        data: Mapping[str, object], *, index: int, default_epic_id: str
    ) -> "Feature":
        title = _coerce_text(data.get("title") or data.get("name") or "")
        identifier = _coerce_identifier(
            data.get("id") or data.get("identifier"),
            fallback_name=title or "Feature",
            prefix="feature",
            index=index,
        )
        epic_id = _coerce_text(
            data.get("epic_id") or data.get("epic") or default_epic_id
        )
        description = _coerce_text(
            data.get("description") or data.get("summary") or ""
        )
        criteria = tuple(_coerce_criteria(data.get("acceptance_criteria")))
        return Feature(
            identifier=identifier,
            title=title or identifier,
            epic_id=epic_id,
            description=description,
            acceptance_criteria=criteria,
        )

    def to_dict(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "id": self.identifier,
            "title": self.title,
            "description": self.description,
            "epic_id": self.epic_id,
        }
        if self.acceptance_criteria:
            payload["acceptance_criteria"] = list(self.acceptance_criteria)
        return payload


@dataclass(frozen=True)
class Story:
    """Represents a user story aligned to a feature."""

    identifier: str
    title: str
    feature_id: str
    description: str = ""
    acceptance_criteria: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def from_mapping(
        data: Mapping[str, object], *, index: int, default_feature_id: str
    ) -> "Story":
        title = _coerce_text(data.get("title") or data.get("name") or "")
        identifier = _coerce_identifier(
            data.get("id") or data.get("identifier"),
            fallback_name=title or "Story",
            prefix="story",
            index=index,
        )
        feature_id = _coerce_text(
            data.get("feature_id")
            or data.get("feature")
            or data.get("featureId")
            or default_feature_id
        )
        description = _coerce_text(
            data.get("description") or data.get("summary") or ""
        )
        criteria = tuple(_coerce_criteria(data.get("acceptance_criteria")))
        return Story(
            identifier=identifier,
            title=title or identifier,
            feature_id=feature_id,
            description=description,
            acceptance_criteria=criteria,
        )

    def to_dict(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "id": self.identifier,
            "title": self.title,
            "description": self.description,
            "feature_id": self.feature_id,
        }
        if self.acceptance_criteria:
            payload["acceptance_criteria"] = list(self.acceptance_criteria)
        return payload


def serialize_backlog(
    epics: Iterable[Epic],
    features: Iterable[Feature],
    stories: Iterable[Story],
) -> MutableMapping[str, object]:
    return {
        "epics": [epic.to_dict() for epic in epics],
        "features": [feature.to_dict() for feature in features],
        "stories": [story.to_dict() for story in stories],
    }


def _group_by(items: Iterable, key_attr: str) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for item in items:
        key = getattr(item, key_attr)
        grouped.setdefault(key, []).append(item)
    return grouped


def render_backlog_markdown(
    project_name: str,
    epics: Sequence[Epic],
    features: Sequence[Feature],
    stories: Sequence[Story],
) -> str:
    if not project_name:
        project_name = "Sprint Zero"
    lines: list[str] = [f"# {project_name} – Sprint Zero Backlog", ""]
    features_by_epic = _group_by(features, "epic_id")
    stories_by_feature = _group_by(stories, "feature_id")

    for epic in epics:
        lines.append(f"## {epic.title} ({epic.identifier})")
        if epic.description:
            lines.append(epic.description)
        epic_features = features_by_epic.get(epic.identifier, [])
        if epic_features:
            for feature in epic_features:
                lines.append(f"- **{feature.title}** ({feature.identifier})")
                if feature.description:
                    lines.append(textwrap.indent(feature.description, "  "))
                story_entries = stories_by_feature.get(feature.identifier, [])
                for story in story_entries:
                    lines.append(
                        f"  - {story.title} ({story.identifier})"
                    )
                    if story.description:
                        lines.append(textwrap.indent(story.description, "    "))
        lines.append("")

    rendered = "\n".join(lines).strip()
    return rendered + "\n"
