"""Implementation of the automated delivery step."""

from __future__ import annotations

import hashlib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import yaml

from douglas.providers.llm_provider import LLMProvider
from douglas.vcs.commits import format_conventional_commit

__all__ = ["DeliveryContext", "DeliveryStory", "StepResult", "run_delivery"]

_PLANNED_STATUSES = {
    "",
    "todo",
    "planned",
    "ready",
    "backlog",
    "not_started",
    "up_next",
    "pending",
}
_CODE_BLOCK_PATTERN = re.compile(r"```(?P<path>[^\n`]+)\n(?P<body>.*?)```", re.DOTALL)


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if normalized:
        return normalized
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:10]


@dataclass(frozen=True)
class DeliveryStory:
    """Structured representation of a backlog story."""

    story_id: str
    title: str
    description: str = ""
    status: str = ""
    feature: Optional[str] = None

    @property
    def slug(self) -> str:
        base = self.story_id or self.title or "story"
        return _slugify(base)

    @property
    def display_id(self) -> str:
        return self.story_id or self.slug

    @property
    def reference(self) -> str:
        return self.story_id or self.slug

    @property
    def helper_function(self) -> str:
        return f"deliver_{self.slug}"

    @property
    def marker(self) -> str:
        return f"{self.display_id}:{self.slug}"

    @property
    def normalized_status(self) -> str:
        normalized = (self.status or "").strip().lower().replace("-", "_")
        return normalized.replace(" ", "_")

    def is_planned(self) -> bool:
        return self.normalized_status in _PLANNED_STATUSES

    def sort_key(self) -> tuple[str, str]:
        return (self.reference.lower(), self.title.lower())


@dataclass
class DeliveryContext:
    """Context required to execute the delivery step."""

    project_root: Path
    backlog_path: Path
    llm: Optional[LLMProvider] = None
    readme_path: Optional[Path] = None
    tests_dir: Optional[Path] = None
    stories: Optional[Sequence[DeliveryStory | Mapping[str, object]]] = None


@dataclass
class StepResult:
    """Summary of delivery outputs for orchestrators."""

    artifacts: list[str]
    commits: list[str]


def run_delivery(context: DeliveryContext) -> StepResult:
    """Generate lightweight helpers, smoke tests, and delivery notes."""

    project_root = Path(context.project_root)
    readme_path = Path(context.readme_path or (project_root / "README.md"))
    tests_root = Path(context.tests_dir or (project_root / "tests"))
    stories = list(_resolve_stories(context))
    planned_stories = sorted((story for story in stories if story.is_planned()), key=DeliveryStory.sort_key)

    if not planned_stories:
        return StepResult([], [])

    helper_package_init = _ensure_helper_package(project_root)

    touched_paths: set[Path] = set()
    if helper_package_init is not None:
        touched_paths.add(helper_package_init)
    commit_subjects: list[str] = []

    for story in planned_stories:
        story_changed, story_paths = _deliver_story(context, story, project_root, tests_root)
        touched_paths.update(story_paths)
        if story_changed:
            story_ref = (story.reference or story.slug).lower()
            subject = format_conventional_commit(
                "feat",
                f"scaffold delivery helper for {story_ref}",
                f"delivery-{story_ref}",
            )
            commit_subjects.append(subject)

    readme_changed = _ensure_readme_notes(readme_path, planned_stories)
    if readme_changed:
        touched_paths.add(readme_path)

    artifacts = sorted(_relative_path(project_root, path) for path in touched_paths)
    return StepResult(artifacts=artifacts, commits=commit_subjects)


def _resolve_stories(context: DeliveryContext) -> Iterable[DeliveryStory]:
    if context.stories:
        for entry in context.stories:
            story = _coerce_story(entry)
            if story:
                yield story
        return

    backlog_path = Path(context.backlog_path)
    try:
        text = backlog_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return

    stories_section = data.get("stories")
    if not isinstance(stories_section, Sequence):
        return

    for entry in stories_section:
        story = _coerce_story(entry)
        if story:
            yield story


def _coerce_story(entry: DeliveryStory | Mapping[str, object]) -> Optional[DeliveryStory]:
    if isinstance(entry, DeliveryStory):
        return entry
    if not isinstance(entry, Mapping):
        return None

    story_id = str(entry.get("id") or entry.get("story_id") or "").strip()
    raw_title = entry.get("name") or entry.get("title") or ""
    title = str(raw_title).strip()
    if not title:
        fallback_slug = _slugify(story_id or "story")
        title = f"Story {fallback_slug.replace('_', ' ').title()}"

    description = entry.get("description") or ""
    if isinstance(description, Sequence) and not isinstance(description, (str, bytes)):
        description = "\n".join(str(item) for item in description)
    description_text = str(description).strip()

    status = str(entry.get("status") or entry.get("state") or "").strip()
    feature = entry.get("feature") or entry.get("feature_id") or entry.get("feature_name")
    feature_text = str(feature).strip() if isinstance(feature, str) else None

    return DeliveryStory(
        story_id=story_id,
        title=title,
        description=description_text,
        status=status,
        feature=feature_text,
    )


def _ensure_helper_package(project_root: Path) -> Optional[Path]:
    package_dir = project_root / "delivery_helpers"
    init_path = package_dir / "__init__.py"
    try:
        package_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    if init_path.exists():
        return None

    init_content = "\"\"\"Helper package for delivery scaffolds.\"\"\"\n"
    init_path.write_text(init_content, encoding="utf-8")
    return init_path


def _deliver_story(
    context: DeliveryContext,
    story: DeliveryStory,
    project_root: Path,
    tests_root: Path,
) -> tuple[bool, set[Path]]:
    helper_path = project_root / "delivery_helpers" / f"{story.slug}.py"
    test_path = tests_root / "delivery" / f"test_{story.slug}.py"

    helper_rel = _relative_path(project_root, helper_path)
    test_rel = _relative_path(project_root, test_path)
    expected_paths = {helper_rel, test_rel}

    provider_changes: dict[str, bool] = {}
    if context.llm is not None:
        prompt = _build_delivery_prompt(story, helper_rel, test_rel)
        provider_changes = _apply_provider_output(
            context.llm, project_root, prompt, expected_paths
        )

    missing_paths = expected_paths - provider_changes.keys()
    fallback_changes: dict[str, bool] = {}
    if not provider_changes or missing_paths:
        fallback_changes = _ensure_story_files(
            story,
            helper_path,
            helper_rel,
            test_path,
            test_rel,
            missing_paths if provider_changes else None,
        )

    combined: dict[str, bool] = {path: False for path in expected_paths}
    combined.update(provider_changes)
    for rel_path, changed in fallback_changes.items():
        combined[rel_path] = combined.get(rel_path, False) or changed

    changed_paths: set[Path] = set()
    for rel_path, changed in combined.items():
        if changed:
            changed_paths.add(project_root / rel_path)

    story_changed = any(combined.values())
    return story_changed, changed_paths


def _build_delivery_prompt(story: DeliveryStory, helper_rel: str, test_rel: str) -> str:
    lines = [
        "DELIVERY STORY BRIEF",
        f"story_id: {story.story_id}",
        f"story_slug: {story.slug}",
        f"story_title: {story.title}",
        f"helper_path: {helper_rel}",
        f"test_path: {test_rel}",
        f"story_marker: {story.marker}",
        "Respond with code fences of the form ```path\\ncontent``` for each file.",
        "Each helper should expose a deterministic function usable in smoke tests.",
    ]
    if story.description:
        lines.append("story_description: " + " ".join(story.description.split()))
    if story.feature:
        lines.append(f"story_feature: {story.feature}")
    return "\n".join(lines)


def _apply_provider_output(
    provider: LLMProvider,
    project_root: Path,
    prompt: str,
    expected_paths: set[str],
) -> dict[str, bool]:
    try:
        response = provider.generate_code(prompt)
    except Exception:
        return {}

    changes: dict[str, bool] = {}
    for rel_path, body in _parse_code_blocks(response):
        relative = rel_path.strip()
        if relative not in expected_paths:
            continue
        destination = project_root / relative
        changed = _write_file(destination, body)
        changes[relative] = changed
    return changes


def _ensure_story_files(
    story: DeliveryStory,
    helper_path: Path,
    helper_rel: str,
    test_path: Path,
    test_rel: str,
    limit: Optional[set[str]] = None,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    module_name = _module_name_from_rel(helper_rel)

    if limit is None or helper_rel in limit:
        helper_content = _render_helper_content(story)
        results[helper_rel] = _write_file(helper_path, helper_content)
    if limit is None or test_rel in limit:
        test_content = _render_test_content(story, module_name)
        results[test_rel] = _write_file(test_path, test_content)
    return results


def _render_helper_content(story: DeliveryStory) -> str:
    title = story.title or story.display_id
    display_id = story.display_id
    marker = story.marker
    return textwrap.dedent(
        f"""\
        \"\"\"Delivery helper for {title} ({display_id}).\"\"\"

        from __future__ import annotations


        def {story.helper_function}(note: str = \"ready\") -> str:
            \"\"\"Return a deterministic marker for smoke tests.\"\"\"
            normalized = note or \"ready\"
            return \"{marker}:\" + normalized
        """
    ).strip() + "\n"


def _render_test_content(story: DeliveryStory, module_name: str) -> str:
    marker = story.marker
    return textwrap.dedent(
        f"""\
        from {module_name} import {story.helper_function}


        def test_{story.helper_function}_uses_story_marker() -> None:
            assert {story.helper_function}(\"ok\") == \"{marker}:ok\"
        """
    ).strip() + "\n"


def _module_name_from_rel(rel_path: str) -> str:
    path = Path(rel_path)
    return path.with_suffix("").as_posix().replace("/", ".")


def _ensure_readme_notes(readme_path: Path, stories: Sequence[DeliveryStory]) -> bool:
    entries = [
        f"- [ ] {story.display_id}: {story.title}"
        for story in sorted(stories, key=DeliveryStory.sort_key)
    ]
    if not entries:
        return False

    start_marker = "<!-- delivery-notes:start -->"
    end_marker = "<!-- delivery-notes:end -->"

    existing = ""
    if readme_path.exists():
        try:
            existing = readme_path.read_text(encoding="utf-8")
        except OSError:
            existing = ""

    section = "\n".join(entries)

    if start_marker in existing and end_marker in existing:
        start_index = existing.index(start_marker) + len(start_marker)
        end_index = existing.index(end_marker, start_index)
        updated = existing[:start_index] + "\n" + section + "\n" + existing[end_index:]
    else:
        prefix = existing.rstrip()
        if prefix:
            prefix += "\n\n"
        updated = (
            f"{prefix}## Delivery notes\n{start_marker}\n{section}\n{end_marker}\n"
        )

    if updated.endswith("\n\n"):
        updated = updated[:-1]

    if updated == existing:
        return False

    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(updated, encoding="utf-8")
    return True


def _parse_code_blocks(output: str) -> Iterable[tuple[str, str]]:
    for match in _CODE_BLOCK_PATTERN.finditer(output or ""):
        path = match.group("path").strip()
        body = match.group("body").rstrip() + "\n"
        yield path, body


def _write_file(path: Path, content: str) -> bool:
    normalized = content if content.endswith("\n") else content + "\n"
    try:
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing == normalized:
                return False
    except OSError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")
    return True


def _relative_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()
