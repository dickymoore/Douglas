"""Standup pipeline that snapshots daily progress and blockers."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from douglas.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class StandupContext:
    project_root: Path
    sprint_index: int
    sprint_day: int
    backlog_path: Path
    questions_dir: Path
    output_dir: Path
    planning_config: Dict[str, Any]


@dataclass
class StandupResult:
    output_path: Path
    story_count: int
    blocker_count: int
    wrote_report: bool


def run_standup(context: StandupContext) -> StandupResult:
    backlog = _load_backlog(context.backlog_path)
    stories = _extract_stories(backlog)
    blockers = _load_open_questions(context.questions_dir)

    context.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = context.output_dir / f"day-{context.sprint_day}.md"

    content = _render_standup(
        sprint_index=context.sprint_index,
        sprint_day=context.sprint_day,
        stories=stories,
        blockers=blockers,
    )

    output_path.write_text(content, encoding="utf-8")
    logger.info(
        "Standup snapshot written to %s (%d stories, %d blockers).",
        output_path,
        len(stories),
        len(blockers),
    )

    return StandupResult(
        output_path=output_path,
        story_count=len(stories),
        blocker_count=len(blockers),
        wrote_report=True,
    )


def _load_backlog(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.info("Backlog file %s not found; standup will reference empty backlog.", path)
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            return data
    except yaml.YAMLError as exc:
        logger.warning("Unable to parse backlog YAML %s: %s", path, exc)
    return {}


def _extract_stories(backlog: Dict[str, Any]) -> List[Dict[str, Any]]:
    stories = backlog.get("stories")
    if isinstance(stories, list):
        normalized: List[Dict[str, Any]] = []
        for entry in stories:
            if isinstance(entry, dict):
                normalized.append(entry)
        return normalized[:10]

    # Derive stories from features/tasks if explicit list missing.
    features = backlog.get("features")
    if isinstance(features, list):
        derived: List[Dict[str, Any]] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            feature_name = feature.get("name") or feature.get("id")
            tasks = feature.get("tasks") if isinstance(feature.get("tasks"), list) else []
            derived.append(
                {
                    "id": feature.get("id", feature_name),
                    "name": feature_name,
                    "feature": feature.get("id"),
                    "acceptance_criteria": feature.get("acceptance_criteria", []),
                    "tasks": tasks,
                }
            )
        return derived[:10]

    return []


def _load_open_questions(questions_dir: Path) -> List[Dict[str, Any]]:
    if not questions_dir.exists():
        return []

    blockers: List[Dict[str, Any]] = []
    for path in sorted(questions_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        question_data: Dict[str, Any] = {}
        if text.lstrip().startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 2:
                fm_text = parts[1]
                try:
                    question_data = yaml.safe_load(fm_text) or {}
                except yaml.YAMLError:
                    question_data = {}
        if not question_data:
            try:
                question_data = yaml.safe_load(text) or {}
            except yaml.YAMLError:
                question_data = {}
        status = str(question_data.get("status", "")).upper()
        if status == "OPEN":
            blockers.append(
                {
                    "id": question_data.get("id") or path.stem,
                    "topic": question_data.get("topic", ""),
                    "path": str(path),
                    "context": question_data.get("context", ""),
                }
            )
    return blockers[:10]


def _render_standup(*, sprint_index: int, sprint_day: int, stories: List[Dict[str, Any]], blockers: List[Dict[str, Any]]) -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story_lines = []
    for story in stories:
        name = story.get("name") or story.get("id") or "Unnamed"
        story_lines.append(f"- {name}")
        tasks = story.get("tasks")
        if isinstance(tasks, list) and tasks:
            for task in tasks[:3]:
                if isinstance(task, dict):
                    task_desc = task.get("description") or task.get("name")
                else:
                    task_desc = str(task)
                story_lines.append(f"  - [ ] {task_desc}")

    blocker_lines = []
    for blocker in blockers:
        topic = blocker.get("topic") or blocker.get("id")
        path = blocker.get("path")
        context = blocker.get("context") or ""
        blocker_lines.append(f"- {topic} ({path})")
        if context:
            blocker_lines.append(textwrap.indent(context.strip(), "  "))

    story_section = "\n".join(story_lines) if story_lines else "- No stories selected yet."
    blocker_section = "\n".join(blocker_lines) if blocker_lines else "- No blockers reported."

    return textwrap.dedent(
        f"""# Sprint {sprint_index} – Day {sprint_day} Standup

        Generated: {date_str}

        ## Focus Stories
        {story_section}

        ## Blockers / Questions
        {blocker_section}
        """
    ).strip() + "\n"
