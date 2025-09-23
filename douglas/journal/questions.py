from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

__all__ = ["Question", "raise_question", "scan_for_answers", "archive_question"]


@dataclass
class Question:
    """Represents a question awaiting a user response."""

    id: str
    sprint: int
    role: str
    topic: str
    status: str
    blocking: bool
    created_at: str
    path: Path
    archive_dir: Path
    project_root: Path
    config: Dict[str, Any]
    context: str = ""
    question: str = ""
    user_answer: str = ""
    agent_follow_up: str = ""
    front_matter: Dict[str, Any] = field(default_factory=dict)

    def normalized_role(self) -> str:
        """Return the normalized role key used for comparisons."""

        return _normalize_role_key(self.role)


def raise_question(
    role: str,
    sprint: int,
    topic: str,
    context_data: Dict[str, Any],
    blocking: bool = True,
) -> str:
    """Create a new question file and log it to the agent summary."""

    env = _prepare_environment(context_data)
    project_root = env["project_root"]
    questions_dir = env["questions_dir"]
    archive_dir = env["archive_dir"]
    filename_pattern = env["filename_pattern"]
    sprint_prefix = env["sprint_prefix"]

    question_context, question_text = _extract_question_strings(context_data)

    questions_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    question_id = f"Q-{timestamp.strftime('%Y%m%d%H%M%S%f')}"
    filename = _render_filename(
        filename_pattern,
        sprint=sprint,
        role=_role_filename_slug(role),
        id=question_id,
    )
    output_path = questions_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    front_matter = {
        "id": question_id,
        "sprint": int(sprint),
        "role": str(role),
        "status": "OPEN",
        "topic": str(topic),
        "blocking": bool(blocking),
        "created_at": timestamp.isoformat(),
    }
    fm_text = yaml.safe_dump(front_matter, sort_keys=False).strip()

    body = _render_sections(
        {
            "Context": question_context,
            "Question": question_text or str(topic),
            "User Answer": "",
            "Agent Follow-up": "",
        }
    )

    output_path.write_text(f"---\n{fm_text}\n---\n\n{body}", encoding="utf-8")

    _log_question_to_summary(
        project_root,
        sprint_prefix,
        sprint,
        role,
        question_id,
        topic,
        status_label="OPEN",
    )

    return question_id


def scan_for_answers(context_data: Dict[str, Any]) -> List[Question]:
    """Return questions that remain open, including any new answers."""

    env = _prepare_environment(context_data)
    questions_dir = env["questions_dir"]

    if not questions_dir.exists():
        return []

    questions: List[Question] = []
    for path in sorted(_iter_markdown_files(questions_dir)):
        raw_text = path.read_text(encoding="utf-8")
        front_matter, body = _split_front_matter(raw_text)
        status = str(front_matter.get("status", "OPEN"))
        if status.upper() != "OPEN":
            continue

        sections = _parse_sections(body)
        question = Question(
            id=str(front_matter.get("id") or path.stem),
            sprint=_coerce_int(
                front_matter.get("sprint"), default=env["default_sprint"]
            ),
            role=str(front_matter.get("role", "")),
            topic=str(front_matter.get("topic", "")),
            status=status,
            blocking=bool(front_matter.get("blocking", False)),
            created_at=str(front_matter.get("created_at", "")),
            path=path,
            archive_dir=env["archive_dir"],
            project_root=env["project_root"],
            config=env["config"],
            context=sections.get("Context", ""),
            question=sections.get("Question", ""),
            user_answer=sections.get("User Answer", ""),
            agent_follow_up=sections.get("Agent Follow-up", ""),
            front_matter=front_matter,
        )
        questions.append(question)

    questions.sort(key=lambda q: (q.created_at, q.id))
    return questions


def archive_question(question: Question) -> Path:
    """Archive a question after recording the agent follow-up."""

    if not question.agent_follow_up or not question.agent_follow_up.strip():
        raise ValueError(
            "Agent follow-up must be provided before archiving the question."
        )

    updated_front_matter = dict(question.front_matter)
    updated_front_matter.setdefault("id", question.id)
    updated_front_matter["status"] = "ANSWERED"
    updated_front_matter["closed_at"] = datetime.now(timezone.utc).isoformat()

    body = _render_sections(
        {
            "Context": question.context,
            "Question": question.question,
            "User Answer": question.user_answer,
            "Agent Follow-up": question.agent_follow_up,
        }
    )
    fm_text = yaml.safe_dump(updated_front_matter, sort_keys=False).strip()

    question.path.write_text(f"---\n{fm_text}\n---\n\n{body}", encoding="utf-8")

    destination = question.archive_dir / question.path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    question.path.replace(destination)

    sprint_prefix = _resolve_sprint_prefix(question.config)
    _log_question_to_summary(
        question.project_root,
        sprint_prefix,
        question.sprint,
        question.role,
        question.id,
        question.topic,
        status_label="ANSWERED",
    )

    return destination


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_environment(context_data: Dict[str, Any]) -> Dict[str, Any]:
    config = context_data.get("config") or {}
    paths = config.get("paths", {}) or {}

    project_root = Path(context_data.get("project_root", ".")).resolve()
    questions_dir = project_root / str(
        paths.get("questions_dir", "user-portal/questions")
    )
    archive_dir = project_root / str(
        paths.get("questions_archive_dir", "user-portal/questions-archive")
    )
    filename_pattern = str(
        (config.get("qna", {}) or {}).get(
            "filename_pattern", "sprint-{sprint}-{role}-{id}.md"
        )
    )
    sprint_prefix = _resolve_sprint_prefix(config)

    return {
        "project_root": project_root,
        "config": config,
        "questions_dir": questions_dir,
        "archive_dir": archive_dir,
        "filename_pattern": filename_pattern,
        "sprint_prefix": sprint_prefix,
        "default_sprint": context_data.get("sprint", 1),
    }


def _resolve_sprint_prefix(config: Dict[str, Any]) -> str:
    paths = config.get("paths", {}) or {}
    return str(paths.get("sprint_prefix", "sprint-"))


def _extract_question_strings(context_data: Dict[str, Any]) -> Tuple[str, str]:
    context_text = str(context_data.get("context", "")).strip()
    question_text = str(context_data.get("question", "")).strip()
    return context_text, question_text


def _render_filename(pattern: str, **replacements: Any) -> str:
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:  # pragma: no cover - defensive fallback
            return ""

    safe = _SafeDict({k: str(v) for k, v in replacements.items()})
    try:
        filename = pattern.format_map(safe)
    except KeyError:  # pragma: no cover - defensive
        filename = pattern
    filename = filename.strip()
    if not filename:
        filename = f"question-{safe.get('id') or datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    if not filename.lower().endswith(".md"):
        filename = f"{filename}.md"
    return filename


def _render_sections(sections: Dict[str, str]) -> str:
    parts: List[str] = []
    for title, content in sections.items():
        parts.append(f"## {title}")
        text = content.rstrip()
        if text:
            parts.append(text)
        else:
            parts.append("")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _split_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text

    parts = text.split("\n---", 1)
    if len(parts) < 2:
        return {}, text

    header = parts[0].lstrip("-\n")
    remainder = parts[1]
    if remainder.startswith("\n"):
        remainder = remainder[1:]

    try:
        data = yaml.safe_load(header) or {}
        if not isinstance(data, dict):
            data = {}
    except yaml.YAMLError:
        data = {}

    body = remainder
    if body.startswith("\n"):
        body = body[1:]
    return data, body


def _parse_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current_title: Optional[str] = None

    for line in text.splitlines():
        heading = _match_heading(line)
        if heading is not None:
            current_title = heading
            sections.setdefault(current_title, [])
            continue

        if current_title is None:
            continue
        sections[current_title].append(line)

    return {title: "\n".join(lines).strip() for title, lines in sections.items()}


def _match_heading(line: str) -> Optional[str]:
    match = re.match(r"^##\s+(.*)$", line.strip())
    if match:
        return match.group(1).strip()
    return None


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*.md") if path.is_file())


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_role_key(role: Optional[str]) -> str:
    if role is None:
        return ""
    return re.sub(r"\s+", "_", str(role).strip().lower())


def _role_filename_slug(role: str) -> str:
    value = str(role).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "role"


def _log_question_to_summary(
    project_root: Path,
    sprint_prefix: str,
    sprint: int,
    role: str,
    question_id: str,
    topic: str,
    *,
    status_label: str,
) -> None:
    role_dir = _normalize_role_key(role)
    sprint_folder = f"{sprint_prefix}{sprint}"
    summary_dir = (
        project_root / "ai-inbox" / "sprints" / sprint_folder / "roles" / role_dir
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.md"

    entry = f"- [{status_label}] {question_id}: {topic}\n"

    if not summary_path.exists():
        summary_path.write_text("# Daily Summary\n\n", encoding="utf-8")

    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
