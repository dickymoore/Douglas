"""Utilities for recording per-role summaries and handoffs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = ["append_summary", "append_handoff"]

_DEFAULT_INBOX_DIR = "ai-inbox"
_DEFAULT_SPRINT_PREFIX = "sprint-"


def append_summary(
    role: str,
    sprint_n: int,
    text: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append a summary entry for the provided role within the sprint inbox."""

    env = _prepare_environment(meta)
    role_dir = _resolve_role_directory(
        env["project_root"], env["inbox_dir"], env["sprint_prefix"], sprint_n, role
    )
    role_dir.mkdir(parents=True, exist_ok=True)

    summary_path = role_dir / "summary.md"
    entry = _build_summary_entry(
        timestamp=env["timestamp"],
        step=env.get("step"),
        text=text,
        details=env.get("details"),
        handoff_ids=env.get("handoff_ids"),
        title=env.get("title"),
    )

    _append_text(summary_path, entry)
    return summary_path


def append_handoff(
    from_role: str,
    to_role: str,
    sprint_n: int,
    topic: str,
    context: Any,
    blocking: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Append a handoff entry describing a request to another role."""

    env = _prepare_environment(meta)
    role_dir = _resolve_role_directory(
        env["project_root"], env["inbox_dir"], env["sprint_prefix"], sprint_n, from_role
    )
    role_dir.mkdir(parents=True, exist_ok=True)

    handoff_path = role_dir / "handoffs.md"
    timestamp = datetime.now(timezone.utc)
    handoff_id = f"HANDOFF-{timestamp.strftime('%Y%m%d%H%M%S%f')}"

    entry_lines = [
        f"## {handoff_id}",
        f"- timestamp: {timestamp.isoformat()}",
        f"- from_role: {_display_role(from_role)}",
        f"- to_role: {_display_role(to_role)}",
        f"- topic: {topic.strip() if topic else '(no topic)'}",
        f"- blocking: {'true' if blocking else 'false'}",
        "",
        "### Context",
        _stringify_context(context),
        "",
    ]

    _append_text(handoff_path, "\n".join(entry_lines))
    return handoff_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_environment(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data: Dict[str, Any] = dict(meta or {})

    project_root = Path(data.pop("project_root", Path.cwd())).resolve()
    config: Mapping[str, Any] = data.pop("config", {}) or {}
    paths_cfg: Mapping[str, Any] = config.get("paths", {}) or {}

    inbox_dir = data.pop("inbox_dir", None) or paths_cfg.get(
        "inbox_dir", _DEFAULT_INBOX_DIR
    )
    sprint_prefix = data.pop("sprint_prefix", None) or paths_cfg.get(
        "sprint_prefix", _DEFAULT_SPRINT_PREFIX
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    details = data.pop("details", None)
    if details is None:
        details = data.pop("summary_details", None)
    if isinstance(details, Mapping):
        details = dict(details)
    elif isinstance(details, Sequence) and not isinstance(
        details, (str, bytes, bytearray)
    ):
        details = list(details)
    elif details is None:
        details = {}
    else:
        details = {"note": details}

    handoff_ids = data.pop("handoff_ids", None) or []
    if isinstance(handoff_ids, str):
        handoff_ids = [handoff_ids]
    elif isinstance(handoff_ids, Iterable):
        handoff_ids = [str(item) for item in handoff_ids]
    else:
        handoff_ids = []

    prepared = {
        "project_root": project_root,
        "config": config,
        "inbox_dir": str(inbox_dir),
        "sprint_prefix": str(sprint_prefix),
        "timestamp": timestamp,
        "details": details,
        "handoff_ids": handoff_ids,
    }

    for key in ("step", "title"):
        if key in data:
            prepared[key] = data.pop(key)

    if data:
        # Preserve any other metadata for downstream formatting.
        prepared.setdefault("extra", {}).update(data)

    return prepared


def _resolve_role_directory(
    project_root: Path, inbox_dir: str, sprint_prefix: str, sprint_n: int, role: str
) -> Path:
    sprint_folder = f"{sprint_prefix}{int(sprint_n)}"
    role_slug = _normalize_role(role) or "unknown"
    return project_root / inbox_dir / "sprints" / sprint_folder / "roles" / role_slug


def _build_summary_entry(
    *,
    timestamp: str,
    step: Optional[str],
    text: str,
    details: Any,
    handoff_ids: Optional[Iterable[str]],
    title: Optional[str],
) -> str:
    step_label = title or (step.strip() if isinstance(step, str) else "update")
    if not step_label:
        step_label = "update"

    body_lines = [f"## {timestamp} - {step_label}"]

    cleaned = (text or "").strip()
    body_lines.append(cleaned if cleaned else "_No summary provided._")

    formatted_details = _format_details(details)
    if formatted_details:
        body_lines.append("")
        body_lines.append("**Details**")
        body_lines.extend(formatted_details)

    if handoff_ids:
        ids = [str(item).strip() for item in handoff_ids if str(item).strip()]
        if ids:
            body_lines.append("")
            body_lines.append("**Handoffs Raised**")
            body_lines.extend(f"- {identifier}" for identifier in ids)

    body_lines.append("")
    return "\n".join(body_lines)


def _append_text(path: Path, text: str) -> None:
    existing = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8") as handle:
        if existing and not text.startswith("\n"):
            handle.write("\n")
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _format_details(details: Any) -> List[str]:
    if not details:
        return []

    if isinstance(details, Mapping):
        lines: List[str] = []
        for key, value in details.items():
            label = _humanize_key(key)
            lines.extend(_format_detail_value(label, value))
        return lines

    if isinstance(details, Sequence) and not isinstance(
        details, (str, bytes, bytearray)
    ):
        return [f"- {str(item).strip()}" for item in details if str(item).strip()]

    return [f"- {str(details).strip()}"]


def _format_detail_value(label: str, value: Any) -> List[str]:
    if value is None:
        return [f"- **{label}**: _n/a_"]

    if isinstance(value, Mapping):
        lines = [f"- **{label}**:"]
        for key, val in value.items():
            lines.append(f"  - {key}: {val}")
        return lines

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        entries = [f"- **{label}**:"]
        for item in value:
            entries.append(f"  - {item}")
        return entries

    return [f"- **{label}**: {value}"]


def _humanize_key(key: Any) -> str:
    text = str(key or "").strip()
    if not text:
        return "value"
    return text.replace("_", " ")


def _normalize_role(role: Optional[str]) -> str:
    if role is None:
        return ""
    return str(role).strip().lower().replace(" ", "_")


def _display_role(role: Optional[str]) -> str:
    normalized = _normalize_role(role)
    return normalized or "unknown"


def _stringify_context(value: Any) -> str:
    if value is None:
        return "(no additional context provided)"
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else "(no additional context provided)"
    if isinstance(value, Mapping):
        parts = [f"{key}: {val}" for key, val in value.items()]
        return "\n".join(parts) if parts else "(no additional context provided)"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(parts) if parts else "(no additional context provided)"
    return str(value)
