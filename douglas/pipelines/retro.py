from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from douglas.journal import retro_collect

__all__ = ["run_retro", "RetroResult"]


@dataclass
class RetroResult:
    """Metadata describing the retro outputs written to disk."""

    sprint_folder: str
    generated_at: str
    instructions: Dict[str, Path]
    backlog_entries: List[Dict[str, Any]]
    llm_payload: Dict[str, Any]


def run_retro(context: Dict[str, Any]) -> RetroResult:
    """Execute the retro step using collected sprint artifacts and an LLM."""

    project_root = Path(context.get("project_root", ".")).resolve()
    config: Dict[str, Any] = context.get("config", {}) or {}
    sprint_manager = context.get("sprint_manager")
    llm = _resolve_llm_provider(context)

    sprint_index = _resolve_sprint_index(context, sprint_manager)
    paths_config = config.get("paths", {}) or {}
    sprint_prefix = str(paths_config.get("sprint_prefix", "sprint-"))
    sprint_folder = f"{sprint_prefix}{sprint_index}"

    role_documents = retro_collect.collect_role_documents(project_root, sprint_folder)

    prompt = _build_prompt(sprint_index, role_documents)
    llm_response = llm.generate_code(prompt)
    parsed = _parse_response(llm_response)

    wins = _coerce_string_list(parsed.get("wins"))
    pain_points = _coerce_string_list(parsed.get("pain_points"))
    risks = _coerce_string_list(parsed.get("risks"))
    role_actions = _normalize_role_instructions(parsed.get("role_instructions"), role_documents)
    backlog_items = _normalize_backlog_items(parsed.get("pre_feature_items"))

    retro_config = config.get("retro", {}) or {}
    output_tokens = {
        str(token).strip().lower()
        for token in (retro_config.get("outputs") or [])
        if str(token).strip()
    }
    if not output_tokens:
        output_tokens = {"role_instructions", "pre_feature_backlog"}

    generated_at = datetime.now(timezone.utc).isoformat()

    instruction_paths: Dict[str, Path] = {}
    if "role_instructions" in output_tokens:
        instruction_paths = _write_role_instructions(
            project_root,
            sprint_folder,
            role_documents,
            role_actions,
            wins,
            pain_points,
            risks,
            generated_at,
        )

    backlog_entries: List[Dict[str, Any]] = []
    if "pre_feature_backlog" in output_tokens and backlog_items:
        backlog_entries = _append_pre_feature_backlog(
            project_root,
            retro_config,
            backlog_items,
            sprint_index,
        )

    payload = {
        "wins": wins,
        "pain_points": pain_points,
        "risks": risks,
        "role_instructions": role_actions,
        "pre_feature_items": backlog_items,
    }

    return RetroResult(
        sprint_folder=sprint_folder,
        generated_at=generated_at,
        instructions=instruction_paths,
        backlog_entries=backlog_entries,
        llm_payload=payload,
    )


# ---------------------------------------------------------------------------
# Prompt construction & parsing
# ---------------------------------------------------------------------------


def _resolve_llm_provider(context: Dict[str, Any]) -> Any:
    for key in ("llm", "llm_provider", "lm_provider"):
        provider = context.get(key)
        if provider is not None and hasattr(provider, "generate_code"):
            return provider
    raise ValueError("Retro step requires an LLM provider with a 'generate_code' method.")


def _resolve_sprint_index(context: Dict[str, Any], sprint_manager: Any) -> int:
    if sprint_manager is not None:
        index = getattr(sprint_manager, "sprint_index", None)
        if index is not None:
            try:
                return int(index)
            except (TypeError, ValueError):
                pass
    index = context.get("sprint_index", 1)
    try:
        return int(index)
    except (TypeError, ValueError):
        return 1


def _build_prompt(sprint_index: int, roles: List[retro_collect.RoleDocuments]) -> str:
    sections: List[str] = []
    sections.append(
        "System: act as the whole team. Review the provided sprint summaries and handoffs to identify wins, pain points, risks, and follow-up actions."
    )
    sections.append(
        "Respond strictly in JSON with the keys: wins (list of strings), pain_points (list of strings), risks (list of strings), role_instructions (mapping of role to list of action strings), and pre_feature_items (list of objects with title, rationale, suggested_owner, acceptance_hints array)."
    )
    sections.append(f"Sprint number: {sprint_index}")
    sections.append("Sprint documents:")

    if not roles:
        sections.append("(No role summaries or handoffs were provided.)")
    else:
        for role_doc in roles:
            sections.append(f"Role: {role_doc.role}")
            summary = role_doc.summary_text.strip() or "(No summary provided.)"
            handoff = role_doc.handoffs_text.strip() or "(No handoffs provided.)"
            sections.append("Summary:\n" + summary)
            sections.append("Hand-offs:\n" + handoff)
            sections.append("---")

    sections.append(
        "Ensure each role listed receives an entry in role_instructions, even if the list is empty when no action items are required."
    )

    return "\n\n".join(sections)


def _parse_response(response_text: Any) -> Dict[str, Any]:
    if response_text is None:
        raise ValueError("LLM response for retro step was empty.")
    text = str(response_text).strip()
    if not text:
        raise ValueError("LLM response for retro step was empty.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    raise ValueError("Unable to parse LLM response for retro step as JSON.")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
        result = []
        for item in value:
            if isinstance(item, (list, tuple, set)):
                result.extend(_coerce_string_list(list(item)))
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return []


def _normalize_role_instructions(
    raw: Any,
    roles: List[retro_collect.RoleDocuments],
) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    role_lookup = {doc.normalized_role(): doc.role for doc in roles}

    items: List[Tuple[Optional[str], Any]] = []
    if isinstance(raw, dict):
        items = [(key, value) for key, value in raw.items()]
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                role_key = entry.get("role") or entry.get("name")
                values = (
                    entry.get("actions")
                    or entry.get("instructions")
                    or entry.get("items")
                    or entry.get("notes")
                )
                items.append((role_key, values))
    elif raw is None:
        items = []

    for key, values in items:
        if key is None:
            continue
        normalized_key = _normalize_role_key(key)
        canonical_role = role_lookup.get(normalized_key) or str(key).strip()
        actions = _coerce_string_list(values)
        mapping[canonical_role] = actions

    for doc in roles:
        mapping.setdefault(doc.role, [])

    return mapping


def _normalize_backlog_items(raw: Any) -> List[Dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, dict):
        raw_items = [raw]
    elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        raw_items = list(raw)
    else:
        return []

    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        entry = {
            "title": title,
            "rationale": str(item.get("rationale", "")).strip(),
            "suggested_owner": str(item.get("suggested_owner", "")).strip(),
            "acceptance_hints": _coerce_string_list(item.get("acceptance_hints")),
        }
        normalized.append(entry)
    return normalized


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_role_instructions(
    project_root: Path,
    sprint_folder: str,
    roles: List[retro_collect.RoleDocuments],
    instructions: Dict[str, List[str]],
    wins: List[str],
    pain_points: List[str],
    risks: List[str],
    generated_at: str,
) -> Dict[str, Path]:
    instructions_dir = project_root / "ai-inbox" / "sprints" / sprint_folder / "roles"
    instructions_dir.mkdir(parents=True, exist_ok=True)

    role_directories = {doc.role: doc.directory for doc in roles}
    paths: Dict[str, Path] = {}

    all_roles = set(instructions.keys()) | set(role_directories.keys())
    for role in sorted(all_roles):
        directory = role_directories.get(role)
        if directory is None:
            slug = _slugify_role(role)
            directory = instructions_dir / slug
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory.mkdir(parents=True, exist_ok=True)

        actions = instructions.get(role, [])
        content = _render_instruction_markdown(
            role,
            sprint_folder,
            wins,
            pain_points,
            risks,
            actions,
            generated_at,
        )
        output_path = directory / "instructions.md"
        output_path.write_text(content, encoding="utf-8")
        paths[directory.name] = output_path

    return paths


def _append_pre_feature_backlog(
    project_root: Path,
    retro_config: Dict[str, Any],
    backlog_items: List[Dict[str, Any]],
    sprint_index: int,
) -> List[Dict[str, Any]]:
    backlog_path = _resolve_backlog_path(project_root, retro_config)
    backlog_path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, Any]] = []
    if backlog_path.exists():
        try:
            loaded = yaml.safe_load(backlog_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            loaded = None
        if isinstance(loaded, list):
            existing = [entry for entry in loaded if isinstance(entry, dict)]

    sprint_prefix = f"PREF-{sprint_index}-"
    max_seq = 0
    for entry in existing:
        entry_id = str(entry.get("id", ""))
        if entry_id.startswith(sprint_prefix):
            try:
                seq = int(entry_id.split("-")[-1])
            except (TypeError, ValueError):
                continue
            if seq > max_seq:
                max_seq = seq

    sprint_label = f"sprint-{sprint_index}"
    new_entries: List[Dict[str, Any]] = []
    next_seq = max_seq + 1
    for item in backlog_items:
        entry = {
            "id": f"PREF-{sprint_index}-{next_seq}",
            "title": item.get("title", ""),
            "rationale": item.get("rationale", ""),
            "suggested_owner": item.get("suggested_owner", ""),
            "acceptance_hints": item.get("acceptance_hints", []),
            "originated_from": [sprint_label, "retro"],
        }
        existing.append(entry)
        new_entries.append(entry)
        next_seq += 1

    backlog_path.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")
    return new_entries


def _render_instruction_markdown(
    role: str,
    sprint_folder: str,
    wins: List[str],
    pain_points: List[str],
    risks: List[str],
    actions: List[str],
    generated_at: str,
) -> str:
    display_role = _display_role_name(role)

    lines: List[str] = []
    lines.append(f"# Retro Actions for {display_role} ({sprint_folder})")
    lines.append("")
    lines.append(f"Generated on {generated_at}.")
    lines.append("")

    lines.append("## Key Wins")
    lines.extend(_format_bullet_section(wins))
    lines.append("")

    lines.append("## Pain Points")
    lines.extend(_format_bullet_section(pain_points))
    lines.append("")

    lines.append("## Risks")
    lines.extend(_format_bullet_section(risks))
    lines.append("")

    lines.append("## Action Items")
    if actions:
        for item in actions:
            lines.append(f"- {item}")
    else:
        lines.append("_No action items recorded._")

    lines.append("")
    return "\n".join(lines)


def _format_bullet_section(items: List[str]) -> List[str]:
    if not items:
        return ["_None recorded._"]
    return [f"- {item}" for item in items]


def _resolve_backlog_path(project_root: Path, retro_config: Dict[str, Any]) -> Path:
    candidate = retro_config.get("backlog_file")
    if candidate:
        path = Path(candidate)
    else:
        path = Path("ai-inbox/backlog/pre-features.yaml")
    if not path.is_absolute():
        path = project_root / path
    return path


def _display_role_name(role: str) -> str:
    text = str(role or "Team").strip()
    if not text:
        text = "Team"
    return re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ")).title()


def _slugify_role(role: str) -> str:
    normalized = _normalize_role_key(role)
    return normalized or "team"


def _normalize_role_key(role: Any) -> str:
    if role is None:
        return ""
    return str(role).strip().lower().replace(" ", "_").replace("-", "_")
