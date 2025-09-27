"""Utilities for generating deterministic sprint planning artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from douglas.domain.sprint import Commitment, SprintPlan
from douglas.providers.llm_provider import LLMProvider


logger = logging.getLogger(__name__)


_SKIPPED_STATUSES = {
    "done",
    "finished",
    "complete",
    "completed",
    "released",
    "archived",
    "canceled",
    "cancelled",
}

_PRIMARY_GOAL_LIMIT = 3


@dataclass
class PlanningContext:
    """Parameters needed to build a sprint plan.

    Args:
        project_root: Root directory of the project being planned.
        backlog_state_path: Path to the serialized backlog JSON file.
        sprint_index: Index (1-based) of the sprint being planned.
        items_per_sprint: Number of commitments to pull into the sprint.
        seed: Global seed used to derive deterministic selection seeds.
        provider: Optional LLM provider used to draft sprint summaries.
        summary_intro: Optional freeform text prepended to generated summaries.
        state_dir: Optional override for where plan JSON artifacts are written.
        markdown_dir: Optional override for where plan markdown is written.
        backlog_fallback: Optional backlog payload used when state is missing.
    """

    project_root: Path
    backlog_state_path: Path
    sprint_index: int
    items_per_sprint: int = 3
    seed: int = 0
    provider: Optional[LLMProvider] = None
    summary_intro: Optional[str] = None
    state_dir: Optional[Path] = None
    markdown_dir: Optional[Path] = None
    backlog_fallback: Optional[Mapping[str, object]] = None


@dataclass
class PlanningStepResult:
    """Outcome of running the sprint planning step."""

    executed: bool
    success: bool
    reason: str
    plan: Optional[SprintPlan] = None
    json_path: Optional[Path] = None
    markdown_path: Optional[Path] = None
    summary_text: Optional[str] = None
    used_fallback: bool = False

    def summary(self, project_root: Optional[Path] = None) -> Dict[str, object]:
        project_root = project_root or Path.cwd()
        summary: Dict[str, object] = {
            "executed": self.executed,
            "success": self.success,
            "reason": self.reason,
        }
        if self.used_fallback:
            summary["used_fallback"] = True
        if self.plan is not None:
            summary.update(
                {
                    "sprint": self.plan.sprint_index,
                    "goals": list(self.plan.goals),
                    "commitments": self.plan.commitment_ids,
                }
            )
        if self.json_path is not None:
            summary["json_path"] = _relative_path(self.json_path, project_root)
        if self.markdown_path is not None:
            summary["markdown_path"] = _relative_path(self.markdown_path, project_root)
        return summary


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _selection_seed(seed: int, sprint_index: int, signature: Optional[str]) -> int:
    material = f"{seed}:{sprint_index}:{signature or ''}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16)


def _load_backlog(path: Path) -> tuple[List[Mapping[str, object]], Optional[str], str]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], None, "missing_backlog"
    except OSError:
        return [], None, "backlog_unreadable"

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return [], None, "invalid_backlog"

    items = payload.get("items")
    if not isinstance(items, list):
        items = []
    signature = SprintPlan.signature_for_items(items)
    return items, signature, "ok" if items else "empty_backlog"


def _sanitize_log_value(value: object) -> str:
    text = str(value)
    return text.replace("\n", " ").replace("\r", " ")


def _normalize_fallback_items(
    items: Iterable[Mapping[str, object]]
) -> List[Mapping[str, object]]:
    normalized_items: List[Mapping[str, object]] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        candidate = dict(raw)
        identifier = (
            candidate.get("id")
            or candidate.get("identifier")
            or candidate.get("external_id")
        )
        title = candidate.get("title") or candidate.get("name")
        status = candidate.get("status") or candidate.get("state")
        if identifier and "id" not in candidate:
            candidate["id"] = identifier
        if title and "title" not in candidate:
            candidate["title"] = title
        if status and "status" not in candidate:
            candidate["status"] = status
        normalized_items.append(candidate)
    return normalized_items


def _extract_fallback_items(
    data: Optional[Mapping[str, object]]
) -> List[Mapping[str, object]]:
    if not isinstance(data, Mapping):
        return []

    collected: List[Mapping[str, object]] = []
    seen_ids: set[str] = set()

    def _collect(sequence: object) -> None:
        if isinstance(sequence, Sequence):
            for normalized in _normalize_fallback_items(
                item for item in sequence if isinstance(item, Mapping)
            ):
                identifier = normalized.get("id")
                if isinstance(identifier, str):
                    if identifier in seen_ids:
                        continue
                    seen_ids.add(identifier)
                collected.append(normalized)

    for key in ("items", "stories", "tasks", "features", "epics"):
        _collect(data.get(key))

    backlog = data.get("backlog")
    if isinstance(backlog, Mapping):
        for key in ("items", "stories", "tasks", "features", "epics"):
            _collect(backlog.get(key))

    return collected


def _filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:
    commitments: List[Commitment] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        normalized: Dict[str, object] = dict(item)
        status = str(normalized.get("status", "")).strip().lower()
        if status in _SKIPPED_STATUSES:
            continue
        try:
            if "title" not in normalized and "name" in normalized:
                normalized["title"] = normalized["name"]
            commitment = Commitment.from_mapping(normalized)
        except ValueError as exc:
            item_reference = (
                normalized.get("id")
                or normalized.get("external_id")
                or normalized.get("title")
                or normalized.get("name")
                or "<unknown>"
            )
            logger.warning(
                "Skipping backlog item %s due to invalid commitment data: %s",
                _sanitize_log_value(item_reference),
                _sanitize_log_value(exc),
            )
            continue
        commitments.append(commitment)
    return commitments


def _select_commitments(
    candidates: List[Commitment],
    seed: int,
    items_per_sprint: int,
) -> List[Commitment]:
    if items_per_sprint <= 0 or not candidates:
        return []
    rng = random.Random(seed)
    indices = list(range(len(candidates)))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:items_per_sprint])
    return [candidates[index] for index in selected_indices]


def _derive_goals(commitments: Sequence[Commitment], sprint_index: int) -> List[str]:
    if not commitments:
        return [f"Refine backlog priorities for Sprint {sprint_index}."]
    goals = [
        f"Deliver: {commitment.title}" for commitment in commitments[:_PRIMARY_GOAL_LIMIT]
    ]
    remaining = len(commitments) - _PRIMARY_GOAL_LIMIT
    if remaining > 0:
        plural = "s" if remaining != 1 else ""
        goals.append(f"Prepare {remaining} additional backlog item{plural} for upcoming work.")
    return goals


def _provider_summary(
    provider: Optional[LLMProvider],
    plan: SprintPlan,
    intro: Optional[str],
) -> Optional[str]:
    if provider is None or not plan.commitments:
        return intro

    payload = {
        "sprint": plan.sprint_index,
        "goals": plan.goals,
        "items_requested": plan.items_requested,
        "commitments": [commitment.to_dict() for commitment in plan.commitments],
    }
    prompt = textwrap.dedent(
        f"""
        You are preparing a sprint planning recap. Using the structured data
        below, write a concise markdown summary. Avoid code fences and keep the
        tone action-oriented.
        <plan-data>
        {json.dumps(payload, indent=2, sort_keys=True)}
        </plan-data>
        """
    ).strip()

    try:
        response = provider.generate_code(prompt)
    except Exception:
        return intro

    summary = response.strip()
    if intro:
        intro_text = intro.strip()
        if intro_text and summary:
            return f"{intro_text}\n\n{summary}"
        if intro_text:
            return intro_text
    return summary or intro


def run_planning(context: PlanningContext) -> PlanningStepResult:
    def _apply_fallback_if_needed(items, signature, status, fallback_payload):
        fallback_items = _extract_fallback_items(fallback_payload)
        if fallback_items:
            items = fallback_items
            signature = SprintPlan.signature_for_items(items)
            status = "ok"
            fallback_used = True
        else:
            fallback_used = False
        return items, signature, status, fallback_used, bool(fallback_items)

    items, signature, status = _load_backlog(context.backlog_state_path)
    fallback_used = False
    if status in {"missing_backlog", "backlog_unreadable", "invalid_backlog"}:
        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(
            items, signature, status, context.backlog_fallback
        )
        if not has_fallback:
            if status == "missing_backlog":
                return PlanningStepResult(False, True, status)
            return PlanningStepResult(False, False, status)
    elif status == "empty_backlog":
        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(
            items, signature, status, context.backlog_fallback
        )
    selection_seed = _selection_seed(context.seed, context.sprint_index, signature)
    requested_count = max(context.items_per_sprint, 0)
    commitments = _select_commitments(
        filtered,
        selection_seed,
        requested_count,
    )
    goals = _derive_goals(commitments, context.sprint_index)

    plan = SprintPlan(
        sprint_index=context.sprint_index,
        commitments=commitments,
        goals=goals,
        items_requested=requested_count,
        backlog_items_total=len(filtered),
        selection_seed=selection_seed,
        backlog_signature=signature,
    )

    summary = _provider_summary(context.provider, plan, context.summary_intro)

    state_dir = context.state_dir or (context.project_root / ".douglas" / "state")
    markdown_dir = context.markdown_dir or (
        context.project_root / "ai-inbox" / "planning"
    )
    json_path = state_dir / f"sprint_plan_{context.sprint_index}.json"
    markdown_path = markdown_dir / f"sprint_{context.sprint_index}.md"

    previous_generated_at: Optional[str] = None
    if json_path.exists():
        try:
            existing_payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(existing_payload, Mapping):
                raw_timestamp = existing_payload.get("generated_at")
                if isinstance(raw_timestamp, str):
                    previous_generated_at = raw_timestamp
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            previous_generated_at = None

    if previous_generated_at:
        try:
            plan.generated_at = datetime.fromisoformat(previous_generated_at)
        except ValueError:
            pass

    try:
        plan.write_json(json_path)
        plan.write_markdown(markdown_path, summary)
    except OSError as exc:
        return PlanningStepResult(False, False, f"io_error:{exc}")

    if commitments:
        reason = "planned_fallback" if fallback_used else "planned"
    elif status == "ok":
        reason = "no_commitments_selected"
    else:
        reason = status
    return PlanningStepResult(
        True,
        True,
        reason,
        plan=plan,
        json_path=json_path,
        markdown_path=markdown_path,
        summary_text=summary,
        used_fallback=fallback_used,
    )


__all__ = ["PlanningContext", "PlanningStepResult", "run_planning"]
