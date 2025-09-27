"""Utilities for generating deterministic sprint planning artifacts."""<<<<<<< HEAD

"""Utilities for generating deterministic sprint planning artifacts."""

from __future__ import annotations

from __future__ import annotations

import hashlib

import jsonimport hashlib

import randomimport json

import textwrapimport logging

from dataclasses import dataclassimport random

from datetime import datetimeimport textwrap

from pathlib import Pathfrom dataclasses import dataclass

from typing import Dict, Iterable, List, Mapping, Optional, Sequencefrom datetime import datetime

from pathlib import Path

from douglas.domain.sprint import Commitment, SprintPlanfrom typing import Dict, Iterable, List, Mapping, Optional, Sequence

from douglas.logging_utils import get_logger

from douglas.providers.llm_provider import LLMProviderfrom douglas.domain.sprint import Commitment, SprintPlan

from douglas.providers.llm_provider import LLMProvider



logger = get_logger(__name__)

logger = logging.getLogger(__name__)

=======

_SKIPPED_STATUSES = {"""Utilities for transforming planning commitments."""

    "done",

    "finished",from __future__ import annotations

    "complete",

    "completed",from dataclasses import dataclass

    "released",from typing import Dict, List, Mapping, Optional, Sequence

    "archived",

    "canceled",from douglas.logging_utils import get_logger

    "cancelled",

    "duplicate",__all__ = ["Commitment", "filter_commitments"]

    "won't do",

    "wont_do",

    "won't_do",logger = get_logger(__name__)

    "wont do",>>>>>>> main

    "obsolete",

}

_SKIPPED_STATUSES = {

# Maximum number of primary goals to display in sprint planning    "done",

_PRIMARY_GOAL_LIMIT = 3<<<<<<< HEAD

    "finished",

    "complete",

@dataclass    "completed",

class PlanningContext:    "released",

    """Parameters needed to build a sprint plan.    "archived",

    "canceled",

    Args:    "cancelled",

        project_root: Root directory of the project being planned.}

        backlog_state_path: Path to the serialized backlog JSON file.

        sprint_index: Index (1-based) of the sprint being planned._PRIMARY_GOAL_LIMIT = 3

        items_per_sprint: Number of commitments to pull into the sprint.

        seed: Global seed used to derive deterministic selection seeds.

        provider: Optional LLM provider used to draft sprint summaries.@dataclass

        summary_intro: Optional freeform text prepended to generated summaries.class PlanningContext:

        state_dir: Optional override for where plan JSON artifacts are written.    """Parameters needed to build a sprint plan.

        markdown_dir: Optional override for where plan markdown is written.

        backlog_fallback: Optional backlog payload used when state is missing.    Args:

    """        project_root: Root directory of the project being planned.

        backlog_state_path: Path to the serialized backlog JSON file.

    project_root: Path        sprint_index: Index (1-based) of the sprint being planned.

    backlog_state_path: Path        items_per_sprint: Number of commitments to pull into the sprint.

    sprint_index: int        seed: Global seed used to derive deterministic selection seeds.

    items_per_sprint: int = 3        provider: Optional LLM provider used to draft sprint summaries.

    seed: int = 0        summary_intro: Optional freeform text prepended to generated summaries.

    provider: Optional[LLMProvider] = None        state_dir: Optional override for where plan JSON artifacts are written.

    summary_intro: Optional[str] = None        markdown_dir: Optional override for where plan markdown is written.

    state_dir: Optional[Path] = None        backlog_fallback: Optional backlog payload used when state is missing.

    markdown_dir: Optional[Path] = None    """

    backlog_fallback: Optional[Mapping[str, object]] = None

    project_root: Path

    backlog_state_path: Path

@dataclass    sprint_index: int

class PlanningStepResult:    items_per_sprint: int = 3

    """Outcome of running the sprint planning step."""    seed: int = 0

    provider: Optional[LLMProvider] = None

    executed: bool    summary_intro: Optional[str] = None

    success: bool    state_dir: Optional[Path] = None

    reason: str    markdown_dir: Optional[Path] = None

    plan: Optional[SprintPlan] = None    backlog_fallback: Optional[Mapping[str, object]] = None

    json_path: Optional[Path] = None

    markdown_path: Optional[Path] = None

    summary_text: Optional[str] = None@dataclass

    used_fallback: bool = Falseclass PlanningStepResult:

    """Outcome of running the sprint planning step."""

    def summary(self, project_root: Optional[Path] = None) -> Dict[str, object]:

        project_root = project_root or Path.cwd()    executed: bool

        summary: Dict[str, object] = {    success: bool

            "executed": self.executed,    reason: str

            "success": self.success,    plan: Optional[SprintPlan] = None

            "reason": self.reason,    json_path: Optional[Path] = None

        }    markdown_path: Optional[Path] = None

        if self.used_fallback:    summary_text: Optional[str] = None

            summary["used_fallback"] = True    used_fallback: bool = False

        if self.plan is not None:

            summary.update(    def summary(self, project_root: Optional[Path] = None) -> Dict[str, object]:

                {        project_root = project_root or Path.cwd()

                    "sprint": self.plan.sprint_index,        summary: Dict[str, object] = {

                    "goals": list(self.plan.goals),            "executed": self.executed,

                    "commitments": self.plan.commitment_ids,            "success": self.success,

                }            "reason": self.reason,

            )        }

        if self.json_path is not None:        if self.used_fallback:

            summary["json_path"] = _relative_path(self.json_path, project_root)            summary["used_fallback"] = True

        if self.markdown_path is not None:        if self.plan is not None:

            summary["markdown_path"] = _relative_path(self.markdown_path, project_root)            summary.update(

        return summary                {

                    "sprint": self.plan.sprint_index,

                    "goals": list(self.plan.goals),

def _relative_path(path: Path, project_root: Path) -> str:                    "commitments": self.plan.commitment_ids,

    try:                }

        return str(path.relative_to(project_root))            )

    except ValueError:        if self.json_path is not None:

        return str(path)            summary["json_path"] = _relative_path(self.json_path, project_root)

        if self.markdown_path is not None:

            summary["markdown_path"] = _relative_path(self.markdown_path, project_root)

def _selection_seed(seed: int, sprint_index: int, signature: Optional[str]) -> int:        return summary

    material = f"{seed}:{sprint_index}:{signature or ''}".encode("utf-8")

    digest = hashlib.sha256(material).hexdigest()

    return int(digest, 16)def _relative_path(path: Path, project_root: Path) -> str:

    try:

        return str(path.relative_to(project_root))

def _load_backlog(path: Path) -> tuple[List[Mapping[str, object]], Optional[str], str]:    except ValueError:

    try:        return str(path)

        raw_text = path.read_text(encoding="utf-8")

    except FileNotFoundError:

        return [], None, "missing_backlog"def _selection_seed(seed: int, sprint_index: int, signature: Optional[str]) -> int:

    except OSError:    material = f"{seed}:{sprint_index}:{signature or ''}".encode("utf-8")

        return [], None, "backlog_unreadable"    digest = hashlib.sha256(material).hexdigest()

    return int(digest, 16)

    try:

        payload = json.loads(raw_text)

    except json.JSONDecodeError:def _load_backlog(path: Path) -> tuple[List[Mapping[str, object]], Optional[str], str]:

        return [], None, "invalid_backlog"    try:

        raw_text = path.read_text(encoding="utf-8")

    items = payload.get("items")    except FileNotFoundError:

    if not isinstance(items, list):        return [], None, "missing_backlog"

        items = []    except OSError:

    signature = SprintPlan.signature_for_items(items)        return [], None, "backlog_unreadable"

    return items, signature, "ok" if items else "empty_backlog"

    try:

        payload = json.loads(raw_text)

def _sanitize_log_value(value: object) -> str:    except json.JSONDecodeError:

    """Sanitize a value for safe logging by removing control characters."""        return [], None, "invalid_backlog"

    text = str(value)

    sanitized_chars = []    items = payload.get("items")

    for char in text:    if not isinstance(items, list):

        codepoint = ord(char)        items = []

        if char in {"\n", "\r"} or codepoint < 32:    signature = SprintPlan.signature_for_items(items)

            sanitized_chars.append(" ")    return items, signature, "ok" if items else "empty_backlog"

        else:

            sanitized_chars.append(char)

    return "".join(sanitized_chars)def _sanitize_log_value(value: object) -> str:

    text = str(value)

    sanitized_chars = []

def _normalize_fallback_items(    for char in text:

    items: Iterable[Mapping[str, object]]        codepoint = ord(char)

) -> List[Mapping[str, object]]:        if char in {"\n", "\r"} or codepoint < 32:

    normalized_items: List[Mapping[str, object]] = []            sanitized_chars.append(" ")

    for raw in items:        else:

        if not isinstance(raw, Mapping):            sanitized_chars.append(char)

            continue    return "".join(sanitized_chars)

        candidate = dict(raw)

        identifier = (

            candidate.get("id")def _normalize_fallback_items(

            or candidate.get("identifier")    items: Iterable[Mapping[str, object]]

            or candidate.get("external_id")) -> List[Mapping[str, object]]:

        )    normalized_items: List[Mapping[str, object]] = []

        title = candidate.get("title") or candidate.get("name")    for raw in items:

        status = candidate.get("status") or candidate.get("state")        if not isinstance(raw, Mapping):

        if identifier and "id" not in candidate:            continue

            candidate["id"] = identifier        candidate = dict(raw)

        if title and "title" not in candidate:        identifier = (

            candidate["title"] = title            candidate.get("id")

        if status and "status" not in candidate:            or candidate.get("identifier")

            candidate["status"] = status            or candidate.get("external_id")

        normalized_items.append(candidate)        )

    return normalized_items        title = candidate.get("title") or candidate.get("name")

        status = candidate.get("status") or candidate.get("state")

        if identifier and "id" not in candidate:

def _extract_fallback_items(            candidate["id"] = identifier

    data: Optional[Mapping[str, object]]        if title and "title" not in candidate:

) -> List[Mapping[str, object]]:            candidate["title"] = title

    if not isinstance(data, Mapping):        if status and "status" not in candidate:

        return []            candidate["status"] = status

        normalized_items.append(candidate)

    collected: List[Mapping[str, object]] = []    return normalized_items

    seen_ids: set[str] = set()



    def _collect(sequence: object) -> None:def _extract_fallback_items(

        if isinstance(sequence, Sequence):    data: Optional[Mapping[str, object]]

            for normalized in _normalize_fallback_items() -> List[Mapping[str, object]]:

                item for item in sequence if isinstance(item, Mapping)    if not isinstance(data, Mapping):

            ):        return []

                identifier = normalized.get("id")

                if isinstance(identifier, str):    collected: List[Mapping[str, object]] = []

                    if identifier in seen_ids:    seen_ids: set[str] = set()

                        continue

                    seen_ids.add(identifier)    def _collect(sequence: object) -> None:

                collected.append(normalized)        if isinstance(sequence, Sequence):

            for normalized in _normalize_fallback_items(

    for key in ("items", "stories", "tasks", "features", "epics"):                item for item in sequence if isinstance(item, Mapping)

        _collect(data.get(key))            ):

                identifier = normalized.get("id")

    backlog = data.get("backlog")                if isinstance(identifier, str):

    if isinstance(backlog, Mapping):                    if identifier in seen_ids:

        for key in ("items", "stories", "tasks", "features", "epics"):                        continue

            _collect(backlog.get(key))                    seen_ids.add(identifier)

                collected.append(normalized)

    return collected

    for key in ("items", "stories", "tasks", "features", "epics"):

        _collect(data.get(key))

def _coerce_string(value: Optional[object]) -> str:

    if value is None:    backlog = data.get("backlog")

        return ""    if isinstance(backlog, Mapping):

    if isinstance(value, str):        for key in ("items", "stories", "tasks", "features", "epics"):

        return value.strip()            _collect(backlog.get(key))

    return str(value).strip()

    return collected



def _summarize_commitment(data: Mapping[str, object]) -> str:

    identifier = _coerce_string(def _filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:

        data.get("id")    commitments: List[Commitment] = []

        or data.get("commitment_id")    for item in items:

        or data.get("reference")        if not isinstance(item, Mapping):

        or data.get("title")            continue

        or data.get("name")        normalized: Dict[str, object] = dict(item)

    )        status = str(normalized.get("status", "")).strip().lower()

    return identifier or "<unknown>"=======

    "completed",

    "complete",

def _filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:    "cancelled",

    """Return valid commitments, logging when items are discarded."""    "canceled",

    commitments: List[Commitment] = []    "duplicate",

    for index, item in enumerate(items):    "won't do",

        if not isinstance(item, Mapping):    "wont_do",

            logger.warning("Skipping non-mapping commitment at index %d", index)    "won't_do",

            continue    "wont do",

        normalized: Dict[str, object] = dict(item)    "obsolete",

        status = _coerce_string(normalized.get("status")).lower()    "archived",

        if status in _SKIPPED_STATUSES:}

            continue

        try:

            if "title" not in normalized and "name" in normalized:@dataclass(frozen=True)

                normalized["title"] = normalized["name"]class Commitment:

            commitment = Commitment.from_mapping(normalized)    """Serializable representation of a planning commitment entry."""

        except ValueError as exc:

            item_reference = (    identifier: str

                normalized.get("id")    title: str

                or normalized.get("external_id")    status: str = ""

                or normalized.get("title")    description: str = ""

                or normalized.get("name")

                or "<unknown>"    @classmethod

            )    def from_mapping(cls, payload: Mapping[str, object]) -> "Commitment":

            logger.warning(        """Create a commitment from a mapping, validating required fields."""

                "Skipping backlog item %s due to invalid commitment data: %s",

                _sanitize_log_value(item_reference),        title = _coerce_string(payload.get("title"))

                _sanitize_log_value(exc),        if not title:

            )            title = _coerce_string(payload.get("name"))

            continue        if not title:

        commitments.append(commitment)            raise ValueError("commitment requires a title")

    return commitments

        identifier = _coerce_string(

            payload.get("id")

def filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:            or payload.get("commitment_id")

    """Public wrapper around :func:`_filter_commitments`."""            or payload.get("ref")

    return _filter_commitments(items)            or payload.get("reference")

        )



def _select_commitments(        status = _coerce_string(payload.get("status"))

    candidates: List[Commitment],        description_value = payload.get("description")

    seed: int,        description = _normalize_multiline(description_value)

    items_per_sprint: int,

) -> List[Commitment]:        return cls(identifier=identifier, title=title, status=status, description=description)

    if items_per_sprint <= 0 or not candidates:

        return []

    rng = random.Random(seed)def _filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:

    indices = list(range(len(candidates)))    """Return valid commitments, logging when items are discarded."""

    rng.shuffle(indices)

    selected_indices = sorted(indices[:items_per_sprint])    commitments: List[Commitment] = []

    return [candidates[index] for index in selected_indices]    for index, item in enumerate(items):

        if not isinstance(item, Mapping):

            logger.warning("Skipping non-mapping commitment at index %d", index)

def _derive_goals(commitments: Sequence[Commitment], sprint_index: int) -> List[str]:            continue

    if not commitments:        normalized: Dict[str, object] = dict(item)

        return [f"Refine backlog priorities for Sprint {sprint_index}."]        status = _coerce_string(normalized.get("status")).lower()

    goals = [>>>>>>> main

        f"Deliver: {commitment.title}" for commitment in commitments[:_PRIMARY_GOAL_LIMIT]        if status in _SKIPPED_STATUSES:

    ]            continue

    remaining = len(commitments) - _PRIMARY_GOAL_LIMIT        try:

    if remaining > 0:            if "title" not in normalized and "name" in normalized:

        plural = "s" if remaining != 1 else ""                normalized["title"] = normalized["name"]

        goals.append(f"Prepare {remaining} additional backlog item{plural} for upcoming work.")            commitment = Commitment.from_mapping(normalized)

    return goals        except ValueError as exc:

<<<<<<< HEAD

            item_reference = (

def _provider_summary(                normalized.get("id")

    provider: Optional[LLMProvider],                or normalized.get("external_id")

    plan: SprintPlan,                or normalized.get("title")

    intro: Optional[str],                or normalized.get("name")

) -> Optional[str]:                or "<unknown>"

    if provider is None or not plan.commitments:            )

        return intro            logger.warning(

                "Skipping backlog item %s due to invalid commitment data: %s",

    payload = {                _sanitize_log_value(item_reference),

        "sprint": plan.sprint_index,                _sanitize_log_value(exc),

        "goals": plan.goals,=======

        "items_requested": plan.items_requested,            summary = _summarize_commitment(normalized)

        "commitments": [commitment.to_dict() for commitment in plan.commitments],            logger.warning(

    }                "Skipping invalid commitment at index %d (%s): %s",

    prompt = textwrap.dedent(                index,

        f"""                summary,

        You are preparing a sprint planning recap. Using the structured data                exc,

        below, write a concise markdown summary. Avoid code fences and keep the>>>>>>> main

        tone action-oriented.            )

        <plan-data>            continue

        {json.dumps(payload, indent=2, sort_keys=True)}        commitments.append(commitment)

        </plan-data>    return commitments

        """

    ).strip()

<<<<<<< HEAD

    try:def _select_commitments(

        response = provider.generate_code(prompt)    candidates: List[Commitment],

    except Exception:    seed: int,

        return intro    items_per_sprint: int,

) -> List[Commitment]:

    summary = response.strip()    if items_per_sprint <= 0 or not candidates:

    if intro:        return []

        intro_text = intro.strip()    rng = random.Random(seed)

        if intro_text and summary:    indices = list(range(len(candidates)))

            return f"{intro_text}\n\n{summary}"    rng.shuffle(indices)

        if intro_text:    selected_indices = sorted(indices[:items_per_sprint])

            return intro_text    return [candidates[index] for index in selected_indices]

    return summary or intro



def _derive_goals(commitments: Sequence[Commitment], sprint_index: int) -> List[str]:

def run_planning(context: PlanningContext) -> PlanningStepResult:    if not commitments:

    def _apply_fallback_if_needed(items, signature, status, fallback_payload):        return [f"Refine backlog priorities for Sprint {sprint_index}."]

        fallback_items = _extract_fallback_items(fallback_payload)    goals = [

        if fallback_items:        f"Deliver: {commitment.title}" for commitment in commitments[:_PRIMARY_GOAL_LIMIT]

            items = fallback_items    ]

            signature = SprintPlan.signature_for_items(items)    remaining = len(commitments) - _PRIMARY_GOAL_LIMIT

            status = "ok"    if remaining > 0:

            fallback_used = True        plural = "s" if remaining != 1 else ""

        else:        goals.append(f"Prepare {remaining} additional backlog item{plural} for upcoming work.")

            fallback_used = False    return goals

        return items, signature, status, fallback_used, bool(fallback_items)



    items, signature, status = _load_backlog(context.backlog_state_path)def _provider_summary(

    fallback_used = False    provider: Optional[LLMProvider],

    if status in {"missing_backlog", "backlog_unreadable", "invalid_backlog"}:    plan: SprintPlan,

        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(    intro: Optional[str],

            items, signature, status, context.backlog_fallback) -> Optional[str]:

        )    if provider is None or not plan.commitments:

        if not has_fallback:        return intro

            if status == "missing_backlog":

                return PlanningStepResult(False, True, status)    payload = {

            return PlanningStepResult(False, False, status)        "sprint": plan.sprint_index,

    elif status == "empty_backlog":        "goals": plan.goals,

        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(        "items_requested": plan.items_requested,

            items, signature, status, context.backlog_fallback        "commitments": [commitment.to_dict() for commitment in plan.commitments],

        )    }

    prompt = textwrap.dedent(

    filtered = _filter_commitments(items)        f"""

    selection_seed = _selection_seed(context.seed, context.sprint_index, signature)        You are preparing a sprint planning recap. Using the structured data

    requested_count = max(context.items_per_sprint, 0)        below, write a concise markdown summary. Avoid code fences and keep the

    commitments = _select_commitments(        tone action-oriented.

        filtered,        <plan-data>

        selection_seed,        {json.dumps(payload, indent=2, sort_keys=True)}

        requested_count,        </plan-data>

    )        """

    goals = _derive_goals(commitments, context.sprint_index)    ).strip()



    plan = SprintPlan(    try:

        sprint_index=context.sprint_index,        response = provider.generate_code(prompt)

        commitments=commitments,    except Exception:

        goals=goals,        return intro

        items_requested=requested_count,  

        backlog_items_total=len(filtered),    summary = response.strip()

        selection_seed=selection_seed,    if intro:

        backlog_signature=signature,        intro_text = intro.strip()

    )        if intro_text and summary:

            return f"{intro_text}\n\n{summary}"

    summary = _provider_summary(context.provider, plan, context.summary_intro)        if intro_text:

            return intro_text

    state_dir = context.state_dir or (context.project_root / ".douglas" / "state")    return summary or intro

    markdown_dir = context.markdown_dir or (

        context.project_root / "ai-inbox" / "planning"

    )def run_planning(context: PlanningContext) -> PlanningStepResult:

    json_path = state_dir / f"sprint_plan_{context.sprint_index}.json"    def _apply_fallback_if_needed(items, signature, status, fallback_payload):

    markdown_path = markdown_dir / f"sprint_{context.sprint_index}.md"        fallback_items = _extract_fallback_items(fallback_payload)

        if fallback_items:

    previous_generated_at: Optional[str] = None            items = fallback_items

    if json_path.exists():            signature = SprintPlan.signature_for_items(items)

        try:            status = "ok"

            existing_payload = json.loads(json_path.read_text(encoding="utf-8"))            fallback_used = True

        except (OSError, json.JSONDecodeError):        else:

            existing_payload = None            fallback_used = False

        else:        return items, signature, status, fallback_used, bool(fallback_items)

            if isinstance(existing_payload, Mapping):

                raw_timestamp = existing_payload.get("generated_at")    items, signature, status = _load_backlog(context.backlog_state_path)

                if isinstance(raw_timestamp, str):    fallback_used = False

                    previous_generated_at = raw_timestamp    if status in {"missing_backlog", "backlog_unreadable", "invalid_backlog"}:

        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(

    if previous_generated_at:            items, signature, status, context.backlog_fallback

        try:        )

            plan.generated_at = datetime.fromisoformat(previous_generated_at)        if not has_fallback:

        except ValueError:            if status == "missing_backlog":

            pass                return PlanningStepResult(False, True, status)

            return PlanningStepResult(False, False, status)

    try:    elif status == "empty_backlog":

        plan.write_json(json_path)        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(

        plan.write_markdown(markdown_path, summary)            items, signature, status, context.backlog_fallback

    except OSError as exc:        )

        return PlanningStepResult(False, False, f"io_error:{exc}")

    filtered = _filter_commitments(items)

    if commitments:    selection_seed = _selection_seed(context.seed, context.sprint_index, signature)

        reason = "planned_fallback" if fallback_used else "planned"    requested_count = max(context.items_per_sprint, 0)

    elif status == "ok":    commitments = _select_commitments(

        reason = "no_commitments_selected"        filtered,

    else:        selection_seed,

        reason = status        requested_count,

    return PlanningStepResult(    )

        True,    goals = _derive_goals(commitments, context.sprint_index)

        True,

        reason,    plan = SprintPlan(

        plan=plan,        sprint_index=context.sprint_index,

        json_path=json_path,        commitments=commitments,

        markdown_path=markdown_path,        goals=goals,

        summary_text=summary,        items_requested=requested_count,

        used_fallback=fallback_used,        backlog_items_total=len(filtered),

    )        selection_seed=selection_seed,

        backlog_signature=signature,

    )

__all__ = ["PlanningContext", "PlanningStepResult", "run_planning", "filter_commitments"]
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
        except (OSError, json.JSONDecodeError):
            existing_payload = None
        else:
            if isinstance(existing_payload, Mapping):
                raw_timestamp = existing_payload.get("generated_at")
                if isinstance(raw_timestamp, str):
                    previous_generated_at = raw_timestamp

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
=======
def filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:
    """Public wrapper around :func:`_filter_commitments`."""

    return _filter_commitments(items)


def _coerce_string(value: Optional[object]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_multiline(value: Optional[object]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(_coerce_string(part) for part in value if part is not None)
    return str(value).strip()


def _summarize_commitment(data: Mapping[str, object]) -> str:
    identifier = _coerce_string(
        data.get("id")
        or data.get("commitment_id")
        or data.get("reference")
        or data.get("title")
        or data.get("name")
    )
    return identifier or "<unknown>"

>>>>>>> main
