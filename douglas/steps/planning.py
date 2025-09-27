"""Utilities for generating deterministic sprint planning artifacts.""""""Utilities for generating deterministic sprint planning artifacts."""



from __future__ import annotationsfrom __future__ import annotations



import hashlibimport hashlib

import jsonimport json

import randomimport random

import textwrapimport textwrap

from dataclasses import dataclassfrom dataclasses import dataclass

from datetime import datetimefrom datetime import datetime

from pathlib import Pathfrom pathlib import Path

from typing import Dict, Iterable, List, Mapping, Optional, Sequencefrom typing import Dict, Iterable, List, Mapping, Optional, Sequence



from douglas.domain.sprint import Commitment, SprintPlanfrom douglas.domain.sprint import Commitment, SprintPlan

from douglas.logging_utils import get_loggerfrom douglas.logging_utils import get_logger

from douglas.providers.llm_provider import LLMProviderfrom douglas.providers.llm_provider import LLMProvider





logger = get_logger(__name__)logger = get_logger(__name__)





_SKIPPED_STATUSES = {_SKIPPED_STATUSES = {

    "done",    "done",

    "finished",    "finished",

    "complete",    "complete",

    "completed",    "completed",

    "released",    "released",

    "archived",    "archived",

    "canceled",    "canceled",

    "cancelled",    "cancelled",

    "duplicate",    "duplicate",

    "won't do",    "won't do",

    "wont_do",    "wont_do",

    "won't_do",    "won't_do",

    "wont do",    "wont do",

    "obsolete",    "obsolete",

}}



# Maximum number of primary goals to display in sprint planning# Maximum number of primary goals to display in sprint planning

_PRIMARY_GOAL_LIMIT = 3_PRIMARY_GOAL_LIMIT = 3





@dataclass@dataclass

class PlanningContext:class PlanningContext:

    """Parameters needed to build a sprint plan.    """Parameters needed to build a sprint plan.



    Args:    Args:

        project_root: Root directory of the project being planned.        project_root: Root directory of the project being planned.

        backlog_state_path: Path to the serialized backlog JSON file.        backlog_state_path: Path to the serialized backlog JSON file.

        sprint_index: Index (1-based) of the sprint being planned.        sprint_index: Index (1-based) of the sprint being planned.

        items_per_sprint: Number of commitments to pull into the sprint.        items_per_sprint: Number of commitments to pull into the sprint.

        seed: Global seed used to derive deterministic selection seeds.        seed: Global seed used to derive deterministic selection seeds.

        provider: Optional LLM provider used to draft sprint summaries.        provider: Optional LLM provider used to draft sprint summaries.

        summary_intro: Optional freeform text prepended to generated summaries.        summary_intro: Optional freeform text prepended to generated summaries.

        state_dir: Optional override for where plan JSON artifacts are written.        state_dir: Optional override for where plan JSON artifacts are written.

        markdown_dir: Optional override for where plan markdown is written.        markdown_dir: Optional override for where plan markdown is written.

        backlog_fallback: Optional backlog payload used when state is missing.        backlog_fallback: Optional backlog payload used when state is missing.

    """    """



    project_root: Path    project_root: Path

    backlog_state_path: Path    backlog_state_path: Path

    sprint_index: int    sprint_index: int

    items_per_sprint: int = 3    items_per_sprint: int = 3

    seed: int = 0    seed: int = 0

    provider: Optional[LLMProvider] = None    provider: Optional[LLMProvider] = None

    summary_intro: Optional[str] = None    summary_intro: Optional[str] = None

    state_dir: Optional[Path] = None    state_dir: Optional[Path] = None

    markdown_dir: Optional[Path] = None    markdown_dir: Optional[Path] = None

    backlog_fallback: Optional[Mapping[str, object]] = None    backlog_fallback: Optional[Mapping[str, object]] = None





@dataclass@dataclass

class PlanningStepResult:class PlanningStepResult:

    """Outcome of running the sprint planning step."""    """Outcome of running the sprint planning step."""



    executed: bool    executed: bool

    success: bool    success: bool

    reason: str    reason: str

    plan: Optional[SprintPlan] = None    plan: Optional[SprintPlan] = None

    json_path: Optional[Path] = None    json_path: Optional[Path] = None

    markdown_path: Optional[Path] = None    markdown_path: Optional[Path] = None

    summary_text: Optional[str] = None    summary_text: Optional[str] = None

    used_fallback: bool = False    used_fallback: bool = False



    def summary(self, project_root: Optional[Path] = None) -> Dict[str, object]:    def summary(self, project_root: Optional[Path] = None) -> Dict[str, object]:

        project_root = project_root or Path.cwd()        project_root = project_root or Path.cwd()

        summary: Dict[str, object] = {        summary: Dict[str, object] = {

            "executed": self.executed,            "executed": self.executed,

            "success": self.success,            "success": self.success,

            "reason": self.reason,            "reason": self.reason,

        }        }

        if self.used_fallback:        if self.used_fallback:

            summary["used_fallback"] = True            summary["used_fallback"] = True

        if self.plan is not None:        if self.plan is not None:

            summary.update(            summary.update(

                {                {

                    "sprint": self.plan.sprint_index,                    "sprint": self.plan.sprint_index,

                    "goals": list(self.plan.goals),                    "goals": list(self.plan.goals),

                    "commitments": self.plan.commitment_ids,                    "commitments": self.plan.commitment_ids,

                }                }

            )            )

        if self.json_path is not None:        if self.json_path is not None:

            summary["json_path"] = _relative_path(self.json_path, project_root)            summary["json_path"] = _relative_path(self.json_path, project_root)

        if self.markdown_path is not None:        if self.markdown_path is not None:

            summary["markdown_path"] = _relative_path(self.markdown_path, project_root)            summary["markdown_path"] = _relative_path(self.markdown_path, project_root)

        return summary        return summary





def _relative_path(path: Path, project_root: Path) -> str:def _relative_path(path: Path, project_root: Path) -> str:

    try:    try:

        return str(path.relative_to(project_root))        return str(path.relative_to(project_root))

    except ValueError:    except ValueError:

        return str(path)        return str(path)





def _selection_seed(seed: int, sprint_index: int, signature: Optional[str]) -> int:def _selection_seed(seed: int, sprint_index: int, signature: Optional[str]) -> int:

    material = f"{seed}:{sprint_index}:{signature or ''}".encode("utf-8")    material = f"{seed}:{sprint_index}:{signature or ''}".encode("utf-8")

    digest = hashlib.sha256(material).hexdigest()    digest = hashlib.sha256(material).hexdigest()

    return int(digest, 16)    return int(digest, 16)





def _load_backlog(path: Path) -> tuple[List[Mapping[str, object]], Optional[str], str]:def _load_backlog(path: Path) -> tuple[List[Mapping[str, object]], Optional[str], str]:

    try:    try:

        raw_text = path.read_text(encoding="utf-8")        raw_text = path.read_text(encoding="utf-8")

    except FileNotFoundError:    except FileNotFoundError:

        return [], None, "missing_backlog"        return [], None, "missing_backlog"

    except OSError:    except OSError:

        return [], None, "backlog_unreadable"        return [], None, "backlog_unreadable"



    try:    try:

        payload = json.loads(raw_text)        payload = json.loads(raw_text)

    except json.JSONDecodeError:    except json.JSONDecodeError:

        return [], None, "invalid_backlog"        return [], None, "invalid_backlog"



    items = payload.get("items")    items = payload.get("items")

    if not isinstance(items, list):    if not isinstance(items, list):

        items = []        items = []

    signature = SprintPlan.signature_for_items(items)    signature = SprintPlan.signature_for_items(items)

    return items, signature, "ok" if items else "empty_backlog"    return items, signature, "ok" if items else "empty_backlog"





def _sanitize_log_value(value: object) -> str:def _sanitize_log_value(value: object) -> str:

    """Sanitize a value for safe logging by removing control characters."""    text = str(value)

    text = str(value)    sanitized_chars = []

    sanitized_chars = []    for char in text:

    for char in text:        codepoint = ord(char)

        codepoint = ord(char)        if char in {"\n", "\r"} or codepoint < 32:

        if char in {"\n", "\r"} or codepoint < 32:            sanitized_chars.append(" ")

            sanitized_chars.append(" ")        else:

        else:            sanitized_chars.append(char)

            sanitized_chars.append(char)    return "".join(sanitized_chars)

    return "".join(sanitized_chars)



def _normalize_fallback_items(

def _normalize_fallback_items(    items: Iterable[Mapping[str, object]]

    items: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:

) -> List[Mapping[str, object]]:    normalized_items: List[Mapping[str, object]] = []

    normalized_items: List[Mapping[str, object]] = []    for raw in items:

    for raw in items:        if not isinstance(raw, Mapping):

        if not isinstance(raw, Mapping):            continue

            continue        candidate = dict(raw)

        candidate = dict(raw)        identifier = (

        identifier = (            candidate.get("id")

            candidate.get("id")            or candidate.get("identifier")

            or candidate.get("identifier")            or candidate.get("external_id")

            or candidate.get("external_id")        )

        )        title = candidate.get("title") or candidate.get("name")

        title = candidate.get("title") or candidate.get("name")        status = candidate.get("status") or candidate.get("state")

        status = candidate.get("status") or candidate.get("state")        if identifier and "id" not in candidate:

        if identifier and "id" not in candidate:            candidate["id"] = identifier

            candidate["id"] = identifier        if title and "title" not in candidate:

        if title and "title" not in candidate:            candidate["title"] = title

            candidate["title"] = title        if status and "status" not in candidate:

        if status and "status" not in candidate:            candidate["status"] = status

            candidate["status"] = status        normalized_items.append(candidate)

        normalized_items.append(candidate)    return normalized_items

    return normalized_items



def _extract_fallback_items(

def _extract_fallback_items(    data: Optional[Mapping[str, object]]

    data: Optional[Mapping[str, object]]) -> List[Mapping[str, object]]:

) -> List[Mapping[str, object]]:    if not isinstance(data, Mapping):

    if not isinstance(data, Mapping):        return []

        return []

    collected: List[Mapping[str, object]] = []

    collected: List[Mapping[str, object]] = []    seen_ids: set[str] = set()

    seen_ids: set[str] = set()

    def _collect(sequence: object) -> None:

    def _collect(sequence: object) -> None:        if isinstance(sequence, Sequence):

        if isinstance(sequence, Sequence):            for normalized in _normalize_fallback_items(

            for normalized in _normalize_fallback_items(                item for item in sequence if isinstance(item, Mapping)

                item for item in sequence if isinstance(item, Mapping)            ):

            ):                identifier = normalized.get("id")

                identifier = normalized.get("id")                if isinstance(identifier, str):

                if isinstance(identifier, str):                    if identifier in seen_ids:

                    if identifier in seen_ids:                        continue

                        continue                    seen_ids.add(identifier)

                    seen_ids.add(identifier)                collected.append(normalized)

                collected.append(normalized)

    for key in ("items", "stories", "tasks", "features", "epics"):

    for key in ("items", "stories", "tasks", "features", "epics"):        _collect(data.get(key))

        _collect(data.get(key))

    backlog = data.get("backlog")

    backlog = data.get("backlog")    if isinstance(backlog, Mapping):

    if isinstance(backlog, Mapping):        for key in ("items", "stories", "tasks", "features", "epics"):

        for key in ("items", "stories", "tasks", "features", "epics"):            _collect(backlog.get(key))

            _collect(backlog.get(key))

    return collected

    return collected



def _filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:

def _coerce_string(value: Optional[object]) -> str:    commitments: List[Commitment] = []

    if value is None:    for item in items:

        return ""        if not isinstance(item, Mapping):

    if isinstance(value, str):            continue

        return value.strip()        normalized: Dict[str, object] = dict(item)

    return str(value).strip()        status = str(normalized.get("status", "")).strip().lower()

=======

    "completed",

def _normalize_multiline(value: Optional[object]) -> str:    "complete",

    if value is None:    "cancelled",

        return ""    "canceled",

    if isinstance(value, str):    "duplicate",

        return value.strip()    "won't do",

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):    "wont_do",

        return "\n".join(_coerce_string(part) for part in value if part is not None)    "won't_do",

    return str(value).strip()    "wont do",

    "obsolete",

    "archived",

def _summarize_commitment(data: Mapping[str, object]) -> str:}

    identifier = _coerce_string(

        data.get("id")

        or data.get("commitment_id")@dataclass(frozen=True)

        or data.get("reference")class Commitment:

        or data.get("title")    """Serializable representation of a planning commitment entry."""

        or data.get("name")

    )    identifier: str

    return identifier or "<unknown>"    title: str

    status: str = ""

    description: str = ""

def _filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:

    """Return valid commitments, logging when items are discarded."""    @classmethod

    commitments: List[Commitment] = []    def from_mapping(cls, payload: Mapping[str, object]) -> "Commitment":

    for index, item in enumerate(items):        """Create a commitment from a mapping, validating required fields."""

        if not isinstance(item, Mapping):

            logger.warning("Skipping non-mapping commitment at index %d", index)        title = _coerce_string(payload.get("title"))

            continue        if not title:

        normalized: Dict[str, object] = dict(item)            title = _coerce_string(payload.get("name"))

        status = _coerce_string(normalized.get("status")).lower()        if not title:

        if status in _SKIPPED_STATUSES:            raise ValueError("commitment requires a title")

            continue

        try:        identifier = _coerce_string(

            if "title" not in normalized and "name" in normalized:            payload.get("id")

                normalized["title"] = normalized["name"]            or payload.get("commitment_id")

            commitment = Commitment.from_mapping(normalized)            or payload.get("ref")

        except ValueError as exc:            or payload.get("reference")

            item_reference = (        )

                normalized.get("id")

                or normalized.get("external_id")        status = _coerce_string(payload.get("status"))

                or normalized.get("title")        description_value = payload.get("description")

                or normalized.get("name")        description = _normalize_multiline(description_value)

                or "<unknown>"

            )        return cls(identifier=identifier, title=title, status=status, description=description)

            logger.warning(

                "Skipping backlog item %s due to invalid commitment data: %s",

                _sanitize_log_value(item_reference),def _filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:

                _sanitize_log_value(exc),    """Return valid commitments, logging when items are discarded."""

            )

            continue    commitments: List[Commitment] = []

        commitments.append(commitment)    for index, item in enumerate(items):

    return commitments        if not isinstance(item, Mapping):

            logger.warning("Skipping non-mapping commitment at index %d", index)

            continue

def filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:        normalized: Dict[str, object] = dict(item)

    """Public wrapper around :func:`_filter_commitments`."""        status = _coerce_string(normalized.get("status")).lower()

    return _filter_commitments(items)>>>>>>> main

        if status in _SKIPPED_STATUSES:

            continue

def _select_commitments(        try:

    candidates: List[Commitment],            if "title" not in normalized and "name" in normalized:

    seed: int,                normalized["title"] = normalized["name"]

    items_per_sprint: int,            commitment = Commitment.from_mapping(normalized)

) -> List[Commitment]:        except ValueError as exc:

    if items_per_sprint <= 0 or not candidates:<<<<<<< HEAD

        return []            item_reference = (

    rng = random.Random(seed)                normalized.get("id")

    indices = list(range(len(candidates)))                or normalized.get("external_id")

    rng.shuffle(indices)                or normalized.get("title")

    selected_indices = sorted(indices[:items_per_sprint])                or normalized.get("name")

    return [candidates[index] for index in selected_indices]                or "<unknown>"

            )

            logger.warning(

def _derive_goals(commitments: Sequence[Commitment], sprint_index: int) -> List[str]:                "Skipping backlog item %s due to invalid commitment data: %s",

    if not commitments:                _sanitize_log_value(item_reference),

        return [f"Refine backlog priorities for Sprint {sprint_index}."]                _sanitize_log_value(exc),

    goals = [=======

        f"Deliver: {commitment.title}" for commitment in commitments[:_PRIMARY_GOAL_LIMIT]            summary = _summarize_commitment(normalized)

    ]            logger.warning(

    remaining = len(commitments) - _PRIMARY_GOAL_LIMIT                "Skipping invalid commitment at index %d (%s): %s",

    if remaining > 0:                index,

        plural = "s" if remaining != 1 else ""                summary,

        goals.append(f"Prepare {remaining} additional backlog item{plural} for upcoming work.")                exc,

    return goals>>>>>>> main

            )

            continue

def _provider_summary(        commitments.append(commitment)

    provider: Optional[LLMProvider],    return commitments

    plan: SprintPlan,

    intro: Optional[str],

) -> Optional[str]:<<<<<<< HEAD

    if provider is None or not plan.commitments:def _select_commitments(

        return intro    candidates: List[Commitment],

    seed: int,

    payload = {    items_per_sprint: int,

        "sprint": plan.sprint_index,) -> List[Commitment]:

        "goals": plan.goals,    if items_per_sprint <= 0 or not candidates:

        "items_requested": plan.items_requested,        return []

        "commitments": [commitment.to_dict() for commitment in plan.commitments],    rng = random.Random(seed)

    }    indices = list(range(len(candidates)))

    prompt = textwrap.dedent(    rng.shuffle(indices)

        f"""    selected_indices = sorted(indices[:items_per_sprint])

        You are preparing a sprint planning recap. Using the structured data    return [candidates[index] for index in selected_indices]

        below, write a concise markdown summary. Avoid code fences and keep the

        tone action-oriented.

        <plan-data>def _derive_goals(commitments: Sequence[Commitment], sprint_index: int) -> List[str]:

        {json.dumps(payload, indent=2, sort_keys=True)}    if not commitments:

        </plan-data>        return [f"Refine backlog priorities for Sprint {sprint_index}."]

        """    goals = [

    ).strip()        f"Deliver: {commitment.title}" for commitment in commitments[:_PRIMARY_GOAL_LIMIT]

    ]

    try:    remaining = len(commitments) - _PRIMARY_GOAL_LIMIT

        response = provider.generate_code(prompt)    if remaining > 0:

    except Exception:        plural = "s" if remaining != 1 else ""

        return intro        goals.append(f"Prepare {remaining} additional backlog item{plural} for upcoming work.")

    return goals

    summary = response.strip()

    if intro:

        intro_text = intro.strip()def _provider_summary(

        if intro_text and summary:    provider: Optional[LLMProvider],

            return f"{intro_text}\n\n{summary}"    plan: SprintPlan,

        if intro_text:    intro: Optional[str],

            return intro_text) -> Optional[str]:

    return summary or intro    if provider is None or not plan.commitments:

        return intro



def run_planning(context: PlanningContext) -> PlanningStepResult:    payload = {

    def _apply_fallback_if_needed(items, signature, status, fallback_payload):        "sprint": plan.sprint_index,

        fallback_items = _extract_fallback_items(fallback_payload)        "goals": plan.goals,

        if fallback_items:        "items_requested": plan.items_requested,

            items = fallback_items        "commitments": [commitment.to_dict() for commitment in plan.commitments],

            signature = SprintPlan.signature_for_items(items)    }

            status = "ok"    prompt = textwrap.dedent(

            fallback_used = True        f"""

        else:        You are preparing a sprint planning recap. Using the structured data

            fallback_used = False        below, write a concise markdown summary. Avoid code fences and keep the

        return items, signature, status, fallback_used, bool(fallback_items)        tone action-oriented.

        <plan-data>

    items, signature, status = _load_backlog(context.backlog_state_path)        {json.dumps(payload, indent=2, sort_keys=True)}

    fallback_used = False        </plan-data>

    if status in {"missing_backlog", "backlog_unreadable", "invalid_backlog"}:        """

        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(    ).strip()

            items, signature, status, context.backlog_fallback

        )    try:

        if not has_fallback:        response = provider.generate_code(prompt)

            if status == "missing_backlog":    except Exception:

                return PlanningStepResult(False, True, status)        return intro

            return PlanningStepResult(False, False, status)

    elif status == "empty_backlog":    summary = response.strip()

        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(    if intro:

            items, signature, status, context.backlog_fallback        intro_text = intro.strip()

        )        if intro_text and summary:

            return f"{intro_text}\n\n{summary}"

    filtered = _filter_commitments(items)        if intro_text:

    selection_seed = _selection_seed(context.seed, context.sprint_index, signature)            return intro_text

    requested_count = max(context.items_per_sprint, 0)    return summary or intro

    commitments = _select_commitments(

        filtered,

        selection_seed,def run_planning(context: PlanningContext) -> PlanningStepResult:

        requested_count,    def _apply_fallback_if_needed(items, signature, status, fallback_payload):

    )        fallback_items = _extract_fallback_items(fallback_payload)

    goals = _derive_goals(commitments, context.sprint_index)        if fallback_items:

            items = fallback_items

    plan = SprintPlan(            signature = SprintPlan.signature_for_items(items)

        sprint_index=context.sprint_index,            status = "ok"

        commitments=commitments,            fallback_used = True

        goals=goals,        else:

        items_requested=requested_count,            fallback_used = False

        backlog_items_total=len(filtered),        return items, signature, status, fallback_used, bool(fallback_items)

        selection_seed=selection_seed,

        backlog_signature=signature,    items, signature, status = _load_backlog(context.backlog_state_path)

    )    fallback_used = False

    if status in {"missing_backlog", "backlog_unreadable", "invalid_backlog"}:

    summary = _provider_summary(context.provider, plan, context.summary_intro)        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(

            items, signature, status, context.backlog_fallback

    state_dir = context.state_dir or (context.project_root / ".douglas" / "state")        )

    markdown_dir = context.markdown_dir or (        if not has_fallback:

        context.project_root / "ai-inbox" / "planning"            if status == "missing_backlog":

    )                return PlanningStepResult(False, True, status)

    json_path = state_dir / f"sprint_plan_{context.sprint_index}.json"            return PlanningStepResult(False, False, status)

    markdown_path = markdown_dir / f"sprint_{context.sprint_index}.md"    elif status == "empty_backlog":

        items, signature, status, fallback_used, has_fallback = _apply_fallback_if_needed(

    previous_generated_at: Optional[str] = None            items, signature, status, context.backlog_fallback

    if json_path.exists():        )

        try:

            existing_payload = json.loads(json_path.read_text(encoding="utf-8"))    filtered = _filter_commitments(items)

        except (OSError, json.JSONDecodeError):    selection_seed = _selection_seed(context.seed, context.sprint_index, signature)

            existing_payload = None    requested_count = max(context.items_per_sprint, 0)

        else:    commitments = _select_commitments(

            if isinstance(existing_payload, Mapping):        filtered,

                raw_timestamp = existing_payload.get("generated_at")        selection_seed,

                if isinstance(raw_timestamp, str):        requested_count,

                    previous_generated_at = raw_timestamp    )

    goals = _derive_goals(commitments, context.sprint_index)

    if previous_generated_at:

        try:    plan = SprintPlan(

            plan.generated_at = datetime.fromisoformat(previous_generated_at)        sprint_index=context.sprint_index,

        except ValueError:        commitments=commitments,

            pass        goals=goals,

        items_requested=requested_count,

    try:        backlog_items_total=len(filtered),

        plan.write_json(json_path)        selection_seed=selection_seed,

        plan.write_markdown(markdown_path, summary)        backlog_signature=signature,

    except OSError as exc:    )

        return PlanningStepResult(False, False, f"io_error:{exc}")

    summary = _provider_summary(context.provider, plan, context.summary_intro)

    if commitments:

        reason = "planned_fallback" if fallback_used else "planned"    state_dir = context.state_dir or (context.project_root / ".douglas" / "state")

    elif status == "ok":    markdown_dir = context.markdown_dir or (

        reason = "no_commitments_selected"        context.project_root / "ai-inbox" / "planning"

    else:    )

        reason = status    json_path = state_dir / f"sprint_plan_{context.sprint_index}.json"

    return PlanningStepResult(    markdown_path = markdown_dir / f"sprint_{context.sprint_index}.md"

        True,

        True,    previous_generated_at: Optional[str] = None

        reason,    if json_path.exists():

        plan=plan,        try:

        json_path=json_path,            existing_payload = json.loads(json_path.read_text(encoding="utf-8"))

        markdown_path=markdown_path,        except (OSError, json.JSONDecodeError):

        summary_text=summary,            existing_payload = None

        used_fallback=fallback_used,        else:

    )            if isinstance(existing_payload, Mapping):

                raw_timestamp = existing_payload.get("generated_at")

                if isinstance(raw_timestamp, str):

__all__ = ["PlanningContext", "PlanningStepResult", "run_planning", "filter_commitments"]                    previous_generated_at = raw_timestamp

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
