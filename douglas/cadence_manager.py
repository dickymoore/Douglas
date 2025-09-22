"""Cadence management utilities for role-based scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from douglas.sprint_manager import CadenceDecision

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from douglas.sprint_manager import SprintManager


@dataclass
class CadenceContext:
    """Snapshot of cadence state for a specific loop step."""

    step_name: str
    role: str
    activity: str
    cadence_value: Optional[object]
    cadence_source: str
    sprint_manager: "SprintManager"
    sprint_day: int
    sprint_length: int
    run_count: int
    per_sprint_consumed: int
    available_events: Dict[str, int]
    decision: Optional[CadenceDecision] = None


def should_run_step(role: str, activity: str, context: CadenceContext) -> bool:
    """Evaluate whether a step should run for the provided role and activity."""

    decision = context.sprint_manager.should_run_step(
        context.step_name, context.cadence_value
    )

    # When cadence is configured at the role level, surface the role/activity in logs.
    if context.cadence_source == "role":
        prefix = f"{role} cadence for {activity}"
        message = decision.reason or ""
        if decision.should_run:
            reason = message or "eligible for this iteration."
            decision = CadenceDecision(True, f"{prefix}: {reason}", decision.event_type)
        else:
            reason = message or "defers execution until the schedule permits."
            decision = CadenceDecision(
                False, f"{prefix} defers execution: {reason}", decision.event_type
            )

    # Record the decision for callers and tests.
    context.decision = decision
    return decision.should_run


class CadenceManager:
    """Determines when steps execute based on role-aware cadence rules."""

    _DEFAULT_STEP_METADATA: Dict[str, Tuple[str, str]] = {
        "generate": ("Developer", "development"),
        "lint": ("Developer", "quality_checks"),
        "typecheck": ("Developer", "quality_checks"),
        "test": ("Tester", "test_cases"),
        "review": ("Developer", "code_review"),
        "feature_refinement": ("ProductOwner", "backlog_refinement"),
        "refine": ("ProductOwner", "backlog_refinement"),
        "demo": ("ProductOwner", "sprint_review"),
        "retrospective": ("ScrumMaster", "retrospective"),
        "commit": ("Developer", "development"),
        "push": ("DevOps", "release"),
        "pr": ("Developer", "code_review"),
        "deploy": ("DevOps", "release"),
    }

    def __init__(
        self,
        cadence_config: Optional[Dict[str, Any]],
        sprint_manager: "SprintManager",
    ) -> None:
        self._raw_config = cadence_config or {}
        self._cadence_map = self._normalize_config(self._raw_config)
        self.sprint_manager = sprint_manager
        self.last_context: Optional[CadenceContext] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_step(self, step_name: str, step_config: Dict[str, Any]) -> CadenceDecision:
        """Return the cadence decision for a loop step."""

        role, activity = self._resolve_role_activity(step_name, step_config)

        cadence_override = step_config.get("cadence")
        cadence_source = "step" if cadence_override is not None else "default"
        cadence_value = cadence_override

        if cadence_value is None:
            cadence_value = self._lookup_role_cadence(role, activity)
            if cadence_value is not None:
                cadence_source = "role"

        context = self._build_context(step_name, role, activity, cadence_value, cadence_source)
        self.last_context = context

        should_execute = should_run_step(role, activity, context)
        decision = context.decision

        if decision is None:
            decision = self.sprint_manager.should_run_step(step_name, cadence_value)
            context.decision = decision

        if not should_execute:
            return decision

        return decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_role_activity(
        self, step_name: str, step_config: Dict[str, Any]
    ) -> Tuple[str, str]:
        role = step_config.get("role")
        activity = step_config.get("activity")

        if role and activity:
            return str(role), str(activity)

        default_role, default_activity = self._DEFAULT_STEP_METADATA.get(
            step_name.lower(), ("Developer", step_name)
        )

        return str(role or default_role), str(activity or default_activity)

    def _lookup_role_cadence(self, role: str, activity: str) -> Optional[object]:
        role_key = self._normalize_key(role)
        activity_key = self._normalize_key(activity)
        return self._cadence_map.get(role_key, {}).get(activity_key)

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Dict[str, object]]:
        normalized: Dict[str, Dict[str, object]] = {}
        for role, activities in (config or {}).items():
            if not isinstance(activities, dict):
                continue
            role_key = self._normalize_key(role)
            if not role_key:
                continue
            normalized.setdefault(role_key, {})
            for activity, value in activities.items():
                activity_key = self._normalize_key(activity)
                if not activity_key:
                    continue
                normalized[role_key][activity_key] = value
        return normalized

    def _build_context(
        self,
        step_name: str,
        role: str,
        activity: str,
        cadence_value: Optional[object],
        cadence_source: str,
    ) -> CadenceContext:
        available_events: Dict[str, int] = {}
        for event in ("feature", "bug", "epic"):
            pending = self.sprint_manager.pending_events.get(event, 0)
            consumed = self.sprint_manager.event_consumption[step_name][event]
            remaining = pending - consumed
            available_events[event] = remaining if remaining > 0 else 0

        per_sprint_consumed = self.sprint_manager.event_consumption[step_name]["sprint"]
        run_count = self.sprint_manager.step_run_counts.get(step_name, 0)

        return CadenceContext(
            step_name=step_name,
            role=role,
            activity=activity,
            cadence_value=cadence_value,
            cadence_source=cadence_source,
            sprint_manager=self.sprint_manager,
            sprint_day=self.sprint_manager.current_day,
            sprint_length=self.sprint_manager.sprint_length_days,
            run_count=run_count,
            per_sprint_consumed=per_sprint_consumed,
            available_events=available_events,
        )

    @staticmethod
    def _normalize_key(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()


__all__ = ["CadenceContext", "CadenceManager", "should_run_step"]
