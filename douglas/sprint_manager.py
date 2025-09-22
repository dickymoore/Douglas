"""Utility helpers for tracking sprint cadence and release policies."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CadenceDecision:
    """Represents the decision for whether a step should execute."""

    should_run: bool
    reason: str
    event_type: Optional[str] = None


def _normalize_cadence(cadence: Optional[object]) -> Optional[str]:
    """Extract the frequency token from a cadence configuration value."""

    if cadence is None:
        return None
    if isinstance(cadence, str):
        return cadence.strip() or None
    if isinstance(cadence, dict):
        for key in ("frequency", "cadence", "type"):
            value = cadence.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def should_run_step(
    step_name: str,
    sprint_day: int,
    sprint_length: int,
    cadence: Optional[object],
    *,
    available_events: Optional[Dict[str, int]] = None,
    run_count: int = 0,
    per_sprint_consumed: int = 0,
) -> CadenceDecision:
    """Determine whether a step should run under the provided cadence rules."""

    frequency = _normalize_cadence(cadence)
    if not frequency:
        return CadenceDecision(True, "Cadence defaults to daily execution.")

    frequency = frequency.lower()
    available_events = available_events or {}
    sprint_day = max(sprint_day, 1)
    sprint_length = max(sprint_length, 0)

    if frequency in {"daily", "every_day", "each_iteration"}:
        return CadenceDecision(True, "Daily cadence executes every iteration.")

    if frequency == "once":
        if run_count == 0:
            return CadenceDecision(True, "Once cadence has not yet executed this sprint.")
        return CadenceDecision(False, "Once cadence already satisfied for this sprint.")

    if frequency in {"on_demand", "on-demand"}:
        return CadenceDecision(False, "On-demand cadence requires manual trigger.")

    if frequency == "per_sprint":
        if sprint_length <= 0:
            return CadenceDecision(True, "Sprint length unspecified; executing per-sprint step.", "sprint")
        if sprint_day == sprint_length and per_sprint_consumed == 0:
            return CadenceDecision(
                True,
                f"Final sprint day reached ({sprint_day}/{sprint_length}).",
                "sprint",
            )
        if sprint_day != sprint_length:
            return CadenceDecision(
                False,
                f"Per-sprint cadence waits for day {sprint_length}; currently day {sprint_day}.",
            )
        return CadenceDecision(False, "Per-sprint cadence already completed for this sprint.")

    event_frequency_map = {
        "per_feature": "feature",
        "per_bug": "bug",
        "per_epic": "epic",
    }

    if frequency in event_frequency_map:
        event_key = event_frequency_map[frequency]
        pending = max(available_events.get(event_key, 0), 0)
        if pending > 0:
            plural = "s" if pending != 1 else ""
            return CadenceDecision(
                True,
                f"{pending} {event_key}{plural} awaiting follow-up.",
                event_key,
            )
        return CadenceDecision(False, f"No pending {event_key} work queued for this cadence.")

    return CadenceDecision(True, f"Unrecognized cadence '{frequency}'; defaulting to execution.")


class SprintManager:
    """Tracks sprint progress and cadence-driven events."""

    _COMMIT_PATTERN = re.compile(r"^(?P<type>[a-zA-Z]+)(?P<breaking>!)?(?:\((?P<scope>[^)]+)\))?:")

    def __init__(self, sprint_length_days: Optional[int] = None) -> None:
        if sprint_length_days is None:
            sprint_length_days = 10
        if sprint_length_days <= 0:
            sprint_length_days = 1

        self.sprint_length_days = sprint_length_days
        self.current_day = 1
        self.current_iteration = 1
        self.sprint_index = 1

        self.pending_events: Dict[str, int] = {"feature": 0, "bug": 0, "epic": 0}
        self.event_consumption: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.step_run_counts: Dict[str, int] = defaultdict(int)

        self.completed_features: set[str] = set()
        self.completed_bugs: set[str] = set()
        self.completed_epics: set[str] = set()

        self.push_executed_this_sprint = False
        self.pr_executed_this_sprint = False

        self.commits_since_last_push = 0
        self.commits_since_last_pr = 0

    # ------------------------------------------------------------------
    # Sprint progression
    # ------------------------------------------------------------------
    def is_final_day(self) -> bool:
        return self.current_day >= self.sprint_length_days

    def finish_iteration(self) -> None:
        self.current_iteration += 1
        self.current_day += 1
        if self.current_day > self.sprint_length_days:
            self.start_new_sprint()

    def start_new_sprint(self) -> None:
        self.sprint_index += 1
        self.current_day = 1
        self.current_iteration = 1

        self.pending_events = {"feature": 0, "bug": 0, "epic": 0}
        self.event_consumption = defaultdict(lambda: defaultdict(int))
        self.step_run_counts = defaultdict(int)

        self.completed_features.clear()
        self.completed_bugs.clear()
        self.completed_epics.clear()

        self.push_executed_this_sprint = False
        self.pr_executed_this_sprint = False
        self.commits_since_last_push = 0
        self.commits_since_last_pr = 0

    # ------------------------------------------------------------------
    # Cadence evaluation
    # ------------------------------------------------------------------
    def should_run_step(self, step_name: str, cadence: Optional[object]) -> CadenceDecision:
        available = {
            "feature": self._available_event_for_consumer("feature", step_name),
            "bug": self._available_event_for_consumer("bug", step_name),
            "epic": self._available_event_for_consumer("epic", step_name),
        }
        per_sprint_consumed = self.event_consumption[step_name]["sprint"]
        run_count = self.step_run_counts[step_name]
        return should_run_step(
            step_name,
            self.current_day,
            self.sprint_length_days,
            cadence,
            available_events=available,
            run_count=run_count,
            per_sprint_consumed=per_sprint_consumed,
        )

    def record_step_execution(self, step_name: str, event_type: Optional[str]) -> None:
        self.step_run_counts[step_name] += 1
        if event_type:
            self.event_consumption[step_name][event_type] += 1

    def has_step_run(self, step_name: str) -> bool:
        return self.step_run_counts.get(step_name, 0) > 0

    # ------------------------------------------------------------------
    # Event and commit tracking
    # ------------------------------------------------------------------
    def mark_feature_completed(self, identifier: Optional[str] = None) -> None:
        self.pending_events["feature"] += 1
        if identifier:
            self.completed_features.add(str(identifier))

    def mark_bug_completed(self, identifier: Optional[str] = None) -> None:
        self.pending_events["bug"] += 1
        if identifier:
            self.completed_bugs.add(str(identifier))

    def mark_epic_completed(self, identifier: Optional[str] = None) -> None:
        self.pending_events["epic"] += 1
        if identifier:
            self.completed_epics.add(str(identifier))

    def record_commit(self, message: Optional[str]) -> None:
        if not message:
            return

        self.commits_since_last_push += 1
        self.commits_since_last_pr += 1

        match = self._COMMIT_PATTERN.match(message.strip())
        if not match:
            return

        commit_type = (match.group("type") or "").lower()
        scope = (match.group("scope") or "").strip()

        if commit_type == "feat":
            self.mark_feature_completed(scope or None)
        elif commit_type == "fix":
            self.mark_bug_completed(scope or None)
        elif commit_type == "epic":
            self.mark_epic_completed(scope or None)
        else:
            if scope and "epic" in scope.lower():
                self.mark_epic_completed(scope)

    # ------------------------------------------------------------------
    # Push / PR policy coordination
    # ------------------------------------------------------------------
    def should_run_push(self, push_policy: str) -> CadenceDecision:
        policy = (push_policy or "per_feature").lower()

        if policy == "per_feature":
            available = self._available_event_for_consumer("feature", "push")
            if available > 0:
                plural = "s" if available != 1 else ""
                return CadenceDecision(True, f"{available} feature{plural} ready to push.", "feature")
            return CadenceDecision(False, "Push policy per_feature: no completed features pending.")

        if policy == "per_bug":
            available = self._available_event_for_consumer("bug", "push")
            if available > 0:
                plural = "s" if available != 1 else ""
                return CadenceDecision(True, f"{available} bug fix{plural} ready to push.", "bug")
            return CadenceDecision(False, "Push policy per_bug: no resolved bugs pending push.")

        if policy == "per_epic":
            available = self._available_event_for_consumer("epic", "push")
            if available > 0:
                plural = "s" if available != 1 else ""
                return CadenceDecision(True, f"{available} epic{plural} ready for integration.", "epic")
            return CadenceDecision(False, "Push policy per_epic: no completed epics pending push.")

        if policy == "per_sprint":
            if not self.is_final_day():
                return CadenceDecision(
                    False,
                    f"Push policy per_sprint waits for day {self.sprint_length_days}; current day {self.current_day}.",
                )
            if self.push_executed_this_sprint:
                return CadenceDecision(False, "Push already executed for this sprint.")
            if self.commits_since_last_push <= 0:
                return CadenceDecision(False, "No new commits accumulated for sprint push.")
            return CadenceDecision(True, "Final sprint day; pushing accumulated commits.", "sprint")

        return CadenceDecision(True, f"Unknown push policy '{policy}'; defaulting to push.")

    def record_push(self, event_type: Optional[str], push_policy: str) -> None:
        self.record_step_execution("push", event_type)
        policy = (push_policy or "per_feature").lower()

        if policy == "per_sprint":
            self.push_executed_this_sprint = True
            self.commits_since_last_push = 0
        else:
            if self.commits_since_last_push > 0:
                self.commits_since_last_push = max(0, self.commits_since_last_push - 1)

    def should_open_pr(self, push_policy: str) -> CadenceDecision:
        policy = (push_policy or "per_feature").lower()

        if policy == "per_feature":
            available = self._available_event_for_consumer("feature", "pr")
            if available > 0:
                plural = "s" if available != 1 else ""
                return CadenceDecision(True, f"{available} feature{plural} ready for PR.", "feature")
            return CadenceDecision(False, "PR policy per_feature: no completed features pending.")

        if policy == "per_bug":
            available = self._available_event_for_consumer("bug", "pr")
            if available > 0:
                plural = "s" if available != 1 else ""
                return CadenceDecision(True, f"{available} bug fix{plural} ready for PR.", "bug")
            return CadenceDecision(False, "PR policy per_bug: no resolved bugs pending PR.")

        if policy == "per_epic":
            available = self._available_event_for_consumer("epic", "pr")
            if available > 0:
                plural = "s" if available != 1 else ""
                return CadenceDecision(True, f"{available} epic{plural} ready for showcase.", "epic")
            return CadenceDecision(False, "PR policy per_epic: no completed epics pending PR.")

        if policy == "per_sprint":
            if not self.is_final_day():
                return CadenceDecision(
                    False,
                    f"PR policy per_sprint waits for day {self.sprint_length_days}; current day {self.current_day}.",
                )
            if self.pr_executed_this_sprint:
                return CadenceDecision(False, "PR already created for this sprint.")
            if self.commits_since_last_pr <= 0:
                return CadenceDecision(False, "No new commits accumulated for sprint PR.")
            return CadenceDecision(True, "Final sprint day; preparing sprint PR.", "sprint")

        return CadenceDecision(True, f"Unknown PR policy '{policy}'; defaulting to create PR.")

    def record_pr(self, event_type: Optional[str], push_policy: str) -> None:
        self.record_step_execution("pr", event_type)
        policy = (push_policy or "per_feature").lower()

        if policy == "per_sprint":
            self.pr_executed_this_sprint = True
            self.commits_since_last_pr = 0
        else:
            if self.commits_since_last_pr > 0:
                self.commits_since_last_pr = max(0, self.commits_since_last_pr - 1)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _available_event_for_consumer(self, event_type: str, consumer: str) -> int:
        pending = self.pending_events.get(event_type, 0)
        consumed = self.event_consumption[consumer][event_type]
        remaining = pending - consumed
        return remaining if remaining > 0 else 0

    def describe_day(self) -> str:
        return f"day {self.current_day} of sprint {self.sprint_index}"
