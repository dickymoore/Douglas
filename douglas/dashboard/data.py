"""Utilities for loading Douglas run-state into dashboard-friendly structures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

_STATUS_BUCKETS = [
    "not_started",
    "in_progress",
    "blocked",
    "in_test",
    "finished",
]


@dataclass
class WorkSummary:
    """Aggregated counts of items by status."""

    counts: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class BurndownPoint:
    """Point in a burn-up/down chart."""

    date: datetime
    remaining: int
    completed: int


@dataclass
class DashboardData:
    """Container of data exposed via the dashboard API."""

    summary: WorkSummary
    burndown: list[BurndownPoint]
    inbox: dict[str, int]
    cumulative_flow: dict[str, list[tuple[str, int]]]


def _safe_load(path: Path) -> Mapping[str, Any] | list[Mapping[str, Any]] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


def _count_statuses(entries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    buckets = {status: 0 for status in _STATUS_BUCKETS}
    for entry in entries:
        status = str(entry.get("status", "not_started")).strip().lower()
        buckets.setdefault(status, 0)
        buckets[status] += 1
    return buckets


def load_work_summary(base_path: Path) -> WorkSummary:
    summary: dict[str, dict[str, int]] = {}
    for category in ("epics", "features", "work-items", "stories"):
        directory = base_path / category
        if not directory.exists():
            continue
        entries: list[Mapping[str, Any]] = []
        for pattern in ("*.yml", "*.yaml", "*.json"):
            for file_path in directory.rglob(pattern):
                payload = _safe_load(file_path)
                if isinstance(payload, Mapping):
                    entries.append(payload)
        summary[category] = _count_statuses(entries)
    return WorkSummary(counts=summary)


def _resolve_state_root(base_path: Path) -> Path:
    direct_state = base_path / "state"
    if direct_state.exists():
        return base_path
    nested = base_path / ".douglas"
    if nested.exists():
        return nested
    return base_path


def load_burndown(base_path: Path) -> list[BurndownPoint]:
    state_root = _resolve_state_root(base_path)
    history_path = state_root / "state" / "sprint_history.json"
    if not history_path.exists():
        history_path = state_root / "state" / "burndown.json"
    payload = _safe_load(history_path)
    points: list[BurndownPoint] = []
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            date_str = str(entry.get("date", "")) or str(entry.get("timestamp", ""))
            try:
                date = datetime.fromisoformat(date_str)
            except ValueError:
                continue
            remaining = int(entry.get("remaining", entry.get("todo", 0)))
            completed = int(entry.get("completed", entry.get("done", 0)))
            points.append(
                BurndownPoint(date=date, remaining=remaining, completed=completed)
            )
    return points


def load_inbox_counts(base_path: Path) -> dict[str, int]:
    inbox_dir = (_resolve_state_root(base_path) / "inbox").resolve()
    unanswered = 0
    answered = 0
    if not inbox_dir.exists():
        inbox_dir = base_path / "inbox"
    if not inbox_dir.exists():
        return {"unanswered": 0, "answered": 0}
    for pattern in ("*.yaml", "*.yml", "*.json"):
        for path in inbox_dir.glob(pattern):
            payload = _safe_load(path)
            if isinstance(payload, Mapping):
                if payload.get("answer"):
                    answered += 1
                else:
                    unanswered += 1
    return {"unanswered": unanswered, "answered": answered}


def load_cumulative_flow(base_path: Path) -> dict[str, list[tuple[str, int]]]:
    state_root = _resolve_state_root(base_path)
    flow_path = state_root / "state" / "cumulative_flow.json"
    payload = _safe_load(flow_path)
    if not isinstance(payload, Mapping):
        return {}
    flow: dict[str, list[tuple[str, int]]] = {}
    for status, datapoints in payload.items():
        if not isinstance(datapoints, list):
            continue
        entries: list[tuple[str, int]] = []
        for entry in datapoints:
            if not isinstance(entry, Mapping):
                continue
            timestamp = str(entry.get("date", ""))
            count = int(entry.get("count", 0))
            entries.append([timestamp, count])
        flow[status] = entries
    return flow


def load_dashboard_data(base_path: Path | str) -> DashboardData:
    base = Path(base_path)
    summary = load_work_summary(base)
    burndown = load_burndown(base)
    inbox = load_inbox_counts(base)
    cumulative_flow = load_cumulative_flow(base)
    return DashboardData(
        summary=summary,
        burndown=burndown,
        inbox=inbox,
        cumulative_flow=cumulative_flow,
    )


__all__ = [
    "BurndownPoint",
    "DashboardData",
    "WorkSummary",
    "load_dashboard_data",
    "load_work_summary",
    "load_burndown",
    "load_inbox_counts",
    "load_cumulative_flow",
]
