"""Sprint planning domain objects used by the planning step."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _normalize_str(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


@dataclass(frozen=True)
class Commitment:
    """Represents a backlog item committed to a sprint."""

    id: str
    title: str
    status: str = "todo"
    owner: Optional[str] = None
    estimate: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "Commitment":
        identifier = _normalize_str(data.get("id"))
        if not identifier:
            raise ValueError("Commitment requires an 'id' field.")
        title = _normalize_str(data.get("title")) or identifier
        status = _normalize_str(data.get("status")) or "todo"
        owner = _normalize_str(data.get("owner"))
        estimate_value = data.get("estimate")
        parsed_estimate: Optional[float]
        if isinstance(estimate_value, (int, float)):
            parsed_estimate = float(estimate_value)
        else:
            parsed_estimate = None
        return cls(id=identifier, title=title, status=status, owner=owner, estimate=parsed_estimate)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "id": self.id,
            "title": self.title,
            "status": self.status,
        }
        if self.owner is not None:
            payload["owner"] = self.owner
        if self.estimate is not None:
            payload["estimate"] = self.estimate
        return payload


@dataclass
class SprintPlan:
    """Structured representation of a sprint planning decision."""

    sprint_index: int
    commitments: List[Commitment] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    items_requested: int = 0
    backlog_items_total: int = 0
    selection_seed: Optional[int] = None
    backlog_signature: Optional[str] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def signature_for_items(items: Sequence[Mapping[str, object]]) -> str:
        """Return a deterministic SHA-256 signature for backlog items.

        The signature is used to detect when the underlying backlog changes
        without relying on object identity or ordering. Only the normalised
        identifier, title, status, and owner fields participate in the hash to
        keep the digest stable across unrelated metadata updates. The return
        value is a hexadecimal digest suitable for persistence and comparisons.
        """
        canonical: List[Dict[str, str]] = []
        for item in items:
            if isinstance(item, Mapping):
                canonical.append(
                    {
                        "id": _normalize_str(item.get("id")) or "",
                        "title": _normalize_str(item.get("title")) or "",
                        "status": _normalize_str(item.get("status")) or "",
                        "owner": _normalize_str(item.get("owner")) or "",
                    }
                )
            else:
                canonical.append({"id": "", "title": "", "status": "", "owner": ""})
        raw = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def commitment_ids(self) -> List[str]:
        return [commitment.id for commitment in self.commitments]

    def to_dict(self) -> Dict[str, object]:
        return {
            "sprint": self.sprint_index,
            "generated_at": self.generated_at.isoformat(),
            "selection_seed": self.selection_seed,
            "items_requested": self.items_requested,
            "backlog": {
                "total_items": self.backlog_items_total,
                "signature": self.backlog_signature,
            },
            "commitments": [commitment.to_dict() for commitment in self.commitments],
            "goals": list(self.goals),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SprintPlan":
        sprint_raw = data.get("sprint", 0)
        try:
            sprint_index = int(sprint_raw or 0)
        except (ValueError, TypeError):
            sprint_index = 0
        commitments_raw = data.get("commitments") or []
        commitments: List[Commitment] = []
        if isinstance(commitments_raw, Sequence):
            for item in commitments_raw:
                if isinstance(item, Mapping):
                    commitments.append(Commitment.from_mapping(item))
        goals_raw = data.get("goals") or []
        goals = [str(goal) for goal in goals_raw if goal is not None]
        items_requested = int(data.get("items_requested", 0) or 0)
        backlog_info = data.get("backlog")
        backlog_items_total = 0
        backlog_signature = None
        if isinstance(backlog_info, Mapping):
            backlog_items_total = int(backlog_info.get("total_items", 0) or 0)
            backlog_signature = _normalize_str(backlog_info.get("signature"))
        seed_value = data.get("selection_seed")
        selection_seed = int(seed_value) if isinstance(seed_value, (int, float)) else None
        generated_raw = _normalize_str(data.get("generated_at"))
        if generated_raw:
            try:
                generated_at = datetime.fromisoformat(generated_raw)
            except ValueError:
                generated_at = datetime.now(timezone.utc)
        else:
            generated_at = datetime.now(timezone.utc)
        return cls(
            sprint_index=sprint_index,
            commitments=commitments,
            goals=goals,
            items_requested=items_requested,
            backlog_items_total=backlog_items_total,
            selection_seed=selection_seed,
            backlog_signature=backlog_signature,
            generated_at=generated_at,
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent) + "\n"

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def to_markdown(self, summary: Optional[str] = None) -> str:
        lines: List[str] = []
        lines.append(f"# Sprint {self.sprint_index} Plan")
        lines.append("")
        meta_details = [
            f"- Generated: {self.generated_at.isoformat()}",
            f"- Commitments selected: {len(self.commitments)} / {self.items_requested}",
            f"- Backlog items available: {self.backlog_items_total}",
        ]
        if self.selection_seed is not None:
            meta_details.append(f"- Selection seed: {self.selection_seed}")
        if self.backlog_signature:
            meta_details.append(f"- Backlog signature: `{self.backlog_signature}`")
        lines.extend(meta_details)
        lines.append("")
        if summary:
            summary = summary.strip()
            if summary:
                lines.append("## Summary")
                lines.append(summary)
                lines.append("")
        lines.append("## Sprint Goals")
        if self.goals:
            for goal in self.goals:
                lines.append(f"- {goal}")
        else:
            lines.append("_No sprint goals recorded._")
        lines.append("")
        lines.append("## Commitments")
        if self.commitments:
            lines.append("| ID | Title | Status | Owner |")
            lines.append("| --- | --- | --- | --- |")
            for commitment in self.commitments:
                owner = commitment.owner or "Unassigned"
                status = commitment.status or "todo"
                lines.append(
                    f"| {commitment.id} | {commitment.title} | {status} | {owner} |"
                )
        else:
            lines.append("_No commitments selected for this sprint._")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write_markdown(self, path: Path, summary: Optional[str] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(summary), encoding="utf-8")


__all__ = ["Commitment", "SprintPlan"]
