"""Utilities for transforming planning commitments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from douglas.logging_utils import get_logger

__all__ = ["Commitment", "filter_commitments"]


logger = get_logger(__name__)


_SKIPPED_STATUSES = {
    "done",
    "completed",
    "complete",
    "cancelled",
    "canceled",
    "duplicate",
    "won't do",
    "wont_do",
    "won't_do",
    "wont do",
    "obsolete",
    "archived",
}


@dataclass(frozen=True)
class Commitment:
    """Serializable representation of a planning commitment entry."""

    identifier: str
    title: str
    status: str = ""
    description: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "Commitment":
        """Create a commitment from a mapping, validating required fields."""

        title = _coerce_string(payload.get("title"))
        if not title:
            title = _coerce_string(payload.get("name"))
        if not title:
            raise ValueError("commitment requires a title")

        identifier = _coerce_string(
            payload.get("id")
            or payload.get("commitment_id")
            or payload.get("ref")
            or payload.get("reference")
        )

        status = _coerce_string(payload.get("status"))
        description_value = payload.get("description")
        description = _normalize_multiline(description_value)

        return cls(identifier=identifier, title=title, status=status, description=description)


def _filter_commitments(items: Sequence[Mapping[str, object]]) -> List[Commitment]:
    """Return valid commitments, logging when items are discarded."""

    commitments: List[Commitment] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            logger.warning("Skipping non-mapping commitment at index %d", index)
            continue
        normalized: Dict[str, object] = dict(item)
        status = _coerce_string(normalized.get("status")).lower()
        if status in _SKIPPED_STATUSES:
            continue
        try:
            if "title" not in normalized and "name" in normalized:
                normalized["title"] = normalized["name"]
            commitment = Commitment.from_mapping(normalized)
        except ValueError as exc:
            summary = _summarize_commitment(normalized)
            logger.warning(
                "Skipping invalid commitment at index %d (%s): %s",
                index,
                summary,
                exc,
            )
            continue
        commitments.append(commitment)
    return commitments


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

