"""Helpers for working with conventional commit messages."""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["format_conventional_commit"]


_SCOPE_PATTERN = re.compile(r"[^a-z0-9]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _sanitize_type(commit_type: str) -> str:
    normalized = str(commit_type).strip().lower()
    sanitized = _SCOPE_PATTERN.sub("", normalized)
    if not sanitized:
        raise ValueError("commit type must contain alphabetic characters")
    return sanitized


def _sanitize_scope(scope: Optional[str]) -> Optional[str]:
    if scope is None:
        return None
    normalized = str(scope or "").strip().lower()
    if not normalized:
        return None
    sanitized = _SCOPE_PATTERN.sub("-", normalized)
    sanitized = sanitized.strip("-")
    return sanitized or None


def _sanitize_description(description: str) -> str:
    collapsed = _WHITESPACE_PATTERN.sub(" ", (description or "").strip())
    if not collapsed:
        raise ValueError("commit description must not be empty")
    if collapsed.endswith((".", "!")):
        collapsed = collapsed[:-1]
    if collapsed and collapsed[0].isalpha():
        collapsed = collapsed[0].lower() + collapsed[1:]
    return collapsed


def format_conventional_commit(
    commit_type: str, description: str, scope: Optional[str] = None
) -> str:
    """Return a formatted conventional commit subject line."""

    normalized_type = _sanitize_type(commit_type)
    normalized_scope = _sanitize_scope(scope)
    normalized_description = _sanitize_description(description)

    if normalized_scope:
        return f"{normalized_type}({normalized_scope}): {normalized_description}"
    return f"{normalized_type}: {normalized_description}"
