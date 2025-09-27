"""String manipulation helpers used across the project."""

from __future__ import annotations

import hashlib
import re
from typing import Optional

__all__ = ["slugify_token"]


_INVALID_CHARS_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify_token(
    value: str,
    *,
    max_length: Optional[int] = None,
    separator: str = "_",
    fallback_hash_len: Optional[int] = None,
) -> str:
    """Return a deterministic, lowercase slug for the given value.

    Args:
        value: Source text to slugify.
        max_length: Optional maximum length for the slug.
        separator: Replacement for invalid character runs. Use ``""`` to remove
            invalid characters altogether.
        fallback_hash_len: Number of characters to take from the fallback hash
            when no slug characters remain. Defaults to ``max_length`` when set,
            otherwise 10.
    """

    text = str(value or "")
    normalized = text.lower()

    if separator:
        collapsed = _INVALID_CHARS_PATTERN.sub(separator, normalized)
        collapsed = re.sub(f"{re.escape(separator)}+", separator, collapsed)
        collapsed = collapsed.strip(separator)
    else:
        collapsed = _INVALID_CHARS_PATTERN.sub("", normalized)

    if collapsed and max_length is not None:
        collapsed = collapsed[:max_length]
        if separator:
            collapsed = collapsed.strip(separator)

    if collapsed:
        return collapsed

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    length = max_length if max_length is not None else (fallback_hash_len or 10)
    return digest[:length]
