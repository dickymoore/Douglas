from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

__all__ = ["RoleDocuments", "collect_role_documents"]


@dataclass
class RoleDocuments:
    """Container for a role's summary and handoff notes."""

    role: str
    directory: Path
    summary_path: Path
    summary_text: str
    handoffs_path: Optional[Path]
    handoffs_text: str

    def normalized_role(self) -> str:
        return _normalize_role_key(self.role)


def collect_role_documents(project_root: Path, sprint_folder: str) -> List[RoleDocuments]:
    """Load summaries and handoffs for each role in the sprint."""

    roles_root = Path(project_root).resolve() / "ai-inbox" / "sprints" / sprint_folder / "roles"
    if not roles_root.is_dir():
        return []

    documents: List[RoleDocuments] = []
    for entry in sorted(roles_root.iterdir()):
        if not entry.is_dir():
            continue

        summary_path = entry / "summary.md"
        handoffs_path = entry / "handoffs.md"
        if not handoffs_path.exists():
            alt_path = entry / "handoff.md"
            if alt_path.exists():
                handoffs_path = alt_path

        summary_text = _read_text(summary_path)
        handoffs_text = _read_text(handoffs_path) if handoffs_path.exists() else ""

        documents.append(
            RoleDocuments(
                role=entry.name,
                directory=entry,
                summary_path=summary_path,
                summary_text=summary_text,
                handoffs_path=handoffs_path if handoffs_path.exists() else None,
                handoffs_text=handoffs_text,
            )
        )

    return documents


def _normalize_role_key(role: Optional[str]) -> str:
    if role is None:
        return ""
    return str(role).strip().lower().replace(" ", "_")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
