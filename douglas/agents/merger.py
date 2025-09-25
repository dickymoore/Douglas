"""Merger agent that consolidates workspace outputs into the main repo."""

from __future__ import annotations

import shutil
from pathlib import Path

from douglas.agents.executor import ParallelAgentExecutor
from douglas.logging import get_logger, log_action


logger = get_logger(__name__)


class MergerAgent:
    """Apply agent workspace changes back into the repository."""

    def __init__(
        self,
        executor: ParallelAgentExecutor,
        *,
        target_root: Path | str | None = None,
    ) -> None:
        self.executor = executor
        self.target_root = Path(target_root or Path.cwd())
        self.target_root.mkdir(parents=True, exist_ok=True)

    @log_action("merge-workspace", logger_factory=lambda: logger)
    def merge(self, agent_id: str, *, clean: bool = False) -> list[Path]:
        """Merge the ``changes`` directory from an agent workspace into the repo."""

        workspace = self.executor.get_workspace(agent_id)
        changes_dir = workspace.changes_dir
        if not changes_dir.exists():
            logger.warning("No changes directory found for agent %s", agent_id)
            return []

        applied_files: list[Path] = []
        for path in sorted(changes_dir.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(changes_dir)
            destination = self.target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self.executor.lock_manager.acquire([destination]):
                shutil.copy2(path, destination)
            applied_files.append(destination)
            logger.debug(
                "Merged %s from agent %s",
                relative,
                agent_id,
                extra={"metadata": {"agent": agent_id, "file": str(relative)}},
            )

        if clean:
            shutil.rmtree(changes_dir)
            changes_dir.mkdir()

        return applied_files


__all__ = ["MergerAgent"]
