"""Parallel agent execution primitives."""

from __future__ import annotations

import concurrent.futures
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from douglas.agents.locks import FileLockManager
from douglas.agents.workspace import AgentCommandResult, AgentWorkspace
from douglas.logging import get_logger


logger = get_logger(__name__)


@dataclass
class AgentExecutionResult:
    """Result produced by a background agent execution."""

    agent_id: str
    workspace: AgentWorkspace
    command_result: AgentCommandResult
    metadata: Mapping[str, str] | None = None


class ParallelAgentExecutor:
    """Coordinate parallel execution of CLI agents."""

    def __init__(
        self,
        *,
        workspace_root: Path | str | None = None,
        lock_manager: FileLockManager | None = None,
        max_workers: Optional[int] = None,
    ) -> None:
        base = Path(workspace_root or ".douglas/workspaces")
        base.mkdir(parents=True, exist_ok=True)
        self.workspace_root = base
        self.lock_manager = lock_manager or FileLockManager(base.parent / "locks")
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._workspaces: dict[str, AgentWorkspace] = {}

    def _workspace_for(self, agent_id: str) -> AgentWorkspace:
        if agent_id not in self._workspaces:
            workspace = AgentWorkspace(agent_id, self.workspace_root / agent_id, self.lock_manager)
            self._workspaces[agent_id] = workspace
        return self._workspaces[agent_id]

    def submit(
        self,
        command: Iterable[str],
        *,
        agent_id: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> concurrent.futures.Future[AgentExecutionResult]:
        """Schedule an agent command for execution."""

        resolved_agent_id = agent_id or uuid.uuid4().hex
        workspace = self._workspace_for(resolved_agent_id)
        logger.debug(
            "Scheduling agent %s command %s",
            resolved_agent_id,
            list(command),
            extra={"metadata": {"agent": resolved_agent_id}},
        )

        def runner() -> AgentExecutionResult:
            command_result = workspace.run_command(command, env=env, timeout=timeout)
            return AgentExecutionResult(
                agent_id=resolved_agent_id,
                workspace=workspace,
                command_result=command_result,
                metadata=metadata,
            )

        return self._executor.submit(runner)

    def shutdown(self, wait: bool = True) -> None:
        logger.debug("Shutting down ParallelAgentExecutor")
        self._executor.shutdown(wait=wait)

    def get_workspace(self, agent_id: str) -> AgentWorkspace:
        """Return the workspace for a particular agent."""

        return self._workspace_for(agent_id)


__all__ = ["AgentExecutionResult", "ParallelAgentExecutor"]
