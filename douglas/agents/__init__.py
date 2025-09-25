"""Parallel agent execution helpers for Douglas."""

from __future__ import annotations

from douglas.agents.executor import AgentExecutionResult, ParallelAgentExecutor
from douglas.agents.locks import FileLockManager
from douglas.agents.merger import MergerAgent
from douglas.agents.workspace import AgentWorkspace

__all__ = [
    "AgentExecutionResult",
    "AgentWorkspace",
    "FileLockManager",
    "MergerAgent",
    "ParallelAgentExecutor",
]
