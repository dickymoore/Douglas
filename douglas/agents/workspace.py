"""Agent workspace helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from douglas.agents.locks import FileLockManager
from douglas.logging import get_logger


logger = get_logger(__name__)


@dataclass
class AgentCommandResult:
    """Represents the result of a command executed inside an agent workspace."""

    returncode: int
    stdout: str
    stderr: str
    duration: float
    command: list[str]


@dataclass
class AgentWorkspace:
    """Isolated filesystem workspace for an agent."""

    agent_id: str
    root: Path
    lock_manager: FileLockManager
    metadata_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "changes").mkdir(exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        if not self.metadata_path.exists():
            self.metadata_path.write_text(json.dumps({"agent_id": self.agent_id}, indent=2))

    @property
    def changes_dir(self) -> Path:
        return self.root / "changes"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def run_command(
        self,
        command: Iterable[str],
        *,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> AgentCommandResult:
        """Execute a command within the workspace."""

        command_list = list(command)
        environment = os.environ.copy()
        if env:
            environment.update(env)
        start = time.perf_counter()
        process = subprocess.run(
            command_list,
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = time.perf_counter() - start
        logger.debug(
            "Agent %s executed %s in %.2fs (rc=%s)",
            self.agent_id,
            command_list,
            duration,
            process.returncode,
            extra={"metadata": {"agent": self.agent_id, "duration": duration}},
        )
        return AgentCommandResult(
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            duration=duration,
            command=command_list,
        )

    def record_change(self, relative_path: Path | str, content: str) -> None:
        target = (self.changes_dir / Path(relative_path)).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def copy_into_changes(self, source: Path, relative_dest: Path | str | None = None) -> None:
        destination = self.changes_dir / (relative_dest or source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    def lock_files(self, *paths: Path | str, timeout: Optional[float] = None):
        absolute_paths = [self.changes_dir / Path(p) for p in paths]
        return self.lock_manager.acquire(absolute_paths, timeout=timeout)


__all__ = ["AgentWorkspace", "AgentCommandResult"]
