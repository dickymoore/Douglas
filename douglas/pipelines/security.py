"""Security tooling pipeline for Douglas."""

from __future__ import annotations

import shutil
import subprocess
from typing import Iterable, Optional, Sequence

# Default security commands prefer widely adopted tooling. Each entry is a tuple
# of the executable name and the arguments to invoke. The pipeline will execute
# every tool that is available on the current PATH, warning about any that are
# skipped and failing when no security tooling can be run at all.
_DEFAULT_SECURITY_COMMANDS: tuple[tuple[str, Sequence[str]], ...] = (
    ("bandit", ("bandit", "-q", "-r", ".")),
    ("semgrep", ("semgrep", "--config", "auto")),
)


def _run_command(command: Sequence[str]) -> None:
    """Execute a security command, surfacing actionable diagnostics on error."""

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(f"Required security tool '{command[0]}' is not installed or not on PATH.")
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        cmd_display = " ".join(command)
        exit_code = exc.returncode or 1
        print(
            f"Security command '{cmd_display}' failed with exit code {exit_code}."
        )
        raise SystemExit(exit_code)


def run_security(
    additional_commands: Optional[Iterable[Sequence[str]]] = None,
) -> None:
    """Run available security tooling and propagate failures as ``SystemExit``."""

    commands_to_run: list[Sequence[str]] = []
    skipped_tools: list[str] = []

    for tool_name, command in _DEFAULT_SECURITY_COMMANDS:
        if shutil.which(tool_name) is None:
            skipped_tools.append(tool_name)
            continue
        commands_to_run.append(tuple(command))

    if additional_commands:
        commands_to_run.extend(additional_commands)

    if not commands_to_run:
        missing = ", ".join(skipped_tools) if skipped_tools else "security tools"
        print(
            "No security tooling could be executed. Install one of the default "
            f"tools ({missing}) or provide custom commands."
        )
        raise SystemExit(1)

    for command in commands_to_run:
        _run_command(command)

    if skipped_tools:
        skipped_display = ", ".join(skipped_tools)
        print(
            "Warning: Skipped security tools that were not installed: "
            f"{skipped_display}."
        )

    print("Security checks passed.")
