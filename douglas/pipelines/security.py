"""Security tooling pipeline helpers."""

from __future__ import annotations

import shutil
import subprocess
from typing import Iterable, Optional, Sequence

DEFAULT_COMMANDS: tuple[Sequence[str], ...] = (
    ("bandit", "-q", "-r", "."),
)


def _run_command(command: Sequence[str]) -> None:
    """Run a security tool and surface actionable diagnostics."""

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(
            f"Required security tool '{command[0]}' is not installed or not on PATH."
        )
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
    """Execute security checks such as Bandit and custom commands."""

    commands_to_run: list[Sequence[str]] = []

    for command in DEFAULT_COMMANDS:
        executable = command[0]
        if shutil.which(executable) is None:
            print(
                f"Skipping '{executable}' because it is not installed. "
                "Install the tool to enable this security check."
            )
            continue
        commands_to_run.append(command)

    if additional_commands:
        commands_to_run.extend(additional_commands)

    if not commands_to_run:
        print(
            "No security tools available. Install Bandit or supply custom commands "
            "to enable the security pipeline."
        )
        return

    for command in commands_to_run:
        _run_command(command)

    print("Security checks passed.")

