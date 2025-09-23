import subprocess
from typing import Iterable, Optional, Sequence


def _run_command(command: Sequence[str]) -> None:
    """Run a single type-check command, exiting with an error on failure."""

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(f"Required type checker '{command[0]}' is not installed or not on PATH.")
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        cmd_display = " ".join(command)
        exit_code = exc.returncode or 1
        print(f"Type-check command '{cmd_display}' failed with exit code {exit_code}.")
        raise SystemExit(exit_code)


def run_typecheck(
    additional_commands: Optional[Iterable[Sequence[str]]] = None,
) -> None:
    """Execute type-check commands and exit with a non-zero code when they fail."""

    commands: list[Sequence[str]] = [
        ["mypy", "."],
    ]

    if additional_commands:
        commands.extend(additional_commands)

    for command in commands:
        _run_command(command)

    print("Type checks passed.")
