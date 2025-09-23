"""Security tooling pipeline for Douglas."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Union

__all__ = [
    "SecurityCheckError",
    "SecurityConfigurationError",
    "SecurityReport",
    "SecurityToolResult",
    "run_security",
]

ToolEntry = Union[str, Sequence[str], Mapping[str, object]]

_DEFAULT_SECURITY_TOOLS: tuple[str, ...] = ("bandit", "semgrep")


@dataclass(slots=True)
class SecurityToolResult:
    name: str
    command: list[str]
    stdout: str
    stderr: str
    exit_code: int


@dataclass(slots=True)
class SecurityReport:
    results: list[SecurityToolResult]
    skipped_tools: list[str] = field(default_factory=list)

    def tool_names(self) -> list[str]:
        return [result.name for result in self.results]


class SecurityConfigurationError(ValueError):
    """Raised when the security pipeline configuration is invalid."""


class SecurityCheckError(SystemExit):
    """Error raised when a security tool fails to execute successfully."""

    def __init__(
        self,
        message: str,
        *,
        tool: str,
        command: Iterable[str],
        exit_code: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ) -> None:
        code = exit_code if exit_code is not None else 1
        super().__init__(code)
        self.message: str = message
        self.tool: str = tool
        self.command: list[str] = list(command)
        self.exit_code: int = code
        self.stdout: Optional[str] = stdout
        self.stderr: Optional[str] = stderr

    def __str__(self) -> str:
        return self.message


@dataclass(slots=True)
class _SecurityToolSpec:
    name: str
    command: list[str]
    optional: bool = False


def run_security(
    tools: Optional[Iterable[ToolEntry]] = None,
    *,
    default_paths: Optional[Iterable[str]] = None,
    additional_commands: Optional[Iterable[Sequence[str]]] = None,
) -> SecurityReport:
    """Run configured security tooling and return an execution report."""

    entries_with_flags: list[tuple[ToolEntry, bool]] = []
    if tools is None:
        entries_with_flags.extend(
            (tool_name, True) for tool_name in _DEFAULT_SECURITY_TOOLS
        )
    else:
        entries_with_flags.extend((entry, False) for entry in tools)

    if additional_commands:
        for command in additional_commands:
            command_seq = list(command)
            if not command_seq:
                raise SecurityConfigurationError(
                    "Security tool command sequences cannot be empty."
                )
            entries_with_flags.append((command_seq, False))

    specs = _normalise_tools(entries_with_flags, default_paths)
    results: list[SecurityToolResult] = []
    skipped_tools: list[str] = []

    for spec in specs:
        if spec.optional:
            command_name = spec.command[0] if spec.command else spec.name
            if shutil.which(command_name) is None:
                skipped_tools.append(spec.name)
                continue
        results.append(_run_tool(spec))

    if not results:
        missing = ", ".join(skipped_tools) if skipped_tools else "security tools"
        message = (
            "No security tooling could be executed. Install one of the default "
            f"tools ({missing}) or provide custom commands."
        )
        raise SecurityCheckError(
            message,
            tool="security",
            command=[],
            exit_code=1,
        )

    if skipped_tools:
        print(
            "Warning: Skipped security tools that were not installed: "
            + ", ".join(skipped_tools)
            + "."
        )

    print(f"Security checks completed: {', '.join(result.name for result in results)}.")
    return SecurityReport(results=results, skipped_tools=skipped_tools)


def _normalise_tools(
    entries_with_flags: Iterable[tuple[ToolEntry, bool]],
    default_paths: Optional[Iterable[str]],
) -> list[_SecurityToolSpec]:
    prepared_paths = _prepare_default_paths(default_paths)

    specs: list[_SecurityToolSpec] = []
    for entry, optional in entries_with_flags:
        specs.append(_normalise_single_entry(entry, prepared_paths, optional))

    if not specs:
        raise SecurityConfigurationError(
            "At least one security tool must be configured for the security pipeline."
        )

    return specs


def _prepare_default_paths(default_paths: Optional[Iterable[str]]) -> list[str]:
    if default_paths is None:
        return ["."]
    paths = [str(path) for path in default_paths if str(path).strip()]
    return paths or ["."]


def _normalise_single_entry(
    entry: ToolEntry, default_paths: list[str], optional_default: bool
) -> _SecurityToolSpec:
    if isinstance(entry, str):
        return _spec_from_name(entry, default_paths, None, None, optional_default)

    if isinstance(entry, Mapping):
        return _spec_from_mapping(entry, default_paths, optional_default)

    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
        command = [str(token) for token in entry]
        if not command:
            raise SecurityConfigurationError(
                "Security tool command sequences cannot be empty."
            )
        name = command[0]
        return _SecurityToolSpec(name=name, command=command, optional=optional_default)

    raise SecurityConfigurationError(
        f"Unsupported security tool specification: {entry!r}"
    )


def _spec_from_mapping(
    entry: Mapping[str, object], default_paths: list[str], optional_default: bool
) -> _SecurityToolSpec:
    command_value = entry.get("command")
    name_value = entry.get("name") or entry.get("tool")
    optional_flag = bool(entry.get("optional", optional_default))

    if command_value is not None:
        if isinstance(command_value, (str, bytes, bytearray)):
            raise SecurityConfigurationError(
                "Security tool 'command' must be a sequence of arguments."
            )
        command = [str(token) for token in command_value]  # type: ignore[arg-type]
        if not command:
            raise SecurityConfigurationError(
                "Security tool command sequences cannot be empty."
            )
        name = str(name_value) if name_value else command[0]
        return _SecurityToolSpec(name=name, command=command, optional=optional_flag)

    if not name_value:
        raise SecurityConfigurationError(
            "Security tool entries must define either 'name' or 'command'."
        )

    args_value = entry.get("args")
    paths_value = entry.get("paths") or entry.get("targets")

    args: Optional[Iterable[object]]
    if args_value is None:
        args = None
    elif isinstance(args_value, (str, bytes, bytearray)):
        args = [args_value]
    elif isinstance(args_value, Iterable) and not isinstance(
        args_value, (str, bytes, bytearray)
    ):
        args = args_value
    else:
        raise SecurityConfigurationError(
            "Security tool 'args' must be an iterable of argument-like objects."
        )

    paths: Optional[Iterable[object]]
    if paths_value is None:
        paths = None
    elif isinstance(paths_value, (str, bytes, bytearray)):
        paths = [paths_value]
    elif isinstance(paths_value, Iterable) and not isinstance(
        paths_value, (str, bytes, bytearray)
    ):
        paths = paths_value
    else:
        raise SecurityConfigurationError(
            "Security tool 'paths' or 'targets' must be an iterable of path-like objects."
        )

    return _spec_from_name(name_value, default_paths, args, paths, optional_flag)


def _spec_from_name(
    name_value: object,
    default_paths: list[str],
    args: Optional[Iterable[object]],
    paths: Optional[Iterable[object]],
    optional: bool,
) -> _SecurityToolSpec:
    name = str(name_value or "").strip()
    if not name:
        raise SecurityConfigurationError("Security tool name cannot be blank.")

    base_command = _default_command_for(name)
    command = list(base_command)

    if args:
        command.extend(str(arg) for arg in args)

    target_paths = _prepare_tool_paths(paths, default_paths)
    command.extend(target_paths)

    return _SecurityToolSpec(name=name, command=command, optional=optional)


def _prepare_tool_paths(
    provided_paths: Optional[Iterable[object]], default_paths: list[str]
) -> list[str]:
    if provided_paths is None:
        return list(default_paths)

    paths = [str(path) for path in provided_paths if str(path).strip()]
    return paths or list(default_paths)


def _default_command_for(name: str) -> list[str]:
    lowered = name.strip().lower()
    if lowered == "bandit":
        return ["bandit", "-q", "-r"]
    if lowered == "semgrep":
        return ["semgrep", "--config", "auto", "--error"]
    raise SecurityConfigurationError(f"Unsupported security tool '{name}'.")


def _run_tool(spec: _SecurityToolSpec) -> SecurityToolResult:
    try:
        completed = subprocess.run(
            spec.command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive logging path
        message = f"Security tool '{spec.command[0]}' is not installed or not on PATH."
        raise SecurityCheckError(
            message,
            tool=spec.name,
            command=spec.command,
            exit_code=1,
            stderr=str(exc),
        ) from exc
    except subprocess.CalledProcessError as exc:
        exit_code = exc.returncode or 1
        stdout = getattr(exc, "stdout", None)
        stderr = getattr(exc, "stderr", None)
        message = f"Security tool '{spec.name}' failed with exit code {exit_code}."
        raise SecurityCheckError(
            message,
            tool=spec.name,
            command=spec.command,
            exit_code=exit_code,
            stdout=stdout or getattr(exc, "output", "") or "",
            stderr=stderr,
        ) from exc

    return SecurityToolResult(
        name=spec.name,
        command=list(spec.command),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        exit_code=completed.returncode or 0,
    )
