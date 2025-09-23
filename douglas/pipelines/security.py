"""Security pipeline integration for Bandit, Semgrep, and custom commands."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Union

__all__ = [
    "SecurityCheckError",
    "SecurityConfigurationError",
    "SecurityReport",
    "SecurityToolResult",
    "run_security",
]

ToolEntry = Union[str, Sequence[str], Mapping[str, object]]

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

    def tool_names(self) -> list[str]:
        return [result.name for result in self.results]

class SecurityConfigurationError(ValueError): ...
class SecurityCheckError(RuntimeError): ...

@dataclass(slots=True)
class _SecurityToolSpec:
    name: str
    command: list[str]

def run_security(
    tools: Optional[Iterable[ToolEntry]] = None,
    *,
    default_paths: Optional[Iterable[str]] = None,
) -> SecurityReport:
    specs = _normalise_tools(tools, default_paths)
    results: list[SecurityToolResult] = []
    for spec in specs:
        results.append(_run_tool(spec))
    print(f"Security checks completed: {', '.join(r.name for r in results)}.")
    return SecurityReport(results)

def _normalise_tools(
    tools: Optional[Iterable[ToolEntry]], default_paths: Optional[Iterable[str]]
) -> list[_SecurityToolSpec]:
    entries: list[ToolEntry] = list(tools) if tools is not None else ["bandit"]

    prepared_paths = _prepare_default_paths(default_paths)

    specs: list[_SecurityToolSpec] = []
    for entry in entries:
        specs.append(_normalise_single_entry(entry, prepared_paths))

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
    entry: ToolEntry, default_paths: list[str]
) -> _SecurityToolSpec:
    if isinstance(entry, str):
        return _spec_from_name(entry, default_paths, None, None)

    if isinstance(entry, Mapping):
        return _spec_from_mapping(entry, default_paths)

    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
        command = [str(token) for token in entry]
        if not command:
            raise SecurityConfigurationError(
                "Security tool command sequences cannot be empty."
            )
        name = command[0]
        return _SecurityToolSpec(name=name, command=command)

    raise SecurityConfigurationError(
        f"Unsupported security tool specification: {entry!r}"
    )


def _spec_from_mapping(
    entry: Mapping[str, object], default_paths: list[str]
) -> _SecurityToolSpec:
    command_value = entry.get("command")
    name_value = entry.get("name") or entry.get("tool")

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
        return _SecurityToolSpec(name=name, command=command)

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
    else:
        args = args_value  # type: ignore[assignment]

    paths: Optional[Iterable[object]]
    if paths_value is None:
        paths = None
    elif isinstance(paths_value, (str, bytes, bytearray)):
        paths = [paths_value]
    elif isinstance(paths_value, Iterable):
        paths = paths_value
    else:
        raise SecurityConfigurationError(
            "Security tool 'paths' or 'targets' must be an iterable of path-like objects."
        )
    return _spec_from_name(name_value, default_paths, args, paths)


def _spec_from_name(
    name_value: object,
    default_paths: list[str],
    args: Optional[Iterable[object]],
    paths: Optional[Iterable[object]],
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

    return _SecurityToolSpec(name=name, command=command)


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
        message = (
            f"Security tool '{spec.command[0]}' is not installed or not on PATH."
        )
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