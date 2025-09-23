"""Utilities for reading and evaluating the user-controlled run state."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Mapping, Union

__all__ = ["RunState", "read_run_state", "should_exit_now"]


class RunState(Enum):
    """Enumerated run states supported by Douglas."""

    CONTINUE = "CONTINUE"
    SOFT_STOP = "SOFT_STOP"
    HARD_STOP = "HARD_STOP"

    @classmethod
    def from_string(cls, raw: str) -> "RunState":
        """Normalize a string into a :class:`RunState` value.

        Unknown values default to :pydata:`RunState.CONTINUE`.
        """

        normalized = (raw or "").strip().upper()
        for state in cls:
            if normalized in {state.name, state.value}:
                return state
        return cls.CONTINUE


def read_run_state(path: Union[str, Path, None]) -> RunState:
    """Return the desired run state recorded at ``path``.

    Missing files, unreadable paths, or unknown content all default to
    :pydata:`RunState.CONTINUE` so Douglas continues operating.
    """

    if not path:
        return RunState.CONTINUE

    run_state_path = Path(path)
    try:
        raw_text = run_state_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return RunState.CONTINUE
    except OSError:
        return RunState.CONTINUE

    return RunState.from_string(raw_text)


def should_exit_now(state: RunState, context: Mapping[str, object]) -> bool:
    """Determine whether Douglas should exit immediately.

    Parameters
    ----------
    state:
        The run-state directive requested by the user.
    context:
        Additional hints about the current execution point. Recognized keys:

        ``allow_soft_stop_exit`` (:class:`bool`)
            When ``True`` a :pydata:`RunState.SOFT_STOP` (or a previously
            observed soft stop) should cause an immediate, graceful exit.
        ``soft_stop_pending`` (:class:`bool`)
            Indicates whether a soft stop directive has already been observed
            earlier in the run. Used alongside ``allow_soft_stop_exit`` to
            ensure the loop exits after finishing the current sprint.
    """

    if state is RunState.HARD_STOP:
        return True

    allow_soft_exit = bool(context.get("allow_soft_stop_exit", False))
    soft_stop_pending = bool(context.get("soft_stop_pending", False))

    if state is RunState.SOFT_STOP:
        return allow_soft_exit

    if soft_stop_pending and allow_soft_exit:
        return True

    return False
