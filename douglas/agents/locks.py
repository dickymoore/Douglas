"""File lock coordination utilities for Douglas agents."""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from douglas.logging import get_logger


logger = get_logger(__name__)


class FileLockAcquisitionError(RuntimeError):
    """Raised when Douglas cannot obtain a lock within the timeout."""


@dataclass(frozen=True)
class LockHandle:
    """Represents an acquired set of locks."""

    paths: tuple[Path, ...]


class FileLockManager:
    """Manage lockfiles so agents coordinate access to shared resources."""

    def __init__(
        self,
        lock_root: Path | str,
        *,
        default_timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> None:
        self.lock_root = Path(lock_root)
        self.lock_root.mkdir(parents=True, exist_ok=True)
        self.default_timeout = float(default_timeout)
        self.poll_interval = float(poll_interval)

    def _lock_path(self, target: Path) -> Path:
        digest = hashlib.sha256(str(target.resolve()).encode("utf-8")).hexdigest()
        return self.lock_root / f"{digest}.lock"

    @contextmanager
    def acquire(
        self,
        paths: Iterable[Path | str],
        *,
        timeout: float | None = None,
    ) -> Iterator[LockHandle]:
        """Acquire locks for the provided paths, yielding once held."""

        resolved: Sequence[Path] = tuple(sorted(Path(p).resolve() for p in paths))
        if not resolved:
            yield LockHandle(())
            return

        timeout = self.default_timeout if timeout is None else float(timeout)
        start = time.monotonic()
        acquired: list[Path] = []

        try:
            for target in resolved:
                lock_path = self._lock_path(target)
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise FileLockAcquisitionError(
                        f"Timed out acquiring lock for {target} after {timeout:.1f}s",
                    )
                deadline = time.monotonic() + remaining
                while True:
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        try:
                            os.close(fd)
                        finally:
                            pass
                        acquired.append(lock_path)
                        break
                    except FileExistsError:
                        if time.monotonic() >= deadline:
                            raise FileLockAcquisitionError(
                                f"Timed out acquiring lock for {target} after {timeout:.1f}s",
                            )
                        time.sleep(self.poll_interval)
                logger.debug(
                    "Acquired lock for %s",
                    target,
                    extra={"metadata": {"lock": str(target)}},
                )
            yield LockHandle(tuple(resolved))
        finally:
            for lock_path in reversed(acquired):
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
            if acquired:
                logger.debug(
                    "Released locks for %s",
                    ", ".join(str(p) for p in resolved),
                    extra={"metadata": {"locks": [str(p) for p in resolved]}},
                )


__all__ = ["FileLockAcquisitionError", "FileLockManager", "LockHandle"]
