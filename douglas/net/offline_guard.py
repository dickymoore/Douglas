"""Network guard that enforces offline execution when requested."""

from __future__ import annotations

import os
import socket
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Optional

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - fallback when requests is unavailable
    requests = None  # type: ignore


class DouglasOfflineError(RuntimeError):
    """Raised when a network operation is attempted in offline mode."""

    def __init__(self, address: Optional[tuple[str, int] | str] = None) -> None:
        target = ""
        if isinstance(address, tuple):
            host, port = address[0], address[1]
            target = f" to {host}:{port}"
        elif isinstance(address, str):
            target = f" to {address}"
        super().__init__(
            "Outbound network access is disabled by DOUGLAS_OFFLINE=1"
            f"{target}."
        )


_GUARD_LOCK = threading.Lock()
_GUARD_ACTIVE = False
_ORIGINAL_SOCKET: Optional[type[socket.socket]] = None
_ORIGINAL_CREATE_CONNECTION: Optional[Callable[..., socket.socket]] = None
_ORIGINAL_SESSION_INIT: Optional[Callable[..., None]] = None


def _normalize_flag(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _guarded_create_connection(*args, **kwargs):
    raise DouglasOfflineError(args[0] if args else None)


class _GuardedSocket(socket.socket):
    def connect(self, address):  # type: ignore[override]
        raise DouglasOfflineError(address)


class _OfflineHTTPAdapter:
    """Requests transport adapter that blocks all outbound requests."""

    def __init__(self) -> None:
        self.timestamp = datetime.utcnow().isoformat()

    def send(self, request, *args, **kwargs):  # pragma: no cover - exercised indirectly
        raise DouglasOfflineError(request.url)

    def close(self) -> None:  # pragma: no cover - compatibility hook
        return None


def _install_requests_guard() -> None:
    if not requests:  # pragma: no cover - requests not installed
        return

    global _ORIGINAL_SESSION_INIT
    if _ORIGINAL_SESSION_INIT is not None:
        return

    adapter = _OfflineHTTPAdapter()
    Session = requests.sessions.Session
    original_init = Session.__init__

    def guarded_init(self, *args, **kwargs):  # type: ignore[override]
        original_init(self, *args, **kwargs)
        self.mount("http://", adapter)
        self.mount("https://", adapter)

    Session.__init__ = guarded_init
    _ORIGINAL_SESSION_INIT = original_init


def activate_offline_guard() -> None:
    """Activate offline network guard for the current process."""

    global _GUARD_ACTIVE, _ORIGINAL_SOCKET, _ORIGINAL_CREATE_CONNECTION

    with _GUARD_LOCK:
        if _GUARD_ACTIVE:
            return

        _ORIGINAL_SOCKET = socket.socket
        _ORIGINAL_CREATE_CONNECTION = getattr(socket, "create_connection", None)
        socket.socket = _GuardedSocket  # type: ignore[assignment]
        if _ORIGINAL_CREATE_CONNECTION is not None:
            socket.create_connection = _guarded_create_connection  # type: ignore[assignment]

        _install_requests_guard()

        _GUARD_ACTIVE = True


def deactivate_offline_guard() -> None:
    """Restore networking primitives to their original behaviour."""

    global _GUARD_ACTIVE, _ORIGINAL_SOCKET, _ORIGINAL_CREATE_CONNECTION, _ORIGINAL_SESSION_INIT

    with _GUARD_LOCK:
        if not _GUARD_ACTIVE:
            return

        if _ORIGINAL_SOCKET is not None:
            socket.socket = _ORIGINAL_SOCKET  # type: ignore[assignment]
        if _ORIGINAL_CREATE_CONNECTION is not None:
            socket.create_connection = _ORIGINAL_CREATE_CONNECTION  # type: ignore[assignment]

        if requests and _ORIGINAL_SESSION_INIT is not None:
            requests.sessions.Session.__init__ = _ORIGINAL_SESSION_INIT  # type: ignore[assignment]

        _ORIGINAL_SOCKET = None
        _ORIGINAL_CREATE_CONNECTION = None
        _ORIGINAL_SESSION_INIT = None

        _GUARD_ACTIVE = False


@contextmanager
def offline_guard():
    """Context manager that temporarily enforces offline mode."""

    activate_offline_guard()
    try:
        yield
    finally:
        deactivate_offline_guard()


def ensure_offline_guard() -> None:
    """Activate the guard if the DOUGLAS_OFFLINE environment flag is set."""

    flag = os.getenv("DOUGLAS_OFFLINE", "")
    if flag and _normalize_flag(flag):
        activate_offline_guard()

