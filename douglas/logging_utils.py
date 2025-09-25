"""Logging utilities for Douglas."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_FILE_HANDLER: Optional[logging.Handler] = None
_FILE_HANDLER_PATH: Optional[str] = None


def configure_logging(level: Optional[str] = None, *, log_file: Optional[str] = None) -> None:
    """Initialise Douglas logging and optionally attach a persistent log sink."""

    global _CONFIGURED, _FILE_HANDLER, _FILE_HANDLER_PATH

    requested_level = level or os.getenv("DOUGLAS_LOG_LEVEL", "INFO")
    numeric_level = _as_level(requested_level)

    logger = logging.getLogger("douglas")

    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(numeric_level)
        logger.propagate = True
        _CONFIGURED = True
    else:
        logger.setLevel(numeric_level)

    target_path = log_file or os.getenv("DOUGLAS_LOG_FILE")
    if not target_path:
        return

    if _FILE_HANDLER_PATH == target_path:
        # Already logging to the requested file.
        return

    # Tear down any previous file handler before attaching a new one.
    if _FILE_HANDLER is not None:
        logger.removeHandler(_FILE_HANDLER)
        _FILE_HANDLER.close()
        _FILE_HANDLER = None
        _FILE_HANDLER_PATH = None

    file_path = Path(target_path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file_path, encoding="utf-8")
    except OSError:
        # Silently ignore file handler setup issues; console logging still works.
        return

    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
    logger.addHandler(handler)
    _FILE_HANDLER = handler
    _FILE_HANDLER_PATH = target_path


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger under the Douglas namespace."""

    configure_logging()  # Ensure handlers exist even if user forgot to configure explicitly
    qualified = name if name.startswith("douglas.") else f"douglas.{name}"
    return logging.getLogger(qualified)


def _as_level(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    return getattr(logging, str(value).upper(), logging.INFO)
