"""Logging utilities for Douglas."""

from __future__ import annotations

import logging
import os
from typing import Optional

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[str] = None) -> None:
    """Initialise Douglas logging if it hasn't already been configured."""

    global _CONFIGURED

    if _CONFIGURED:
        if level:
            _set_level(level)
        return

    log_level = level or os.getenv("DOUGLAS_LOG_LEVEL", "INFO")
    numeric_level = _as_level(log_level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))

    logger = logging.getLogger("douglas")
    logger.setLevel(numeric_level)
    logger.addHandler(handler)
    logger.propagate = True

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger under the Douglas namespace."""

    configure_logging()  # Ensure handlers exist even if user forgot to configure explicitly
    if name.startswith("douglas."):
        qualified = name
    else:
        qualified = f"douglas.{name}"
    return logging.getLogger(qualified)


def _as_level(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    return getattr(logging, str(value).upper(), logging.INFO)


def _set_level(level: str) -> None:
    numeric_level = _as_level(level)
    logging.getLogger("douglas").setLevel(numeric_level)
