"""Structured logging helpers for Douglas components."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import ContextDecorator
from dataclasses import dataclass
from logging import Handler, LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
_DEFAULT_BACKUP_COUNT = 5
_LOGGER_NAME = "douglas"
_LOCK = threading.RLock()
_CONFIGURED = False
_FILE_HANDLER: Optional[Handler] = None

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",  # cyan
    "INFO": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[41m",  # red background
}
_RESET_COLOR = "\033[0m"


class DouglasJsonFormatter(logging.Formatter):
    """Custom JSON formatter with Douglas metadata."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"

    def format(self, record: LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        metadata = {}
        if hasattr(record, "metadata") and isinstance(record.metadata, Mapping):
            metadata.update(record.metadata)
        if "extra" in record.__dict__:
            extra = record.__dict__["extra"]
            if isinstance(extra, Mapping) and "metadata" in extra:
                metadata.update(extra["metadata"])
        if metadata:
            payload["metadata"] = metadata
        return json.dumps(payload, default=str, ensure_ascii=False)


class DouglasConsoleFormatter(logging.Formatter):
    """Human-friendly console formatter with colour support."""

    default_time_format = "%H:%M:%S"

    def format(self, record: LogRecord) -> str:  # noqa: D401 - inherited docs
        record.__dict__.setdefault("component", record.name)
        base = super().format(record)
        level = record.levelname
        colour = _LEVEL_COLORS.get(level)
        if not colour or not sys.stdout.isatty():
            return base
        return f"{colour}{base}{_RESET_COLOR}"


def _coerce_level(value: Optional[str]) -> int:
    if not value:
        return logging.INFO
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)
        mapped = getattr(logging, value.upper(), None)
        if isinstance(mapped, int):
            return mapped
    if isinstance(value, int):
        return value
    return logging.INFO


def _create_log_directory() -> Path:
    log_dir = Path(os.getenv("DOUGLAS_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def configure_logging(
    level: Optional[str] = None,
    *,
    log_file: Optional[Path | str] = None,
    enable_json: bool = True,
) -> None:
    """Initialise Douglas logging infrastructure."""

    global _CONFIGURED, _FILE_HANDLER

    with _LOCK:
        resolved_level = _coerce_level(level or os.getenv("DOUGLAS_LOG_LEVEL"))
        logger = logging.getLogger(_LOGGER_NAME)

        if not _CONFIGURED:
            logger.handlers.clear()
            logger.setLevel(resolved_level)
            logger.propagate = False

            console_handler = logging.StreamHandler()
            console_formatter = DouglasConsoleFormatter(
                fmt="%(asctime)s %(levelname)s %(component)s %(message)s",
                datefmt="%H:%M:%S",
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            _CONFIGURED = True
        else:
            logger.setLevel(resolved_level)

        if log_file:
            target_file = Path(log_file)
            target_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            target_file = _create_log_directory() / "douglas.log"

        if _FILE_HANDLER and getattr(_FILE_HANDLER, "baseFilename", None) == str(target_file):
            return

        if _FILE_HANDLER is not None:
            logger.removeHandler(_FILE_HANDLER)
            try:
                _FILE_HANDLER.close()
            finally:
                _FILE_HANDLER = None

        rotation_handler = RotatingFileHandler(
            target_file,
            maxBytes=_DEFAULT_MAX_BYTES,
            backupCount=_DEFAULT_BACKUP_COUNT,
            encoding="utf-8",
        )

        if enable_json:
            formatter: logging.Formatter = DouglasJsonFormatter()
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(component)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        rotation_handler.setFormatter(formatter)
        logger.addHandler(rotation_handler)
        _FILE_HANDLER = rotation_handler


def get_logger(name: str, *, metadata: Optional[Mapping[str, Any]] = None) -> logging.Logger:
    """Return a logger scoped under the Douglas namespace."""

    configure_logging()
    qualified = name if name.startswith(f"{_LOGGER_NAME}.") else f"{_LOGGER_NAME}.{name}"
    logger = logging.getLogger(qualified)
    if metadata:
        return DouglasLoggerAdapter(logger, dict(metadata))
    return logger


class DouglasLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that injects metadata for structured logging."""

    def process(self, msg: str, kwargs: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(kwargs.get("extra", {}))
        metadata = dict(self.extra)
        if "metadata" in kwargs:
            metadata.update(kwargs["metadata"])  # type: ignore[arg-type]
        if metadata:
            extra["metadata"] = metadata
        kwargs["extra"] = extra
        return msg, dict(kwargs)


class log_exceptions(ContextDecorator):
    """Context manager/decorator that logs uncaught exceptions."""

    def __init__(self, logger: logging.Logger, *, message: str = "Unhandled error") -> None:
        self.logger = logger
        self.message = message

    def __enter__(self) -> "log_exceptions":  # noqa: D401 - context protocol
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        if exc_type is not None:
            self.logger.error(
                self.message,
                exc_info=(exc_type, exc_value, exc_traceback),
            )
        return False

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper


def log_action(
    action: str,
    *,
    start_level: int = logging.INFO,
    success_level: int = logging.INFO,
    failure_level: int = logging.ERROR,
    logger_factory: Callable[[], logging.Logger] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that emits structured entry/exit logs around a callable."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_logger = logger_factory() if logger_factory else get_logger(func.__module__)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            func_logger.log(
                start_level,
                "%s:start",
                action,
                extra={"metadata": {"action": action, "event": "start"}},
            )
            try:
                result = func(*args, **kwargs)
            except Exception:
                func_logger.log(
                    failure_level,
                    "%s:error",
                    action,
                    extra={"metadata": {"action": action, "event": "error"}},
                    exc_info=True,
                )
                raise
            duration = time.perf_counter() - start_time
            func_logger.log(
                success_level,
                "%s:success",
                action,
                extra={"metadata": {"action": action, "event": "success", "duration": duration}},
            )
            return result

        return wrapper

    return decorator


@dataclass(slots=True)
class LogRecordBuilder:
    """Helper for building structured log payloads."""

    logger: logging.Logger
    level: int = logging.INFO
    component: Optional[str] = None
    message: str = ""
    metadata: dict[str, Any] | None = None

    def emit(self) -> None:
        payload = self.message
        extra: dict[str, Any] = {}
        if self.component:
            extra.setdefault("component", self.component)
        if self.metadata:
            extra.setdefault("metadata", self.metadata)
        self.logger.log(self.level, payload, extra=extra)


__all__ = [
    "DouglasJsonFormatter",
    "DouglasConsoleFormatter",
    "DouglasLoggerAdapter",
    "LogRecordBuilder",
    "configure_logging",
    "get_logger",
    "log_action",
    "log_exceptions",
]
