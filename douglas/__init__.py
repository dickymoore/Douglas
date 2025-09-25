"""Top-level package for Douglas."""

from __future__ import annotations

from importlib import metadata


try:
    __version__: str = metadata.version("douglas")
except metadata.PackageNotFoundError:  # pragma: no cover - runtime fallback during dev
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
