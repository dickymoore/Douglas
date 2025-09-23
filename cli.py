"""Backwards-compatible shim that exposes the packaged Douglas CLI."""

from douglas.cli import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
