from __future__ import annotations


def get_welcome_message(name: str = "world") -> str:
    """Return a friendly greeting for ${project_name}."""

    return f"Hello, {name}! Welcome to ${project_name}."
