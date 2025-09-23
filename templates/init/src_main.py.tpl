from __future__ import annotations


def greet(name: str = "world") -> str:
    """Return a friendly greeting for the $project_name project."""

    return f"Hello, {name}!"


def main() -> None:
    """Entrypoint for the scaffolded $language application."""

    print(greet())


if __name__ == "__main__":
    main()
