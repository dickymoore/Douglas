"""Typer CLI wiring for the Douglas orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from douglas.core import Douglas

app = typer.Typer(help="AI-assisted development loop orchestrator")


def _config_option(help_text: str) -> Optional[Path]:
    """Shared configuration file option declaration for CLI commands."""

    return typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help=help_text,
    )


def _create_orchestrator(config_path: Optional[Path]) -> Douglas:
    """Instantiate the Douglas orchestrator using an optional config override."""
    if config_path is None:
        return Douglas()
    return Douglas(config_path=str(config_path))


@app.command()
def run(
    config: Optional[Path] = _config_option(
        "Path to the douglas.yaml configuration file to use."
    ),
) -> None:
    """Execute the configured Douglas development loop."""

    orchestrator = _create_orchestrator(config)
    orchestrator.run_loop()


@app.command()
def check(
    config: Optional[Path] = _config_option(
        "Path to the douglas.yaml configuration file to validate."
    ),
) -> None:
    """Validate configuration and environment prerequisites."""

    orchestrator = _create_orchestrator(config)
    orchestrator.check()


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Directory name for the new project."),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Skip prompts when generating project scaffolding.",
    ),
    config: Optional[Path] = _config_option(
        "Path to the douglas.yaml configuration file to seed the scaffold."
    ),
) -> None:
    """Scaffold a new repository using Douglas templates."""

    orchestrator = _create_orchestrator(config)
    orchestrator.init_project(project_name, non_interactive=non_interactive)


def main() -> None:
    """Entry point used by the console script."""

    app()


if __name__ == "__main__":
    main()
