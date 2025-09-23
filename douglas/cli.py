"""Typer CLI wiring for the Douglas orchestrator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import typer

from douglas.core import Douglas

app = typer.Typer(help="AI-assisted development loop orchestrator")

_DEFAULT_INIT_CONFIG = {
    "project": {"language": "python"},
    "ai": {"provider": "openai"},
    "history": {"max_log_excerpt_length": Douglas.MAX_LOG_EXCERPT_LENGTH},
}


def _create_orchestrator(
    config_path: Optional[Path], *, default_config: Optional[dict] = None
) -> Douglas:
    """Instantiate the Douglas orchestrator using an optional config override."""
    if config_path is not None:
        return Douglas(config_path=config_path)

    inferred_path = Path("douglas.yaml")
    if inferred_path.exists():
        return Douglas(inferred_path)
    elif default_config is not None:
        return Douglas(
            config_path=inferred_path,
            config_data=deepcopy(default_config),
        )
    else:
        raise FileNotFoundError(
            f"Configuration file '{inferred_path}' does not exist and no default configuration was provided."
        )

    return Douglas(
        config_path=inferred_path,
        config_data=deepcopy(default_config),
    )


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to the douglas.yaml configuration file to use.",
    )
) -> None:
    """Execute the configured Douglas development loop."""

    orchestrator = _create_orchestrator(config)
    orchestrator.run_loop()


@app.command()
def check(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to the douglas.yaml configuration file to validate.",
    )
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
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to the douglas.yaml configuration file to seed the scaffold.",
    ),
) -> None:
    """Scaffold a new repository using Douglas templates."""

    orchestrator = _create_orchestrator(
        config,
        default_config=_DEFAULT_INIT_CONFIG,
    )
    orchestrator.init_project(project_name, non_interactive=non_interactive)


def main() -> None:
    """Entry point used by the console script."""

    app()


if __name__ == "__main__":
    main()
