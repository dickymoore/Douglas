"""Typer CLI wiring for the Douglas orchestrator."""

from __future__ import annotations

from copy import deepcopy
import importlib.resources as resources
from pathlib import Path
from string import Template
from typing import Callable, Optional

import typer
import yaml

from douglas.core import Douglas

app = typer.Typer(help="AI-assisted development loop orchestrator")


def _load_default_init_config() -> dict:
    """Load the seed configuration used when bootstrapping a new project."""

    try:
        template_path = resources.files("douglas") / "templates" / "douglas.yaml.tpl"
        template_text = template_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        # Fallback to a minimal configuration when the template file is missing or cannot be read due to I/O errors.
        return {
            "project": {"language": "python"},
            "ai": {"provider": "openai", "model": "gpt-4"},
            "history": {
                "max_log_excerpt_length": Douglas.MAX_LOG_EXCERPT_LENGTH,
            },
        }

    rendered = Template(template_text).safe_substitute(PROJECT_NAME="DouglasProject")
    config = yaml.safe_load(rendered) or {}

    history_cfg = config.setdefault("history", {})
    history_cfg.setdefault(
        "max_log_excerpt_length", Douglas.MAX_LOG_EXCERPT_LENGTH
    )

    return config


def _create_orchestrator(
    config_path: Optional[Path], *, default_config_factory: Optional[Callable[[], dict]] = None
) -> Douglas:
    """Instantiate the Douglas orchestrator using an optional config override."""
    if config_path is not None:
        return Douglas(config_path=config_path)

    inferred_path = Path("douglas.yaml")
    if inferred_path.exists():
        return Douglas(inferred_path)
    if default_config_factory is None:
        raise FileNotFoundError(
            "No douglas.yaml configuration file found. Run `douglas init <project-name>` to create a new project or ensure you are in a valid Douglas project directory."
        )

    config_data = default_config_factory()
    if config_data is None:
        raise FileNotFoundError(
            f"No configuration file found at '{inferred_path}' and the default configuration factory returned None."
        )

    return Douglas(
        config_path=inferred_path,
        config_data=deepcopy(config_data),
    )


@app.command()
def run(
    config: Optional[Path] = typer.Option(
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


def _create_orchestrator(
    config_path: Optional[Path], *, allow_missing_config: bool = False
) -> Douglas:
    """Instantiate the Douglas orchestrator using an optional config override."""
    if config_path is not None:
        return Douglas(config_path=config_path)

    inferred_path = Path("douglas.yaml")
    if inferred_path.exists():
        return Douglas(config_path=inferred_path)

    if allow_missing_config:
        return Douglas(config_path=inferred_path, config={})

    return Douglas(config_path=inferred_path)


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

    orchestrator = _create_orchestrator(
        config,
        default_config_factory=_load_default_init_config,
    )
    orchestrator = _create_orchestrator(config, allow_missing_config=True)
    orchestrator.init_project(project_name, non_interactive=non_interactive)


def main() -> None:
    """Entry point used by the console script."""
    app()


if __name__ == "__main__":
    main()
