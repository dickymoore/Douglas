"""Typer CLI wiring for the Douglas orchestrator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import typer

from douglas import __version__
from douglas.logging_utils import configure_logging
from douglas.core import Douglas, TEMPLATE_ROOT
from douglas.providers.claude_code_provider import ClaudeCodeProvider
from douglas.providers.codex_provider import CodexProvider
from douglas.providers.copilot_provider import CopilotProvider
from douglas.providers.gemini_provider import GeminiProvider
from douglas.providers.openai_provider import OpenAIProvider

app = typer.Typer(help="AI-assisted development loop orchestrator")


def _version_callback(value: bool) -> None:
    """Print the Douglas package version when requested."""

    if value:
        typer.echo(f"Douglas {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the Douglas version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    log_level: str = typer.Option(
        "",
        "--log-level",
        help="Set Douglas log level (e.g. info, warning, debug). Overrides DOUGLAS_LOG_LEVEL.",
    ),
) -> None:
    """Global callback to wire shared options like --version."""

    configure_logging(log_level or None)

    return None


_PROVIDER_DEFAULT_MODELS = {
    "codex": CodexProvider.DEFAULT_MODEL,
    "openai": OpenAIProvider.DEFAULT_MODEL,
    "claude_code": ClaudeCodeProvider.DEFAULT_MODEL,
    "claude": ClaudeCodeProvider.DEFAULT_MODEL,
    "gemini": GeminiProvider.DEFAULT_MODEL,
    "copilot": CopilotProvider.DEFAULT_MODEL,
}


def _load_default_init_config() -> dict:
    return deepcopy(Douglas.load_scaffold_config())


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


def _determine_orchestrator_config(
    config_path: Optional[Path],
    *,
    default_config: Optional[dict] = None,
    default_config_factory: Optional[Callable[[], dict]] = None,
    allow_missing_config: bool = False,
) -> tuple[Path, Optional[dict]]:
    """Resolve the path and configuration data used to instantiate ``Douglas``."""

    if default_config is not None and default_config_factory is not None:
        raise ValueError(
            "Provide only one of default_config or default_config_factory when "
            "determining orchestrator configuration."
        )

    if config_path is not None:
        return config_path, None

    inferred_path = Path("douglas.yaml")
    if inferred_path.exists():
        return inferred_path, None

    if default_config is not None:
        return inferred_path, deepcopy(default_config)

    if default_config_factory is not None:
        try:
            config_data = default_config_factory()
        except Exception as exc:  # pragma: no cover - defensive failure mode
            raise FileNotFoundError(
                "No configuration file found at "
                f"'{inferred_path}' and the default configuration factory raised "
                f"{exc.__class__.__name__}: {exc}."
            ) from exc

        if config_data is None:
            raise FileNotFoundError(
                f"No configuration file found at '{inferred_path}' and the "
                "default configuration factory returned None."
            )

        return inferred_path, config_data

    if allow_missing_config:
        return inferred_path, {}

    raise FileNotFoundError(
        "No douglas.yaml configuration file found. Run `douglas init <project-name>` "
        "to create a new project or ensure you are in a valid Douglas project "
        "directory."
    )


def _create_orchestrator(
    config_path: Optional[Path],
    *,
    default_config: Optional[dict] = None,
    default_config_factory: Optional[Callable[[], dict]] = None,
    allow_missing_config: bool = False,
) -> Douglas:
    """Instantiate the Douglas orchestrator using optional config overrides."""

    resolved_path, config_data = _determine_orchestrator_config(
        config_path,
        default_config=default_config,
        default_config_factory=default_config_factory,
        allow_missing_config=allow_missing_config,
    )

    if config_data is None:
        return Douglas(config_path=resolved_path)

    return Douglas(config_path=resolved_path, config_data=config_data)


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
    path: Optional[Path] = typer.Argument(
        None,
        help="Target directory for the new project (defaults to current working directory).",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Project display name to embed in generated files (defaults to directory name).",
    ),
    template: str = typer.Option(
        "python",
        "--template",
        "-t",
        help="Project template to scaffold (python or blank).",
    ),
    provider: str = typer.Option(
        "codex",
        "--provider",
        help="Default AI provider to configure for the generated project.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model identifier to record for the default AI provider.",
    ),
    push_policy: str = typer.Option(
        "per_feature",
        "--push-policy",
        help="Push policy to encode in the generated douglas.yaml.",
    ),
    sprint_length: int = typer.Option(
        Douglas.DEFAULT_SPRINT_LENGTH_DAYS,
        "--sprint-length",
        help="Sprint length (in iterations) to record in the generated douglas.yaml.",
    ),
    ci: str = typer.Option(
        "github",
        "--ci",
        help="Continuous integration workflow to generate (github or none).",
    ),
    git: bool = typer.Option(
        False,
        "--git/--no-git",
        help="Initialise a git repository and create the first commit after scaffolding.",
    ),
    license_: str = typer.Option(
        "none",
        "--license",
        help="License file to include in the scaffold (mit or none).",
    ),
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
        allow_missing_config=True,
    )
    target_dir = path if path is not None else Path(".")
    template_choice = template.strip().lower()
    push_policy_choice = push_policy.strip().lower()
    ci_choice = ci.strip().lower()
    license_choice = license_.strip().lower()
    provider_choice = (provider or "codex").strip().lower() or "codex"
    model_choice = model
    if model_choice is None:
        model_choice = _PROVIDER_DEFAULT_MODELS.get(provider_choice)

    orchestrator.init_project(
        target_dir,
        name=name,
        template=template_choice,
        push_policy=push_policy_choice,
        sprint_length=sprint_length,
        ci=ci_choice,
        git=git,
        license_type=license_choice,
        non_interactive=non_interactive,
        ai_provider=provider_choice,
        ai_model=model_choice,
    )


def main() -> None:
    """Entry point used by the console script."""

    app()


if __name__ == "__main__":
    main()
