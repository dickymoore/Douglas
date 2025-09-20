# Douglas

Douglas is a developer lifecycle companion that automates an AI‑assisted build, test, review, and iterate loop.
It is provider‑agnostic, config‑first, and can bootstrap new repos that include Douglas alongside your app.

> Status: Douglas is not yet published on PyPI. Install from source as shown below.

## Install from source

```bash
# Clone your repo and enter it
git clone https://github.com/dickymoore/Douglas
cd douglas

# Create a virtualenv and activate it
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Editable install with dev tools
pip install -e .[dev]

# Optional: set up pre-commit hooks
pre-commit install
```

### Run the CLI

If you installed with `pip install -e .`, you should have the `douglas` command:

```bash
douglas --help
douglas check
douglas init --project myapp --non-interactive
douglas run
```

If you prefer not to install, run it directly:

```bash
python cli.py --help
```

Or via `pipx` from the repo root:

```bash
pipx install .
douglas --help
```

## Features

- AI‑assisted code generation through a provider interface
- Configurable workflow defined in `douglas.yaml` (plan, generate, lint, typecheck, test, review, commit, pr)
- Provider‑agnostic design with pluggable adapters
- Templates and bootstrap via `douglas init`
- Local‑first CI strategy with pre‑commit and incremental checks

## Quickstart

1) Initialize a new project:

```bash
douglas init --project myapp
cd myapp
```

2) Run the AI loop:

```bash
douglas run
```

This executes the steps defined in `douglas.yaml`.

## Commands

- `douglas init`  Scaffold a new repository with Douglas configuration and templates
- `douglas run`   Execute the AI‑assisted development loop according to your config
- `douglas check` Validate configuration and environment
- `douglas doctor` Diagnose environment and tool availability

## Configuration (`douglas.yaml`)

Example:

```yaml
project:
  name: "myapp"
  description: "My project description"
  license: "MIT"
  language: "python"
ai:
  provider: "openai"
  model: "gpt-4"
  prompt: "system_prompt.md"
loop:
  steps:
    - generate
    - lint
    - typecheck
    - test
    - review
    - commit
    - pr
  exit_conditions:
    - "tests_pass"
vcs:
  default_branch: "main"
  conventional_commits: true
ci:
  local_runner: true
  github_actions: true
  cache:
    pip: true
paths:
  app_src: "src"
  tests: "tests"
```

## Architecture Overview

```
Douglas CLI
   └── douglas-core (orchestrates AI loop)
           ├── providers (LLM adapters)
           ├── pipelines (lint, test, build steps)
           └── integrations (git, GitHub)
   └── templates (project/file scaffolding)
```

## Notes on distribution

- When you are ready to publish to PyPI, update the README to add `pip install douglas` (or your chosen package name) and verify the console script entry point in `pyproject.toml`.
- If the name `douglas` is not available on PyPI, choose an alternative package name and update `pyproject.toml` and docs accordingly.
