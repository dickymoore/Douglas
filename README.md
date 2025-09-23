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
- Sprint-aware cadence controls to orchestrate demos, retros, and release steps
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
cadence:
  ProductOwner:
    sprint_review: per_sprint
  ScrumMaster:
    retrospective: per_sprint
loop:
  steps:
    - name: generate
    - name: lint
    - name: typecheck
    - name: test
    - name: retro
      cadence: per_sprint
    - name: demo
      cadence: per_sprint
    - name: commit
    - name: push
    - name: pr
  exit_conditions:
    - "ci_pass"
    - "sprint_demo_complete"
  max_iterations: 3
push_policy: "per_feature"
sprint:
  length_days: 10
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

Exit conditions are evaluated after each loop iteration. Configure
`loop.max_iterations` to cap the number of passes while still exiting early when
conditions like `tests_pass`, `ci_pass`, or `sprint_demo_complete` are satisfied.

### Cadence configuration

The optional top-level `cadence` map assigns scheduling preferences to each
role/activity pair. Values can be simple strings (for example `per_sprint` or
`daily`) or objects with a `frequency` field. When omitted, Douglas falls back to
built-in defaults for common steps.

Each `loop.steps` entry can also declare its own cadence. A step-level cadence
overrides role defaults so you can, for example, run the demo only on the last
day of the sprint:

```yaml
loop:
  steps:
    - name: demo
      cadence: per_sprint
```

### Sprint settings

Set `sprint.length_days` to communicate the length of a sprint to the cadence
engine. The sprint manager tracks the current day, completed features/bugs, and
whether per-sprint pushes or demos have already run. If the length is omitted,
Douglas assumes a ten-day sprint.

### Push/PR policy

Control when Douglas pushes commits or opens pull requests with `push_policy`:

- `per_feature` – push after each feature is finished (default).
- `per_bug` – push once a bug fix is completed.
- `per_epic` – defer until an entire epic is ready.
- `per_sprint` – hold changes locally and push/PR on the final sprint day.

When a policy defers release (for example `per_sprint`), the loop logs why the
push or PR step was skipped so you know the decision was intentional.

The sprint manager also records the cadence decisions in its history so exit
conditions like `sprint_demo_complete` or `push_complete` can end the loop as
soon as the relevant milestones are satisfied.

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
