# Douglas

**Douglas** is a developer lifecycle companion that automates an AI-assisted build-test-review-iterate loop for your projects. It provides a configurable pipeline that can generate code, run linters/tests, and assist in code reviews, minimizing CI time and manual effort.

## Features

- **AI-assisted code generation:** Integrate LLMs (e.g., OpenAI, Anthropic) to help generate code based on project tasks and prompts.
- **Configurable workflow:** Define a customizable sequence of steps (plan, generate, lint, test, review, commit, pull-request, etc.) in `douglas.yaml`.
- **Provider-agnostic design:** Swap out LLM providers or tooling with minimal changes via adapter interfaces.
- **Templates and bootstrap:** Quickly scaffold new projects with Douglas built-in using `douglas init`.
- **Local-first CI strategy:** Use pre-commit hooks, lint checks, and incremental testing to catch issues before pushing, saving CI minutes.

## Quickstart

1. **Install Douglas**  
   ```sh
   pip install douglas
   ```

2. **Initialize a new project**  
   ```sh
   douglas init --project myapp
   cd myapp
   ```

3. **Run the AI loop**  
   ```sh
   douglas run
   ```
   This will execute the loop steps defined in `douglas.yaml`.

4. **Commit and push changes**  
   By default, Douglas will create commits and pull requests automatically based on loop outcomes.

## Commands

- `douglas init`: Scaffold a new repository with Douglas configuration and templates.  
- `douglas run`: Execute the AI-assisted development loop as per configuration.  
- `douglas check`: Validate the current setup and configuration.  
- `douglas doctor`: Diagnose environment, tool availability, and dependencies.  

## Configuration (`douglas.yaml`)

Douglas uses a YAML configuration file (`douglas.yaml`) for project settings. Example sections:

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

For detailed docs, see CONTRIBUTING.md and the `docs/` directory.

## Architecture Overview

```
Douglas CLI
   └── douglas-core (orchestrates AI loop)
           ├── providers (LLM adapters)
           ├── pipelines (lint, test, build steps)
           └── integrations (git, GitHub)
   └── templates (project/file scaffolding)
```
