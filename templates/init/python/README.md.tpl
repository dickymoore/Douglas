# ${project_name}

This project was bootstrapped with [Douglas](${douglas_readme_url}).

## Quickstart

1. Verify the Codex CLI is up to date (`codex --version` should report `codex-cli 0.40.0` or newer); upgrade with `pip install -U codex-cli` if needed.
2. Run `codex login` to authenticate via your browser (Codex is the default provider), then optionally confirm the session with `codex exec "echo Codex CLI ready"`.
3. (Optional) Copy `.env.example` to `.env` if you plan to provide tokens manually.
4. Create a virtual environment with `make venv`.
5. Run the tests with `make test`.
6. Update `system_prompt.md` with the product vision and early hypotheses; Sprint Zero planning will read it to draft epics/features/stories/tasks.
7. (Optional) If you already have a backlog, populate `ai-inbox/backlog/pre-features.yaml`; otherwise Douglas will generate one during Sprint Zero.
8. Start iterating with `douglas run` once you're ready. Each loop runs a daily standup snapshot, refines the backlog, and then executes the engineering steps.

### Before you run the loop

- Ensure the `douglas` CLI is available in this shell. The project virtualenv only installs app dependencies, so either reuse the environment that ran `douglas init` or install Douglas into this venv (`pip install -e /path/to/Douglas`).
- Verify your Codex CLI is at least version 0.40 (`codex --version`) and can run non-interactively (`codex exec "echo Codex CLI ready"`); upgrade with `pip install -U codex-cli` and re-run `codex login` if either check fails.
- Initialise Git (`git init`) if you didn't supply `--git` during scaffolding so Douglas can inspect history and manage commits, then capture the scaffold as your first commit (`git add . && git commit -m "Initial scaffold"`).
- The default pipelines expect `ruff`, `black`, `isort`, `mypy`, and the OpenAI SDK. They are included in `requirements-dev.txt`; rerun `make venv` after pulling updates, or install them manually inside the active venv. If a tool still reports missing, run `pip install ruff black isort mypy openai` while the venv is active.
- Install provider SDKs (`pip install openai`, etc.) if you plan to use them directly and export the corresponding API keys. Running `python -c "import openai"` inside the venv is a quick sanity check.
- Sprint Zero planning is enabled by default (`planning.enabled: true` in `douglas.yaml`) and runs every loop to keep the backlog fresh. Set `sprint_zero_only: true` or disable planning entirely if you prefer manual backlog grooming.
- Douglas records standup notes under `ai-inbox/sprints/` each iteration. Review these alongside retrospective and demo artifacts to track progress.

Refer to the Douglas README for detailed documentation and advanced workflows.
