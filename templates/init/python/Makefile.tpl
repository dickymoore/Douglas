.PHONY: venv test

venv:
python -m venv .venv
MERGE_CONFLICT< codex/implement-bootstrapping-command-and-readme-update
. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements-dev.txt
. .venv/bin/activate && pip install -e .
MERGE_CONFLICT=
. .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements-dev.txt && \
    pip install -e .
MERGE_CONFLICT> main

test:
. .venv/bin/activate && pytest -q
