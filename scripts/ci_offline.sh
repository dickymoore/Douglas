#!/bin/sh
set -eu

export DOUGLAS_OFFLINE=1

python -m pip install -e .[dev]

ruff check .
black --check .
isort --check-only .
mypy

pytest -q -k "offline or replay or mock or null" --maxfail=1

douglas run --ai-mode mock --seed 123 || true
douglas dashboard render . ./.douglas/dashboard-static || true
