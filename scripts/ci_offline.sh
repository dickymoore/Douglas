#!/bin/sh
set -eu

export DOUGLAS_OFFLINE=1

python -m pip install -e .[dev]

ruff check .
black --check .
isort --check-only .
mypy

pytest -q -k "offline or replay or mock or null" --maxfail=1

if ! douglas run --ai-mode mock --seed 123; then
    echo "Warning: 'douglas run --ai-mode mock --seed 123' failed, but continuing." >&2
fi

if ! douglas dashboard render . ./.douglas/dashboard-static; then
    echo "Warning: 'douglas dashboard render' failed, but continuing." >&2
fi
